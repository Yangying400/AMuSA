import torch
import numpy as np
import pandas as pd
import os
import random
from AMuSA.trainer import train_signature_classifier, load_model, get_encoded_features
from AMuSA.utils import post_process_predictions, calculate_sample_based_metrics, visualize_results
from AMuSA.data_loader import load_data

def train_ensemble(train_mutation_file, train_exposure_file, test_mutation_file, test_exposure_file, 
                  signature_file, model_dir, n_models=5, encoding_dim=128, epochs=100, batch_size=32, 
                  learning_rate=1e-4, weight_decay=1e-4, verbose=True, exposure_threshold=0.05, 
                  decision_threshold=0.5, optimize_thresholds=True, precision_weight=0.9, 
                  max_active_signatures=6):
    os.makedirs(model_dir, exist_ok=True)
    model_paths = []
    
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}...")
        
        try:
            seed = 42 + i*10
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            model_path = os.path.join(model_dir, f"model_{i+1}.pth")
            metrics_path = os.path.join(model_dir, f"metrics_{i+1}.csv")
            model_paths.append(model_path)
            
            current_lr = learning_rate * (0.8 + 0.4 * np.random.random())
            current_weight_decay = weight_decay * (0.8 + 0.4 * np.random.random())
            current_precision_weight = precision_weight * (0.9 + 0.2 * np.random.random())
            
            _, _, _, _ = train_signature_classifier(
                train_mutation_file=train_mutation_file,
                train_exposure_file=train_exposure_file,
                test_mutation_file=test_mutation_file,
                test_exposure_file=test_exposure_file,
                signature_file=signature_file,
                model_save_path=model_path,
                encoding_dim=encoding_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=current_lr,
                weight_decay=current_weight_decay,
                verbose=verbose,
                metrics_output_path=metrics_path,
                exposure_threshold=exposure_threshold,
                decision_threshold=decision_threshold,
                optimize_thresholds=optimize_thresholds,
                precision_weight=current_precision_weight,
                max_active_signatures=max_active_signatures
            )
            
        except Exception as e:
            print(f"Error training model {i+1}: {e}")
            if model_path in model_paths:
                model_paths.remove(model_path)
            continue
    
    model_paths = [path for path in model_paths if os.path.exists(path)]
    print(f"\nSuccessfully trained {len(model_paths)} models.")
    
    return model_paths

def ensemble_predict(model_paths, mutation_file, threshold=0.6, verbose=True, max_active_signatures=6,
                    signature_file=None, calculate_exposure_values=False, exposure_method='nnls', 
                    min_exposure=0.01):
    if not model_paths:
        print("Warning: No valid model paths provided")
        if calculate_exposure_values:
            return np.array([]), np.array([]), [], [], None
        else:
            return np.array([]), np.array([]), [], []
    
    first_model = torch.load(model_paths[0], map_location='cpu')
    scaler = first_model['scaler']
    signature_names = first_model['signature_names']
    
    X_new, _, _, sample_ids, _ = load_data(
        mutation_file=mutation_file,
        exposure_file=None,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )
    
    all_probs = []
    successful_models = 0
    
    for path in model_paths:
        try:
            classifier, autoencoder, _, _ = load_model(path)
            
            X_new_encoded = get_encoded_features(autoencoder, X_new)
            
            classifier.eval()
            with torch.no_grad():
                probs, _, _ = classifier(torch.FloatTensor(X_new_encoded).to(classifier.thresholds.device))
                all_probs.append(probs.cpu().numpy())
                successful_models += 1
                
        except Exception as e:
            print(f"Error loading model or predicting: {e}")
            continue
    
    if successful_models == 0:
        print("Warning: All models failed")
        if calculate_exposure_values:
            return np.array([]), np.array([]), sample_ids, signature_names, None
        else:
            return np.array([]), np.array([]), sample_ids, signature_names
    
    avg_probs = np.mean(all_probs, axis=0)
    
    if isinstance(threshold, (list, np.ndarray)):
        active_preds = np.zeros_like(avg_probs)
        for i in range(avg_probs.shape[1]):
            active_preds[:, i] = (avg_probs[:, i] >= threshold[i]).astype(float)
    else:
        active_preds = (avg_probs >= threshold).astype(float)
    
    if max_active_signatures is not None:
        active_preds = post_process_predictions(avg_probs, threshold, max_active_signatures)
    
    exposure_values = None
    if calculate_exposure_values and signature_file is not None:
        signature_df = pd.read_csv(signature_file, index_col=0)
        signature_matrix = signature_df.values
        
        mutation_df = pd.read_csv(mutation_file, index_col=0).T
        mutation_data = mutation_df.values
        
        from AMuSA.exposure_calculator import calculate_exposures
        exposure_values = calculate_exposures(
            mutation_data=mutation_data,
            signature_matrix=signature_matrix, 
            active_signatures=active_preds,
            method=exposure_method,
            min_exposure=min_exposure
        )
    
    if verbose:
        for i, sample_id in enumerate(sample_ids):
            active_sigs = [signature_names[j] for j in np.where(active_preds[i] > 0)[0]]
            print(f"Sample {sample_id}: Active signatures: {', '.join(active_sigs) if active_sigs else 'None'}")
    
    if calculate_exposure_values:
        return active_preds, avg_probs, sample_ids, signature_names, exposure_values
    else:
        return active_preds, avg_probs, sample_ids, signature_names

def evaluate_ensemble(model_paths, test_mutation_file, test_exposure_file, signature_file, 
                     output_dir=None, max_active_signatures=6, exposure_threshold=0.05):
    if not model_paths:
        print("Warning: No valid model paths provided")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    first_model = torch.load(model_paths[0], map_location='cpu')
    scaler = first_model['scaler']
    signature_names = first_model['signature_names']
    
    X_test, y_test, _, sample_ids, _ = load_data(
        mutation_file=test_mutation_file,
        exposure_file=test_exposure_file,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )
    
    signature_matrix = None
    mutation_data = None
    if signature_file and os.path.exists(signature_file):
        signature_df = pd.read_csv(signature_file, index_col=0)
        signature_matrix = signature_df.values
        
        mutation_df = pd.read_csv(test_mutation_file, index_col=0).T
        mutation_data = mutation_df.values
    
    if y_test is None:
        print("No exposure data provided. Cannot evaluate model.")
        return None
    
    if np.max(y_test) > 1:
        y_test_binary = (y_test > exposure_threshold).astype(float)
        true_exposures = y_test
    else:
        y_test_binary = y_test
        true_exposures = y_test if signature_matrix is not None else None
    
    all_probs = []
    successful_models = 0
    
    for path in model_paths:
        try:
            classifier, autoencoder, _, _ = load_model(path)
            
            X_test_encoded = get_encoded_features(autoencoder, X_test)
            
            classifier.eval()
            with torch.no_grad():
                probs, _, _ = classifier(torch.FloatTensor(X_test_encoded).to(classifier.thresholds.device))
                all_probs.append(probs.cpu().numpy())
                successful_models += 1
                
        except Exception as e:
            print(f"Error loading model or predicting: {e}")
            continue
    
    if successful_models == 0:
        print("Warning: All models failed")
        return None
    
    avg_probs = np.mean(all_probs, axis=0)
    
    all_thresholds = []
    for path in model_paths:
        try:
            model_dict = torch.load(path, map_location='cpu')
            if 'thresholds' in model_dict:
                all_thresholds.append(model_dict['thresholds'])
        except:
            pass
    
    if all_thresholds:
        avg_thresholds = np.mean(all_thresholds, axis=0)
    else:
        avg_thresholds = np.ones(len(signature_names)) * 0.5
    
    active_preds = np.zeros_like(avg_probs)
    for i in range(avg_probs.shape[1]):
        active_preds[:, i] = (avg_probs[:, i] >= avg_thresholds[i]).astype(float)
    
    if max_active_signatures is not None:
        active_preds = post_process_predictions(avg_probs, avg_thresholds, max_active_signatures)
    
    metrics = calculate_sample_based_metrics(
        y_true=true_exposures if true_exposures is not None else y_test_binary,
        y_pred=active_preds,
        y_pred_probs=avg_probs,
        signature_matrix=signature_matrix,
        mutation_data=mutation_data,
        exposure_threshold=exposure_threshold
    )
    
    print(f"\nEnhanced Ensemble Performance Evaluation (Sample-based):")
    print(f"Sample-based Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    print(f"Sample-based Precision: {metrics['sample_precision']:.4f}")
    print(f"Sample-based Recall: {metrics['sample_recall']:.4f}") 
    print(f"Sample-based F1 Score: {metrics['sample_f1']:.4f}")
    print(f"Sample-based Jaccard Similarity: {metrics['sample_jaccard']:.4f}")
    
    if signature_matrix is not None:
        print(f"Fitting Error: {metrics.get('fitting_error', 0.0):.4f}")
        print(f"Cosine Similarity (Reconstructed Spectra): {metrics.get('cosine_similarity', 0.0):.4f}")
    
    print(f"Traditional AUC: {metrics.get('auc', 0.0):.4f}")
    
    if output_dir:
        visualize_results(
            y_test_binary, active_preds, signature_names, output_dir,
            signature_matrix=signature_matrix, mutation_data=mutation_data
        )
        
        thresholds_df = pd.DataFrame({
            'signature': signature_names,
            'threshold': avg_thresholds
        })
        thresholds_df.to_csv(os.path.join(output_dir, 'ensemble_thresholds.csv'), index=False)
        
        sample_results = pd.DataFrame({
            'sample_id': sample_ids,
            'jaccard_similarity': metrics['sample_metrics']['jaccard'],
            'precision': metrics['sample_metrics']['precision'],
            'recall': metrics['sample_metrics']['recall'],
            'f1_score': metrics['sample_metrics']['f1'],
            'fitting_error': metrics['sample_metrics']['fitting_error'],
            'cosine_similarity': metrics['sample_metrics']['cosine_similarity']
        })
        sample_results.to_csv(os.path.join(output_dir, 'ensemble_sample_metrics.csv'), index=False)
        
        aggregate_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        pd.DataFrame([aggregate_metrics]).to_csv(
            os.path.join(output_dir, 'ensemble_aggregate_metrics.csv'), index=False
        )
        
        model_performances = []
        for i, path in enumerate(model_paths):
            try:
                model_dict = torch.load(path, map_location='cpu')
                if 'metrics' in model_dict:
                    model_metrics = model_dict['metrics']
                    model_metrics['model_id'] = i + 1
                    model_metrics['model_path'] = os.path.basename(path)
                    model_performances.append(model_metrics)
            except:
                pass
        
        if model_performances:
            model_comparison_df = pd.DataFrame(model_performances)
            model_comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    return metrics