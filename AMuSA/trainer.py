import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import time
import random
from AMuSA.models_fix import Autoencoder, SignatureClassifier,FocalLoss
from AMuSA.data_loader import load_data
from AMuSA.utils import device, find_optimal_thresholds, post_process_predictions, evaluate_binary_classification
from AMuSA.utils import plot_metrics, visualize_results, calculate_sample_based_metrics

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    
    # Create autoencoder
    autoencoder = Autoencoder(
        input_dim=checkpoint['scaler'].n_features_in_, 
        encoding_dim=checkpoint['encoding_dim']
    ).to(device)
    
    # Create classifier
    classifier = SignatureClassifier(
        num_signatures=checkpoint['num_signatures'], 
        encoding_dim=checkpoint['encoding_dim'],
        threshold=checkpoint.get('thresholds', 0.5)
    ).to(device)
    
    # Load weights
    autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    return classifier, autoencoder, checkpoint['scaler'], checkpoint['signature_names']

def predict(model_path, mutation_file, signature_file=None, verbose=True, max_active_signatures=7,
           calculate_exposure_values=False, exposure_method='nnls', min_exposure=0.01):
    # Load model
    classifier, autoencoder, scaler, signature_names = load_model(model_path)
    
    # Load mutation data
    X_new, _, _, sample_ids, _ = load_data(
        mutation_file=mutation_file,
        exposure_file=None,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )
    
    # Get encoded features
    X_new_encoded = get_encoded_features(autoencoder, X_new)
    
    # Predict active signatures
    classifier.eval()
    with torch.no_grad():
        probs, active_preds, _ = classifier(torch.FloatTensor(X_new_encoded).to(device))
        
        probs = probs.cpu().numpy()
        active_preds = active_preds.cpu().numpy()
    
    # Apply maximum active signatures limit
    if max_active_signatures is not None:
        active_preds = post_process_predictions(
            probs, 
            classifier.thresholds.cpu().numpy(),
            max_active_signatures
        )
    
    exposure_values = None
    if calculate_exposure_values and signature_file is not None:
        # Load signature matrix for exposure calculation
        signature_df = pd.read_csv(signature_file, index_col=0)
        signature_matrix = signature_df.values
        
        # Load original mutation data for NNLS
        mutation_df = pd.read_csv(mutation_file, index_col=0).T
        mutation_data = mutation_df.values
        
        # Calculate exposures using the selected method
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
        return active_preds, probs, sample_ids, signature_names, exposure_values
    else:
        return active_preds, probs, sample_ids, signature_names

def evaluate_model(model_path, test_mutation_file, test_exposure_file, signature_file, 
                  output_dir=None, max_active_signatures=6, exposure_threshold=0.05):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    classifier, autoencoder, scaler, signature_names = load_model(model_path)
    
    # Load test data
    X_test, y_test, _, sample_ids, _ = load_data(
        mutation_file=test_mutation_file,
        exposure_file=test_exposure_file,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )
    
    # Load signature matrix and mutation data for metrics
    signature_matrix = None
    mutation_data = None
    if signature_file and os.path.exists(signature_file):
        signature_df = pd.read_csv(signature_file, index_col=0)
        signature_matrix = signature_df.values
        
        mutation_df = pd.read_csv(test_mutation_file, index_col=0).T
        mutation_data = mutation_df.values
    
    # Convert exposure values to binary if needed
    if y_test is None:
        print("No exposure data provided. Cannot evaluate model.")
        return None
    
    if np.max(y_test) > 1:
        y_test_binary = (y_test > exposure_threshold).astype(float)
        true_exposures = y_test
    else:
        y_test_binary = y_test
        true_exposures = y_test if signature_matrix is not None else None
    
    # Get encoded features
    X_test_encoded = get_encoded_features(autoencoder, X_test)
    
    # Predict active signatures
    classifier.eval()
    with torch.no_grad():
        probs, active_preds, _ = classifier(torch.FloatTensor(X_test_encoded).to(device))
        
        probs = probs.cpu().numpy()
        active_preds = active_preds.cpu().numpy()
    
    # Apply maximum active signatures limit
    if max_active_signatures is not None:
        active_preds = post_process_predictions(
            probs, 
            classifier.thresholds.cpu().numpy(),
            max_active_signatures
        )
    
    # Calculate sample-based metrics
    metrics = calculate_sample_based_metrics(
        y_true=y_test_binary if true_exposures is None else y_test,
        y_pred=active_preds,
        y_pred_probs=probs,
        signature_matrix=signature_matrix,
        mutation_data=mutation_data,
        exposure_threshold=exposure_threshold
    )
    
    # Print main metrics
    print(f"\nModel Performance Evaluation (Sample-based):")
    print(f"Sample-based Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    print(f"Sample-based Precision: {metrics['sample_precision']:.4f}")
    print(f"Sample-based Recall: {metrics['sample_recall']:.4f}")
    print(f"Sample-based F1 Score: {metrics['sample_f1']:.4f}")
    print(f"Sample-based Jaccard Similarity: {metrics['sample_jaccard']:.4f}")
    
    if signature_matrix is not None:
        print(f"Fitting Error: {metrics.get('fitting_error', 0.0):.4f}")
        print(f"Cosine Similarity (Reconstructed Spectra): {metrics.get('cosine_similarity', 0.0):.4f}")
    
    print(f"Traditional AUC: {metrics.get('auc', 0.0):.4f}")
    
    # Visualization
    if output_dir:
        visualize_results(
            y_test_binary, active_preds, signature_names, output_dir,
            signature_matrix=signature_matrix, mutation_data=mutation_data
        )
        
        # Save detailed sample-based metrics
        sample_results = pd.DataFrame({
            'sample_id': sample_ids,
            'jaccard_similarity': metrics['sample_metrics']['jaccard'],
            'precision': metrics['sample_metrics']['precision'],
            'recall': metrics['sample_metrics']['recall'],
            'f1_score': metrics['sample_metrics']['f1'],
            'fitting_error': metrics['sample_metrics']['fitting_error'],
            'cosine_similarity': metrics['sample_metrics']['cosine_similarity']
        })
        sample_results.to_csv(os.path.join(output_dir, 'sample_based_metrics.csv'), index=False)
        
        # Save aggregate metrics
        aggregate_metrics = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        pd.DataFrame([aggregate_metrics]).to_csv(
            os.path.join(output_dir, 'aggregate_metrics.csv'), index=False
        )
    
    return metrics

def train_autoencoder(X_train, X_val, encoding_dim=128, epochs=100, batch_size=32, learning_rate=1e-3, 
                      weight_decay=1e-4, verbose=True):
    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=verbose)

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(torch.FloatTensor(X_train).to(device))
        loss = criterion(outputs, torch.FloatTensor(X_train).to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        autoencoder.eval()
        with torch.no_grad():
            val_outputs = autoencoder(torch.FloatTensor(X_val).to(device))
            val_loss = criterion(val_outputs, torch.FloatTensor(X_val).to(device))
            val_losses.append(val_loss.item())

        scheduler.step(val_loss)

        if verbose and epoch % 10 == 0:
            print(f"Autoencoder Epoch {epoch}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    return autoencoder

def get_encoded_features(autoencoder, X):
    autoencoder.eval()
    with torch.no_grad():
        encoded = autoencoder.encoder(torch.FloatTensor(X).to(device))
    return encoded.cpu().numpy()

def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is None:
        weights = torch.ones_like(target)
        
    loss = -weights * (target * torch.log(output + 1e-7) + (1 - target) * torch.log(1 - output + 1e-7))
    return torch.mean(loss)

def train_signature_classifier(
    train_mutation_file,
    train_exposure_file,
    test_mutation_file,
    test_exposure_file,
    signature_file,
    model_save_path,
    encoding_dim=128,
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-4,
    verbose=True,
    metrics_output_path=None,
    exposure_threshold=0.05,
    decision_threshold=0.5,
    optimize_thresholds=True,
    precision_weight=0.9,
    max_active_signatures=6
):
    # Load training data
    X_train, y_train, signature_names, train_sample_ids, scaler = load_data(
        mutation_file=train_mutation_file,
        exposure_file=train_exposure_file,
        signature_file=signature_file,
        scaler=None,
        train=True
    )

    # Load test data
    X_test, y_test, _, test_sample_ids, _ = load_data(
        mutation_file=test_mutation_file,
        exposure_file=test_exposure_file,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )

    # Load signature matrix and mutation data for metrics
    signature_matrix = None
    train_mutation_data = None
    test_mutation_data = None
    
    if signature_file and os.path.exists(signature_file):
        signature_df = pd.read_csv(signature_file, index_col=0)
        signature_matrix = signature_df.values
        
        # Load mutation data for reconstruction metrics
        train_mutation_df = pd.read_csv(train_mutation_file, index_col=0).T
        train_mutation_data = train_mutation_df.values
        
        test_mutation_df = pd.read_csv(test_mutation_file, index_col=0).T
        test_mutation_data = test_mutation_df.values

    # Convert exposure values to binary if not already
    if y_train is not None:
        if np.max(y_train) > 1:
            print("Converting exposure values to binary")
            y_train_binary = (y_train > exposure_threshold).astype(float)
            y_test_binary = (y_test > exposure_threshold).astype(float)
            train_exposures = y_train
            test_exposures = y_test
        else:
            y_train_binary = y_train
            y_test_binary = y_test
            train_exposures = None
            test_exposures = None
    else:
        print("No exposure data provided. Cannot train classifier.")
        return None, None, None, None
    
    num_signatures = y_train_binary.shape[1]

    # Train autoencoder
    autoencoder = train_autoencoder(
        X_train,
        X_test,
        encoding_dim=encoding_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate * 2,
        weight_decay=weight_decay,
        verbose=verbose
    )

    # Get encoded features
    X_train_encoded = get_encoded_features(autoencoder, X_train)
    X_test_encoded = get_encoded_features(autoencoder, X_test)

    # Create classifier
    classifier = SignatureClassifier(
        num_signatures=num_signatures, 
        encoding_dim=encoding_dim,
        dropout_rate=0.2,
        threshold=decision_threshold
    ).to(device)

    # Set class weights to handle imbalance
    pos_counts = np.sum(y_train_binary, axis=0)
    total_samples = y_train_binary.shape[0]
    neg_counts = total_samples - pos_counts
    class_ratio = neg_counts / (pos_counts + 1e-5)
    pos_weights = torch.FloatTensor(class_ratio).to(device)
    
    # BCE loss with class weights
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    criterion = FocalLoss(alpha=1, gamma=2)

    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, verbose=verbose
    )

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_encoded),
        torch.FloatTensor(y_train_binary)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test_encoded),
        torch.FloatTensor(y_test_binary)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize metrics tracking
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_sample_precision': [],
        'val_sample_recall': [],
        'val_sample_f1': [],
        'val_sample_jaccard': [],
        'val_fitting_error': [],
        'val_cosine_similarity': [],
        'val_auc': [],
        'learning_rate': []
    }

    best_f1 = 0
    patience_counter = 0
    patience_limit = 15  # Early stopping
    best_thresholds = [decision_threshold] * num_signatures

    for epoch in range(1, epochs + 1):
        # Training phase
        classifier.train()
        train_loss = 0.0

        for batch_encoded_X, batch_y in train_loader:
            batch_encoded_X, batch_y = batch_encoded_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            probs, active_preds, logits = classifier(batch_encoded_X)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * batch_encoded_X.size(0)

        # Calculate average loss
        train_loss /= len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']

        # Validation phase
        classifier.eval()
        val_loss = 0.0
        all_y_true = []
        all_y_pred_probs = []
        all_y_pred_active = []

        with torch.no_grad():
            for batch_encoded_X, batch_y in test_loader:
                batch_encoded_X, batch_y = batch_encoded_X.to(device), batch_y.to(device)
                
                # Forward pass
                probs, active_preds, logits = classifier(batch_encoded_X)
                loss = criterion(logits, batch_y)
                
                # Accumulate loss
                val_loss += loss.item() * batch_encoded_X.size(0)
                
                # Collect predictions and true values
                all_y_true.append(batch_y.cpu().numpy())
                all_y_pred_probs.append(probs.cpu().numpy())
                all_y_pred_active.append(active_preds.cpu().numpy())

        # Calculate average loss
        val_loss /= len(test_loader.dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Combine batches
        all_y_true_np = np.vstack(all_y_true)
        all_y_pred_probs_np = np.vstack(all_y_pred_probs)
        all_y_pred_active_np = np.vstack(all_y_pred_active)

        # Optimize thresholds
        current_thresholds = best_thresholds
        if optimize_thresholds and epoch > epochs // 4:
            optimal_thresholds = find_optimal_thresholds(
                all_y_true_np, 
                all_y_pred_probs_np, 
                init_threshold=decision_threshold,
                precision_weight=precision_weight
            )
            
            # Update classifier thresholds
            classifier.update_thresholds(optimal_thresholds)
            current_thresholds = optimal_thresholds

        # Post-process predictions
        all_y_pred_processed = post_process_predictions(
            all_y_pred_probs_np, 
            current_thresholds, 
            max_active_signatures
        )

        # Calculate sample-based metrics
        val_metrics = calculate_sample_based_metrics(
            y_true=test_exposures if test_exposures is not None else all_y_true_np,
            y_pred=all_y_pred_processed,
            y_pred_probs=all_y_pred_probs_np,
            signature_matrix=signature_matrix,
            mutation_data=test_mutation_data,
            exposure_threshold=exposure_threshold
        )

        # Record metrics
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        metrics['val_sample_precision'].append(val_metrics['sample_precision'])
        metrics['val_sample_recall'].append(val_metrics['sample_recall'])
        metrics['val_sample_f1'].append(val_metrics['sample_f1'])
        metrics['val_sample_jaccard'].append(val_metrics['sample_jaccard'])
        metrics['val_fitting_error'].append(val_metrics.get('fitting_error', 0.0))
        metrics['val_cosine_similarity'].append(val_metrics['cosine_similarity'])
        metrics['val_auc'].append(val_metrics.get('auc', 0.0))
        metrics['learning_rate'].append(current_lr)

        # Check for best model using sample-based F1
        current_f1 = val_metrics['sample_f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            best_thresholds = current_thresholds
            
            # Save model
            model_dict = {
                'classifier_state_dict': classifier.state_dict(),
                'autoencoder_state_dict': autoencoder.state_dict(),
                'scaler': scaler,
                'signature_names': signature_names,
                'num_signatures': num_signatures,
                'encoding_dim': encoding_dim,
                'thresholds': current_thresholds,
                'epoch': epoch,
                'metrics': {
                    'sample_precision': val_metrics['sample_precision'],
                    'sample_recall': val_metrics['sample_recall'],
                    'sample_f1': val_metrics['sample_f1'],
                    'sample_jaccard': val_metrics['sample_jaccard'],
                    'fitting_error': val_metrics.get('fitting_error', 0.0),
                    'cosine_similarity': val_metrics.get('cosine_similarity', 0.0),
                    'accuracy': val_metrics.get('accuracy', 0.0),
                    'auc': val_metrics.get('auc', 0.0)
                }
            }
            
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model_dict, model_save_path)
            
            if verbose:
                print(f"Epoch {epoch}: New best model saved.")
                print(f"  Sample F1: {val_metrics['sample_f1']:.4f}")
                print(f"  Sample Precision: {val_metrics['sample_precision']:.4f}")
                print(f"  Sample Recall: {val_metrics['sample_recall']:.4f}")
                print(f"  Jaccard Similarity: {val_metrics['sample_jaccard']:.4f}")
                if signature_matrix is not None:
                    print(f"  Fitting Error: {val_metrics.get('fitting_error', 0.0):.4f}")
                    print(f"  Cosine Similarity: {val_metrics.get('cosine_similarity', 0.0):.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break

        # Output progress
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Sample Metrics - F1: {val_metrics['sample_f1']:.4f}, "
                  f"Precision: {val_metrics['sample_precision']:.4f}, "
                  f"Recall: {val_metrics['sample_recall']:.4f}")
            if signature_matrix is not None:
                print(f"Reconstruction Metrics - Fitting Error: {val_metrics.get('fitting_error', 0.0):.4f}, "
                      f"Cosine Sim: {val_metrics.get('cosine_similarity', 0.0):.4f}")

    # Save metrics
    if metrics_output_path is not None:
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        pd.DataFrame(metrics).to_csv(metrics_output_path, index=False)
        
        # Save final thresholds
        thresholds_df = pd.DataFrame({
            'signature': signature_names,
            'threshold': best_thresholds
        })
            
        thresholds_path = os.path.join(os.path.dirname(metrics_output_path), 'signature_thresholds.csv')
        thresholds_df.to_csv(thresholds_path, index=False)
        
        if verbose:
            print(f"Metrics saved to {metrics_output_path}")
            print(f"Thresholds saved to {thresholds_path}")

    return classifier, autoencoder, scaler, signature_names
