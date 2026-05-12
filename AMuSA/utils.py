import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from scipy.optimize import nnls
from scipy.spatial.distance import cosine

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_sample_based_metrics(y_true, y_pred, y_pred_probs, signature_matrix=None, 
                                 mutation_data=None, exposure_threshold=0.05):
    n_samples, n_signatures = y_true.shape
    
    if np.max(y_true) > 1:
        y_true_binary = (y_true > exposure_threshold).astype(float)
        true_exposures = y_true
    else:
        y_true_binary = y_true
        true_exposures = None
    
    sample_jaccard = np.zeros(n_samples)
    sample_precision = np.zeros(n_samples)
    sample_recall = np.zeros(n_samples)
    sample_f1 = np.zeros(n_samples)
    sample_cosine_similarity = np.zeros(n_samples)
    fitting_errors = np.zeros(n_samples)
    
    for i in range(n_samples):
        true_active = set(np.where(y_true_binary[i] > 0)[0])
        pred_active = set(np.where(y_pred[i] > 0)[0])
        
        if len(true_active) == 0 and len(pred_active) == 0:
            sample_jaccard[i] = 1.0
        else:
            intersection = len(true_active.intersection(pred_active))
            union = len(true_active.union(pred_active))
            sample_jaccard[i] = intersection / union if union > 0 else 0.0
        
        if len(pred_active) > 0:
            sample_precision[i] = len(true_active.intersection(pred_active)) / len(pred_active)
        else:
            sample_precision[i] = 1.0 if len(true_active) == 0 else 0.0
            
        if len(true_active) > 0:
            sample_recall[i] = len(true_active.intersection(pred_active)) / len(true_active)
        else:
            sample_recall[i] = 1.0 if len(pred_active) == 0 else 0.0
            
        if sample_precision[i] + sample_recall[i] > 0:
            sample_f1[i] = 2 * sample_precision[i] * sample_recall[i] / (sample_precision[i] + sample_recall[i])
        else:
            sample_f1[i] = 0.0
    
    if signature_matrix is not None and mutation_data is not None:
        estimated_exposures = estimate_exposures_from_predictions(
            y_pred, signature_matrix, mutation_data
        )
        
        if true_exposures is not None:
            for i in range(n_samples):
                true_norm = true_exposures[i] / (np.sum(true_exposures[i]) + 1e-10)
                est_norm = estimated_exposures[i] / (np.sum(estimated_exposures[i]) + 1e-10)
                fitting_errors[i] = np.sum(np.abs(true_norm - est_norm)) / 2
        
        if mutation_data is not None:
            reconstructed_spectra = np.dot(estimated_exposures, signature_matrix.T)
            
            for i in range(n_samples):
                original_spectrum = mutation_data[i]
                reconstructed_spectrum = reconstructed_spectra[i]
                
                if np.linalg.norm(original_spectrum) > 0 and np.linalg.norm(reconstructed_spectrum) > 0:
                    sample_cosine_similarity[i] = np.dot(original_spectrum, reconstructed_spectrum) / (
                        np.linalg.norm(original_spectrum) * np.linalg.norm(reconstructed_spectrum)
                    )
                else:
                    sample_cosine_similarity[i] = 0.0
    
    metrics = {
        'sample_jaccard': np.mean(sample_jaccard),
        'sample_precision': np.mean(sample_precision),
        'sample_recall': np.mean(sample_recall),
        'sample_f1': np.mean(sample_f1),
        'fitting_error': np.mean(fitting_errors),
        'cosine_similarity': np.mean(sample_cosine_similarity) if signature_matrix is not None else 0.0,
        
        'signature_precision': precision_score(y_true_binary.flatten(), y_pred.flatten(), zero_division=0),
        'signature_recall': recall_score(y_true_binary.flatten(), y_pred.flatten(), zero_division=0),
        'signature_f1': f1_score(y_true_binary.flatten(), y_pred.flatten(), zero_division=0),
        'accuracy': accuracy_score(y_true_binary.flatten(), y_pred.flatten()),
        
        'sample_metrics': {
            'jaccard': sample_jaccard,
            'precision': sample_precision,
            'recall': sample_recall,
            'f1': sample_f1,
            'fitting_error': fitting_errors,
            'cosine_similarity': sample_cosine_similarity
        }
    }
    
    try:
        metrics['auc'] = roc_auc_score(y_true_binary.flatten(), y_pred_probs.flatten())
    except:
        metrics['auc'] = 0.0
    
    return metrics

def estimate_exposures_from_predictions(y_pred, signature_matrix, mutation_data):
    n_samples, n_signatures = y_pred.shape
    exposures = np.zeros((n_samples, n_signatures))
    
    for i in range(n_samples):
        active_sigs = np.where(y_pred[i] > 0)[0]
        
        if len(active_sigs) > 0:
            active_sig_matrix = signature_matrix[:, active_sigs]
            coeffs, _ = nnls(active_sig_matrix, mutation_data[i])
            exposures[i, active_sigs] = coeffs
    
    return exposures

def find_optimal_thresholds(y_true, y_pred_probs, init_threshold=0.5, precision_weight=0.95):
    num_signatures = y_true.shape[1]
    optimal_thresholds = []
    
    baseline_thresholds = np.ones(num_signatures) * init_threshold
    baseline_pred = (y_pred_probs >= baseline_thresholds).astype(float)
    baseline_metrics = calculate_sample_based_metrics(y_true, baseline_pred, y_pred_probs)
    baseline_f1 = baseline_metrics['sample_f1']
    
    for i in range(num_signatures):
        if np.sum(y_true[:, i]) > 0:
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            
            beta = (1 - precision_weight) / precision_weight
            f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
            
            if len(thresholds) > 0:
                best_idx = np.argmax(f_beta_scores[:-1])
                best_threshold = thresholds[best_idx]
            else:
                best_threshold = init_threshold
        else:
            best_threshold = max(0.8, init_threshold)
        
        optimal_thresholds.append(best_threshold)
    
    return optimal_thresholds

def post_process_predictions(probs, thresholds, max_active_signatures=6):
    if isinstance(thresholds, (list, np.ndarray, torch.Tensor)):
        active_preds = np.zeros_like(probs)
        for i in range(probs.shape[1]):
            active_preds[:, i] = (probs[:, i] >= thresholds[i]).astype(float)
    else:
        active_preds = (probs >= thresholds).astype(float)
    
    if max_active_signatures is not None:
        for i in range(active_preds.shape[0]):
            active_count = np.sum(active_preds[i])
            if active_count > max_active_signatures:
                sorted_indices = np.argsort(-probs[i])
                active_preds[i, :] = 0
                active_preds[i, sorted_indices[:max_active_signatures]] = 1
            elif active_count == 0 and max_active_signatures > 0:
                best_sig = np.argmax(probs[i])
                active_preds[i, best_sig] = 1
    
    return active_preds

def evaluate_binary_classification(y_true, y_pred_probs, threshold=0.4, signature_matrix=None, 
                                 mutation_data=None, exposure_threshold=0.05):
    if isinstance(threshold, (list, np.ndarray, torch.Tensor)):
        y_pred = np.zeros_like(y_pred_probs)
        for i in range(y_pred_probs.shape[1]):
            y_pred[:, i] = (y_pred_probs[:, i] >= threshold[i]).astype(int)
    else:
        y_pred = (y_pred_probs >= threshold).astype(int)
    
    metrics = calculate_sample_based_metrics(
        y_true, y_pred, y_pred_probs, 
        signature_matrix=signature_matrix,
        mutation_data=mutation_data,
        exposure_threshold=exposure_threshold
    )
    
    metrics['precision'] = metrics['sample_precision']
    metrics['recall'] = metrics['sample_recall'] 
    metrics['f1'] = metrics['sample_f1']
    metrics['jaccard_similarity'] = metrics['sample_jaccard']
    
    return metrics

def plot_sample_metrics(metrics, output_dir='plots', prefix=''):
    os.makedirs(output_dir, exist_ok=True)
    
    sample_metrics = metrics.get('sample_metrics', {})
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_names = ['jaccard', 'precision', 'recall', 'f1', 'fitting_error', 'cosine_similarity']
    titles = ['Jaccard Similarity', 'Precision', 'Recall', 'F1 Score', 'Fitting Error', 'Cosine Similarity']
    
    for i, (metric_name, title) in enumerate(zip(metric_names, titles)):
        if metric_name not in sample_metrics:
            axes[i].text(0.5, 0.5, f'{title} not available', ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].set_title(title)
            continue
            
        values = sample_metrics[metric_name]
        
        if len(values) == 0 or np.sum(np.abs(values)) == 0:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(title)
            continue
            
        try:
            axes[i].hist(values, bins=min(20, max(5, len(values)//3)), alpha=0.7, edgecolor='black')
            axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(values):.3f}')
            axes[i].set_xlabel(title)
            axes[i].set_ylabel('Number of Samples')
            axes[i].set_title(f'{title} Distribution')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error plotting\n{title}', ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].set_title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}sample_metrics_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def visualize_results(y_true, y_pred, signature_names, output_dir='results', 
                     signature_matrix=None, mutation_data=None):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        metrics = calculate_sample_based_metrics(
            y_true, y_pred, y_pred, 
            signature_matrix=signature_matrix,
            mutation_data=mutation_data
        )
        
        plot_sample_metrics(metrics, output_dir, 'enhanced_')
    except Exception as e:
        print(f"Warning: Could not calculate or plot sample metrics: {e}")
    
    try:
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Inactive', 'Active'],
                    yticklabels=['Inactive', 'Active'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create confusion matrix: {e}")
    
    try:
        sample_metrics = metrics.get('sample_metrics', {})
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        if 'precision' in sample_metrics and 'recall' in sample_metrics:
            plt.scatter(sample_metrics['precision'], sample_metrics['recall'], alpha=0.6)
            plt.xlabel('Sample Precision')
            plt.ylabel('Sample Recall')
            plt.title('Precision vs Recall (per sample)')
            plt.grid(alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Precision/Recall data not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 2)
        if 'jaccard' in sample_metrics and 'f1' in sample_metrics:
            plt.scatter(sample_metrics['jaccard'], sample_metrics['f1'], alpha=0.6)
            plt.xlabel('Jaccard Similarity')
            plt.ylabel('F1 Score')
            plt.title('Jaccard vs F1 (per sample)')
            plt.grid(alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Jaccard/F1 data not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 3)
        if ('cosine_similarity' in sample_metrics and 'fitting_error' in sample_metrics and 
            np.sum(sample_metrics['cosine_similarity']) > 0):
            plt.scatter(sample_metrics['cosine_similarity'], sample_metrics['fitting_error'], alpha=0.6)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Fitting Error')
            plt.title('Cosine Similarity vs Fitting Error')
            plt.grid(alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Cosine/Fitting data not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 4)
        n_active_true = np.sum(y_true, axis=1)
        n_active_pred = np.sum(y_pred, axis=1)
        plt.scatter(n_active_true, n_active_pred, alpha=0.6)
        plt.xlabel('True Active Signatures')
        plt.ylabel('Predicted Active Signatures')
        plt.title('Active Signature Count Comparison')
        plt.plot([0, max(n_active_true.max(), n_active_pred.max())], 
                 [0, max(n_active_true.max(), n_active_pred.max())], 'r--')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_based_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create sample-based analysis plots: {e}")

def plot_metrics(metrics_file, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    metrics_df = pd.read_csv(metrics_file)
    
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    if 'val_sample_precision' in metrics_df.columns:
        plt.plot(metrics_df['epoch'], metrics_df['val_sample_precision'], label='Sample Precision')
        plt.plot(metrics_df['epoch'], metrics_df['val_sample_recall'], label='Sample Recall')
        plt.plot(metrics_df['epoch'], metrics_df['val_sample_f1'], label='Sample F1')
    else:
        plt.plot(metrics_df['epoch'], metrics_df['val_precision'], label='Precision')
        plt.plot(metrics_df['epoch'], metrics_df['val_recall'], label='Recall')
        plt.plot(metrics_df['epoch'], metrics_df['val_f1'], label='F1')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics (Sample-based)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()