import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine
import scipy.stats as stats

def calculate_sample_based_exposure_metrics(true_exposures, predicted_exposures, 
                                          signature_matrix=None, mutation_data=None, 
                                          signature_names=None, sample_names=None,
                                          output_dir=None, prefix=''):
    n_samples, n_signatures = true_exposures.shape
    
    if signature_names is None:
        signature_names = [f"Sig_{i+1}" for i in range(n_signatures)]
    if sample_names is None:
        sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    
    # Initialize sample-level metrics
    sample_metrics = {
        'fitting_error': np.zeros(n_samples),
        'cosine_similarity_exposure': np.zeros(n_samples),
        'cosine_similarity_spectrum': np.zeros(n_samples),
        'correlation_exposure': np.zeros(n_samples),
        'mse': np.zeros(n_samples),
        'mae': np.zeros(n_samples),
        'active_signature_overlap': np.zeros(n_samples),
        'active_signature_precision': np.zeros(n_samples),
        'active_signature_recall': np.zeros(n_samples),
        'active_signature_f1': np.zeros(n_samples)
    }
    
    # Calculate per-sample metrics
    for i in range(n_samples):
        true_exp = true_exposures[i]
        pred_exp = predicted_exposures[i]
        
        # Normalize exposures to sum to 1
        true_norm = true_exp / (np.sum(true_exp) + 1e-10)
        pred_norm = pred_exp / (np.sum(pred_exp) + 1e-10)
        
        # Fitting error (Total Variation Distance)
        sample_metrics['fitting_error'][i] = np.sum(np.abs(true_norm - pred_norm)) / 2
        
        # Cosine similarity for exposures
        if np.linalg.norm(true_exp) > 0 and np.linalg.norm(pred_exp) > 0:
            sample_metrics['cosine_similarity_exposure'][i] = np.dot(true_exp, pred_exp) / (
                np.linalg.norm(true_exp) * np.linalg.norm(pred_exp)
            )
        else:
            sample_metrics['cosine_similarity_exposure'][i] = 0.0
        
        # Pearson correlation for exposures
        if np.std(true_exp) > 0 and np.std(pred_exp) > 0:
            sample_metrics['correlation_exposure'][i] = np.corrcoef(true_exp, pred_exp)[0, 1]
        else:
            sample_metrics['correlation_exposure'][i] = 0.0
        
        # MSE and MAE
        sample_metrics['mse'][i] = np.mean((true_exp - pred_exp) ** 2)
        sample_metrics['mae'][i] = np.mean(np.abs(true_exp - pred_exp))
        
        # Active signature analysis (binary classification metrics)
        threshold = 0.05  # 5% threshold for active signatures
        true_active = (true_exp > threshold).astype(int)
        pred_active = (pred_exp > threshold).astype(int)
        
        # Calculate overlap metrics
        tp = np.sum((true_active == 1) & (pred_active == 1))
        fp = np.sum((true_active == 0) & (pred_active == 1))
        fn = np.sum((true_active == 1) & (pred_active == 0))
        
        # Jaccard similarity for active signatures
        union = np.sum((true_active == 1) | (pred_active == 1))
        sample_metrics['active_signature_overlap'][i] = tp / union if union > 0 else 1.0
        
        # Precision, Recall, F1 for active signatures
        sample_metrics['active_signature_precision'][i] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        sample_metrics['active_signature_recall'][i] = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        
        if sample_metrics['active_signature_precision'][i] + sample_metrics['active_signature_recall'][i] > 0:
            sample_metrics['active_signature_f1'][i] = 2 * sample_metrics['active_signature_precision'][i] * sample_metrics['active_signature_recall'][i] / (
                sample_metrics['active_signature_precision'][i] + sample_metrics['active_signature_recall'][i]
            )
        else:
            sample_metrics['active_signature_f1'][i] = 0.0
    
    # Calculate spectrum reconstruction metrics if data is available
    if signature_matrix is not None and mutation_data is not None:
        # Reconstruct spectra from predicted exposures
        reconstructed_spectra = np.dot(predicted_exposures, signature_matrix.T)
        
        for i in range(n_samples):
            original_spectrum = mutation_data[i]
            reconstructed_spectrum = reconstructed_spectra[i]
            
            # Cosine similarity for reconstructed spectra
            if np.linalg.norm(original_spectrum) > 0 and np.linalg.norm(reconstructed_spectrum) > 0:
                sample_metrics['cosine_similarity_spectrum'][i] = np.dot(original_spectrum, reconstructed_spectrum) / (
                    np.linalg.norm(original_spectrum) * np.linalg.norm(reconstructed_spectrum)
                )
            else:
                sample_metrics['cosine_similarity_spectrum'][i] = 0.0
    
    # Aggregate metrics
    aggregate_metrics = {
        'sample_fitting_error_mean': np.mean(sample_metrics['fitting_error']),
        'sample_fitting_error_std': np.std(sample_metrics['fitting_error']),
        'sample_cosine_similarity_exposure_mean': np.mean(sample_metrics['cosine_similarity_exposure']),
        'sample_cosine_similarity_spectrum_mean': np.mean(sample_metrics['cosine_similarity_spectrum']),
        'sample_correlation_exposure_mean': np.mean(sample_metrics['correlation_exposure']),
        'sample_mse_mean': np.mean(sample_metrics['mse']),
        'sample_mae_mean': np.mean(sample_metrics['mae']),
        'sample_active_overlap_mean': np.mean(sample_metrics['active_signature_overlap']),
        'sample_active_precision_mean': np.mean(sample_metrics['active_signature_precision']),
        'sample_active_recall_mean': np.mean(sample_metrics['active_signature_recall']),
        'sample_active_f1_mean': np.mean(sample_metrics['active_signature_f1']),
        'n_samples': n_samples,
        'n_signatures': n_signatures
    }
    
    # Combine all metrics
    all_metrics = {
        'sample_metrics': sample_metrics,
        'aggregate_metrics': aggregate_metrics,
        'sample_names': sample_names,
        'signature_names': signature_names
    }
    
    # Save detailed results if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample-level metrics
        sample_df = pd.DataFrame({
            'sample_name': sample_names,
            **sample_metrics
        })
        sample_df.to_csv(os.path.join(output_dir, f'{prefix}sample_exposure_metrics.csv'), index=False)
        
        # Save aggregate metrics
        aggregate_df = pd.DataFrame([aggregate_metrics])
        aggregate_df.to_csv(os.path.join(output_dir, f'{prefix}aggregate_exposure_metrics.csv'), index=False)
        
        # Create visualizations
        create_sample_based_visualizations(
            sample_metrics, aggregate_metrics, sample_names, 
            output_dir, prefix
        )
        
        print(f"Sample-based exposure metrics saved to {output_dir}")
    
    return all_metrics

def create_sample_based_visualizations(sample_metrics, aggregate_metrics, sample_names, 
                                     output_dir, prefix=''):
    """Create visualizations for sample-based exposure metrics"""
    
    # Distribution plots for key metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    key_metrics = [
        ('fitting_error', 'Fitting Error'),
        ('cosine_similarity_exposure', 'Cosine Similarity (Exposure)'),
        ('cosine_similarity_spectrum', 'Cosine Similarity (Spectrum)'),
        ('active_signature_overlap', 'Active Signature Overlap'),
        ('active_signature_f1', 'Active Signature F1'),
        ('correlation_exposure', 'Exposure Correlation')
    ]
    
    for i, (metric_key, metric_title) in enumerate(key_metrics):
        values = sample_metrics[metric_key]
        
        # Skip if all zeros (e.g., spectrum metrics when no signature matrix provided)
        if np.sum(values) == 0 and metric_key == 'cosine_similarity_spectrum':
            axes[i].text(0.5, 0.5, 'No spectrum data', ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].set_title(metric_title)
            continue
        
        # Plot histogram
        axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
        axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(values):.3f}')
        axes[i].set_xlabel(metric_title)
        axes[i].set_ylabel('Number of Samples')
        axes[i].set_title(f'{metric_title} Distribution')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}sample_metrics_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix of sample metrics
    metrics_for_corr = ['fitting_error', 'cosine_similarity_exposure', 'correlation_exposure', 
                       'active_signature_overlap', 'active_signature_f1']
    
    corr_data = {metric: sample_metrics[metric] for metric in metrics_for_corr}
    corr_df = pd.DataFrame(corr_data)
    corr_matrix = corr_df.corr()
    
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Sample Metrics Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}sample_metrics_correlation.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sample performance scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Fitting error vs Cosine similarity
    axes[0, 0].scatter(sample_metrics['fitting_error'], 
                      sample_metrics['cosine_similarity_exposure'], alpha=0.6)
    axes[0, 0].set_xlabel('Fitting Error')
    axes[0, 0].set_ylabel('Cosine Similarity (Exposure)')
    axes[0, 0].set_title('Fitting Error vs Cosine Similarity')
    axes[0, 0].grid(alpha=0.3)
    
    # Active signature metrics
    axes[0, 1].scatter(sample_metrics['active_signature_precision'], 
                      sample_metrics['active_signature_recall'], alpha=0.6)
    axes[0, 1].set_xlabel('Active Signature Precision')
    axes[0, 1].set_ylabel('Active Signature Recall')
    axes[0, 1].set_title('Precision vs Recall (Active Signatures)')
    axes[0, 1].grid(alpha=0.3)
    
    # Exposure correlation vs Active F1
    axes[1, 0].scatter(sample_metrics['correlation_exposure'], 
                      sample_metrics['active_signature_f1'], alpha=0.6)
    axes[1, 0].set_xlabel('Exposure Correlation')
    axes[1, 0].set_ylabel('Active Signature F1')
    axes[1, 0].set_title('Exposure Correlation vs Active Signature F1')
    axes[1, 0].grid(alpha=0.3)
    
    # MSE vs MAE
    axes[1, 1].scatter(sample_metrics['mse'], sample_metrics['mae'], alpha=0.6)
    axes[1, 1].set_xlabel('MSE')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MSE vs MAE')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}sample_performance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_exposures(mutation_data, signature_matrix, active_signatures=None, 
                       method='nnls', min_exposure=0.01, mutation_count=None):
    """
    Enhanced exposure calculation with sample-based validation
    """
    n_samples = mutation_data.shape[0]
    n_signatures = signature_matrix.shape[1]
    exposures = np.zeros((n_samples, n_signatures))
    
    # Method selection logic
    if method == 'auto':
        if mutation_count is not None and np.mean(mutation_count) < 1000:
            method = 'bidirectional'
        else:
            method = 'nnls'
    
    successful_fits = 0
    
    for i in range(n_samples):
        sample_mutation = mutation_data[i]
        
        # Get active signatures for this sample
        if active_signatures is not None:
            if active_signatures.ndim == 1:
                active_sig_indices = np.where(active_signatures > 0)[0]
            else:
                active_sig_indices = np.where(active_signatures[i] > 0)[0]
                
            if len(active_sig_indices) == 0:
                active_sig_indices = np.arange(n_signatures)
        else:
            active_sig_indices = np.arange(n_signatures)
            
        # Extract relevant signatures
        sig_subset = signature_matrix[:, active_sig_indices]
        
        try:
            # Calculate exposures based on method
            if method == 'nnls':
                coefficients, residual = nnls(sig_subset, sample_mutation)
            elif method == 'linear_regression':
                model = LinearRegression(positive=True)
                model.fit(sig_subset, sample_mutation)
                coefficients = model.coef_
            elif method == 'bidirectional':
                # Use bidirectional NNLS if available
                coefficients = nnls_bidirectional(sample_mutation, sig_subset, 
                                                thresh_backward=0.001, 
                                                thresh_forward=0.001)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Map coefficients back to full signature set
            for j, idx in enumerate(active_sig_indices):
                exposures[i, idx] = coefficients[j]
                
            successful_fits += 1
            
        except Exception as e:
            print(f"Warning: Failed to fit sample {i}: {e}")
            continue
    
    # Filter small exposures
    exposures[exposures < min_exposure] = 0
    
    # Normalize exposures per sample
    for i in range(n_samples):
        total_exposure = np.sum(exposures[i])
        if total_exposure > 0:
            exposures[i] = exposures[i] / total_exposure
    
    print(f"Successfully fitted {successful_fits}/{n_samples} samples")
    
    return exposures

def nnls_bidirectional(mutation_data, signature_matrix, thresh_backward=0.001, 
                      thresh_forward=None, max_iter=100, per_trial=True):
    """
    Bidirectional NNLS implementation for improved feature selection stability
    """
    n_features, n_signatures = signature_matrix.shape
    
    # Initial fit
    h, _ = nnls(signature_matrix, mutation_data)
    indices_retained = np.where(h > 0)[0]
    indices_all = np.arange(0, n_signatures)
    
    # Set forward threshold if not provided
    if thresh_forward is None:
        thresh_forward = thresh_backward
    
    # Iterative optimization
    for i_iter in range(max_iter):
        # Backward step (remove signatures)
        if len(indices_retained) <= 1:
            backward_stop = True
        else:
            # Current likelihood
            h, _ = nnls(signature_matrix[:, indices_retained], mutation_data)
            p_current = signature_matrix[:, indices_retained] @ h
            p_current = p_current / (np.sum(p_current) + 1e-10)
            loglikelihood_current = _multinomial_loglikelihood(mutation_data, p_current, per_trial)
            
            # Test removing each signature
            loglikelihoods = []
            for index in indices_retained:
                test_indices = np.array([idx for idx in indices_retained if idx != index])
                if len(test_indices) > 0:
                    h_test, _ = nnls(signature_matrix[:, test_indices], mutation_data)
                    p_test = signature_matrix[:, test_indices] @ h_test
                    p_test = p_test / (np.sum(p_test) + 1e-10)
                    loglikelihoods.append(_multinomial_loglikelihood(mutation_data, p_test, per_trial))
                else:
                    loglikelihoods.append(-np.inf)
            
            loglikelihoods = np.array(loglikelihoods)
            likelihood_changes = loglikelihood_current - loglikelihoods
            
            if np.min(likelihood_changes) >= thresh_backward:
                backward_stop = True
            else:
                backward_stop = False
                remove_idx = np.argmin(likelihood_changes)
                indices_retained = np.array([idx for i, idx in enumerate(indices_retained) if i != remove_idx])
        
        # Forward step (add signatures)
        indices_others = np.array([idx for idx in indices_all if idx not in indices_retained])
        
        if len(indices_others) == 0:
            forward_stop = True
        else:
            # Current likelihood
            h, _ = nnls(signature_matrix[:, indices_retained], mutation_data)
            p_current = signature_matrix[:, indices_retained] @ h
            p_current = p_current / (np.sum(p_current) + 1e-10)
            loglikelihood_current = _multinomial_loglikelihood(mutation_data, p_current, per_trial)
            
            # Test adding each signature
            loglikelihoods = []
            for index in indices_others:
                test_indices = np.sort(np.append(indices_retained, index))
                h_test, _ = nnls(signature_matrix[:, test_indices], mutation_data)
                p_test = signature_matrix[:, test_indices] @ h_test
                p_test = p_test / (np.sum(p_test) + 1e-10)
                loglikelihoods.append(_multinomial_loglikelihood(mutation_data, p_test, per_trial))
            
            loglikelihoods = np.array(loglikelihoods)
            likelihood_changes = loglikelihoods - loglikelihood_current
            
            if np.max(likelihood_changes) <= thresh_forward:
                forward_stop = True
            else:
                forward_stop = False
                add_idx = np.argmax(likelihood_changes)
                indices_retained = np.sort(np.append(indices_retained, indices_others[add_idx]))
        
        # Stop condition
        if backward_stop and forward_stop:
            break
    
    # Final fit
    h, _ = nnls(signature_matrix[:, indices_retained], mutation_data)
    exposure = np.zeros(n_signatures)
    exposure[indices_retained] = h
    
    return exposure

def _multinomial_loglikelihood(x, p, epsilon=1e-16, per_trial=True):
    """Calculate multinomial log-likelihood"""
    p = p.astype(float)
    p = np.clip(p, epsilon, 1.0)
    p = p / np.sum(p)
    
    if per_trial:
        return np.sum(x * np.log(p)) / (np.sum(x) + epsilon)
    else:
        return np.sum(x * np.log(p))

def reconstruct_mutations(exposures, signature_matrix):
    """
    Reconstruct mutation counts from exposures and signature matrix
    """
    return np.dot(exposures, signature_matrix.T)

def evaluate_reconstruction_quality(original_mutations, reconstructed_mutations, 
                                  sample_names=None, output_dir=None, prefix=''):
    """
    Evaluate reconstruction quality with sample-based metrics
    """
    n_samples = original_mutations.shape[0]
    
    if sample_names is None:
        sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    
    # Sample-level reconstruction metrics
    sample_cosine = np.zeros(n_samples)
    sample_mse = np.zeros(n_samples)
    sample_mae = np.zeros(n_samples)
    sample_r2 = np.zeros(n_samples)
    
    for i in range(n_samples):
        original = original_mutations[i]
        reconstructed = reconstructed_mutations[i]
        
        # Cosine similarity
        if np.linalg.norm(original) > 0 and np.linalg.norm(reconstructed) > 0:
            sample_cosine[i] = np.dot(original, reconstructed) / (
                np.linalg.norm(original) * np.linalg.norm(reconstructed)
            )
        else:
            sample_cosine[i] = 0.0
        
        # MSE and MAE
        sample_mse[i] = np.mean((original - reconstructed) ** 2)
        sample_mae[i] = np.mean(np.abs(original - reconstructed))
        
        # R-squared
        ss_res = np.sum((original - reconstructed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        sample_r2[i] = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Aggregate metrics
    metrics = {
        'sample_cosine_mean': np.mean(sample_cosine),
        'sample_cosine_std': np.std(sample_cosine),
        'sample_mse_mean': np.mean(sample_mse),
        'sample_mae_mean': np.mean(sample_mae),
        'sample_r2_mean': np.mean(sample_r2),
        'sample_metrics': {
            'cosine_similarity': sample_cosine,
            'mse': sample_mse,
            'mae': sample_mae,
            'r2': sample_r2
        },
        'sample_names': sample_names
    }
    
    # Save results if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample-level results
        sample_df = pd.DataFrame({
            'sample_name': sample_names,
            'cosine_similarity': sample_cosine,
            'mse': sample_mse,
            'mae': sample_mae,
            'r2': sample_r2
        })
        sample_df.to_csv(os.path.join(output_dir, f'{prefix}reconstruction_sample_metrics.csv'), index=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cosine similarity distribution
        axes[0, 0].hist(sample_cosine, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(sample_cosine), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(sample_cosine):.3f}')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].set_title('Cosine Similarity Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # R² distribution
        axes[0, 1].hist(sample_r2, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(sample_r2), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(sample_r2):.3f}')
        axes[0, 1].set_xlabel('R²')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('R² Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Cosine vs R²
        axes[1, 0].scatter(sample_cosine, sample_r2, alpha=0.6)
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_title('Cosine Similarity vs R²')
        axes[1, 0].grid(alpha=0.3)
        
        # MSE vs MAE
        axes[1, 1].scatter(sample_mse, sample_mae, alpha=0.6)
        axes[1, 1].set_xlabel('MSE')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('MSE vs MAE')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}reconstruction_quality_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Reconstruction quality metrics saved to {output_dir}")
    
    return metrics