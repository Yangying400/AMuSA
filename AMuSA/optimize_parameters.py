#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import optuna
import numpy as np
import pandas as pd
import torch
import time
import json
import argparse
from datetime import datetime

from AMuSA.trainer import train_signature_classifier, evaluate_model
from AMuSA.ensemble import train_ensemble, evaluate_ensemble


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Signature classification pipeline with Optuna optimization"
    )

    parser.add_argument("--train_mutation", required=True)
    parser.add_argument("--train_exposure", required=True)
    parser.add_argument("--test_mutation", required=True)
    parser.add_argument("--test_exposure", required=True)
    parser.add_argument("--signature_file", required=True)

    parser.add_argument("--output_dir", default="output/optuna_optimization")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exposure_threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=24 * 3600)
    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--use_ensemble", action="store_true")
    parser.add_argument("--n_models", type=int, default=5)

    parser.add_argument("--primary_metric", default="sample_f1")
    parser.add_argument("--use_composite_score", action="store_true")
    parser.add_argument("--type", choices=["SBS", "DBS", "ID"], required=True)
    
    return parser.parse_args()


# =========================================================
# CONFIG
# =========================================================
def build_config(args):
    return {
        # Data
        "train_mutation": args.train_mutation,
        "train_exposure": args.train_exposure,
        "test_mutation": args.test_mutation,
        "test_exposure": args.test_exposure,
        "signature_file": args.signature_file,
        "type": args.type,
        # Output
        "output_dir": args.output_dir,
        "optuna_dir": os.path.join(args.output_dir, "optuna_optimization"),

        "final_model_dir": os.path.join(args.output_dir, "final_model", f"{args.type}_models"),
        

        # Training
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "exposure_threshold": args.exposure_threshold,
        "seed": args.seed,

        # Optimization
        "n_trials": args.n_trials,
        "timeout": args.timeout,
        "n_jobs": args.n_jobs,
        "study_name": "signature_classifier_optimization_enhanced",
        "storage": None,

        "use_ensemble": args.use_ensemble,
        "n_models": args.n_models,

        # Metrics
        "primary_metric": args.primary_metric,
        "use_composite_score": args.use_composite_score,

        "secondary_metrics": {
            "sample_jaccard": 0.1,
            "fitting_error": -0.05,
            "cosine_similarity": 0.05,
            "sample_precision": 0.05,
            "sample_recall": 0.05
        },

        "f1_weight_in_composite": 0.8,

        # Search space
        "param_search": {
            "encoding_dim": [64, 128, 256],
            "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
            "weight_decay": [1e-6, 1e-5, 1e-4],
            "decision_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
            "precision_weight": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "max_active_signatures": [3, 4, 5, 6, 7, 8]
        }
    }
# =========================================================
# SCORE
# =========================================================
def calculate_composite_score(metrics, config):
    primary_score = metrics.get(config['primary_metric'], 0.0)
    
    if not config['use_composite_score']:
        return primary_score
    
    # For composite scoring, give primary metric higher weight
    f1_weight = config.get('f1_weight_in_composite', 0.8)
    composite_score = f1_weight * primary_score
    
    # Add weighted secondary metrics with remaining weight
    remaining_weight = 1.0 - f1_weight
    secondary_weight_sum = sum(abs(w) for w in config['secondary_metrics'].values())
    
    if secondary_weight_sum > 0:
        for metric_name, weight in config['secondary_metrics'].items():
            metric_value = metrics.get(metric_name, 0.0)

            if 'error' in metric_name.lower():
                metric_value = max(0, 1 - metric_value)

            normalized_weight = (weight / secondary_weight_sum) * remaining_weight
            composite_score += normalized_weight * metric_value
    
    return composite_score

def objective_single_model(trial, CONFIG):
    params = {
        'encoding_dim': trial.suggest_categorical('encoding_dim', CONFIG['param_search']['encoding_dim']),
        'learning_rate': trial.suggest_categorical('learning_rate', CONFIG['param_search']['learning_rate']),
        'weight_decay': trial.suggest_categorical('weight_decay', CONFIG['param_search']['weight_decay']),
        'decision_threshold': trial.suggest_categorical('decision_threshold', CONFIG['param_search']['decision_threshold']),
        'precision_weight': trial.suggest_categorical('precision_weight', CONFIG['param_search']['precision_weight']),
        'max_active_signatures': trial.suggest_categorical('max_active_signatures', CONFIG['param_search']['max_active_signatures']),
        'optimize_thresholds': True
    }
    
    trial_dir = os.path.join(CONFIG['output_dir'], f'trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)
    model_path = os.path.join(trial_dir, 'model.pth')
    metrics_path = os.path.join(trial_dir, 'metrics.csv')

    with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Train model
    print(f"\n=== Training model for trial {trial.number} ===")
    print(f"Parameters: {params}")
    
    try:
        # Train model
        classifier, autoencoder, scaler, signature_names = train_signature_classifier(
            train_mutation_file=CONFIG['train_mutation'],
            train_exposure_file=CONFIG['train_exposure'],
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            model_save_path=model_path,
            encoding_dim=params['encoding_dim'],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            metrics_output_path=metrics_path,
            exposure_threshold=CONFIG['exposure_threshold'],
            decision_threshold=params['decision_threshold'],
            optimize_thresholds=params['optimize_thresholds'],
            precision_weight=params['precision_weight'],
            max_active_signatures=params['max_active_signatures'],
            verbose=False
        )
        
        # Evaluate model
        print(f"\n=== Evaluating model for trial {trial.number} ===")
        metrics = evaluate_model(
            model_path=model_path,
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            output_dir=os.path.join(trial_dir, 'evaluation'),
            max_active_signatures=params['max_active_signatures'],
            exposure_threshold=CONFIG['exposure_threshold']
        )
        
        # Calculate composite score
        score = calculate_composite_score(metrics, CONFIG)
        
        # Log detailed results
        result = {
            'trial': trial.number,
            'params': params,
            'metrics': metrics,
            'primary_score': metrics.get(CONFIG['primary_metric'], 0.0),
            'composite_score': score,
            'individual_metrics': {
                'sample_f1': metrics.get('sample_f1', 0.0),
                'sample_precision': metrics.get('sample_precision', 0.0),
                'sample_recall': metrics.get('sample_recall', 0.0),
                'sample_jaccard': metrics.get('sample_jaccard', 0.0),
                'cosine_similarity': metrics.get('cosine_similarity', 0.0)
            }
        }
        
        with open(os.path.join(trial_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=4, default=str)
        
        print(f"Trial {trial.number} results:")
        print(f"  Primary metric ({CONFIG['primary_metric']}): {metrics.get(CONFIG['primary_metric'], 0.0):.4f}")
        print(f"  Composite score: {score:.4f}")
        print(f"  Sample F1: {metrics.get('sample_f1', 0.0):.4f}")
        print(f"  Cosine Similarity: {metrics.get('cosine_similarity', 0.0):.4f}")
        
        return score
        
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()
        return -1.0

def objective_ensemble_model(trial, CONFIG):
    params = {
        'encoding_dim': trial.suggest_categorical('encoding_dim', CONFIG['param_search']['encoding_dim']),
        'learning_rate': trial.suggest_categorical('learning_rate', CONFIG['param_search']['learning_rate']),
        'weight_decay': trial.suggest_categorical('weight_decay', CONFIG['param_search']['weight_decay']),
        'decision_threshold': trial.suggest_categorical('decision_threshold', CONFIG['param_search']['decision_threshold']),
        'precision_weight': trial.suggest_categorical('precision_weight', CONFIG['param_search']['precision_weight']),
        'max_active_signatures': trial.suggest_categorical('max_active_signatures', CONFIG['param_search']['max_active_signatures']),
        'optimize_thresholds': True,
        'n_models': CONFIG['n_models']
    }

    trial_dir = os.path.join(CONFIG['output_dir'], f'ensemble_trial_{trial.number}')
    os.makedirs(trial_dir, exist_ok=True)
    model_dir = os.path.join(trial_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(trial_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Train ensemble
    print(f"\n=== Training ensemble for trial {trial.number} ===")
    print(f"Parameters: {params}")
    
    try:
        # Train ensemble
        model_paths = train_ensemble(
            train_mutation_file=CONFIG['train_mutation'],
            train_exposure_file=CONFIG['train_exposure'],
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            model_dir=model_dir,
            n_models=params['n_models'],
            encoding_dim=params['encoding_dim'],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            exposure_threshold=CONFIG['exposure_threshold'],
            decision_threshold=params['decision_threshold'],
            optimize_thresholds=params['optimize_thresholds'],
            precision_weight=params['precision_weight'],
            max_active_signatures=params['max_active_signatures'],
            verbose=False
        )
        
        if not model_paths:
            print(f"No models were successfully trained for trial {trial.number}")
            return -1.0
        
        # Evaluate ensemble
        print(f"\n=== Evaluating ensemble for trial {trial.number} ===")
        metrics = evaluate_ensemble(
            model_paths=model_paths,
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            output_dir=os.path.join(trial_dir, 'evaluation'),
            max_active_signatures=params['max_active_signatures'],
            exposure_threshold=CONFIG['exposure_threshold']
        )
        
        # Calculate composite score
        score = calculate_composite_score(metrics, CONFIG)
        
        # Log detailed results
        result = {
            'trial': trial.number,
            'params': params,
            'metrics': metrics,
            'primary_score': metrics.get(CONFIG['primary_metric'], 0.0),
            'composite_score': score,
            'individual_metrics': {
                'sample_f1': metrics.get('sample_f1', 0.0),
                'sample_precision': metrics.get('sample_precision', 0.0),
                'sample_recall': metrics.get('sample_recall', 0.0),
                'sample_jaccard': metrics.get('sample_jaccard', 0.0),
                'cosine_similarity': metrics.get('cosine_similarity', 0.0)
            },
            'n_successful_models': len(model_paths)
        }
        
        with open(os.path.join(trial_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=4, default=str)
        
        print(f"Ensemble trial {trial.number} results:")
        print(f"  Successful models: {len(model_paths)}/{params['n_models']}")
        print(f"  Primary metric ({CONFIG['primary_metric']}): {metrics.get(CONFIG['primary_metric'], 0.0):.4f}")
        print(f"  Composite score: {score:.4f}")
        print(f"  Sample F1: {metrics.get('sample_f1', 0.0):.4f}")
        print(f"  Cosine Similarity: {metrics.get('cosine_similarity', 0.0):.4f}")
        
        return score
        
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()
        return -1.0

def run_optimization(CONFIG):
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    storage = None
    if CONFIG['storage']:
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{os.path.join(CONFIG['output_dir'], CONFIG['storage'])}"
        )

    direction = "maximize"

    study = optuna.create_study(
        study_name=CONFIG['study_name'],
        storage=storage,
        direction=direction,
        load_if_exists=True
    )
    
    # objective function
    if CONFIG['use_ensemble']:
        objective = lambda trial: objective_ensemble_model(trial, CONFIG)
    else:
        objective = lambda trial: objective_single_model(trial, CONFIG)
    
    config_to_save = CONFIG.copy()
    config_to_save['optimization_strategy'] = 'sample_based_composite_score'
    config_to_save['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(os.path.join(CONFIG['output_dir'], 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)
    
    # Run optimization
    print(f"Starting optimization with {CONFIG['n_trials']} trials...")
    print(f"Primary metric: {CONFIG['primary_metric']}")
    print(f"Using composite score: {CONFIG['use_composite_score']}")
    
    if CONFIG['use_composite_score']:
        print(f"Secondary metrics weights: {CONFIG['secondary_metrics']}")
    
    start_time = time.time()
    
    study.optimize(
        objective, 
        n_trials=CONFIG['n_trials'],
        timeout=CONFIG['timeout'],
        n_jobs=CONFIG['n_jobs'],
        show_progress_bar=True
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*60)
    print(f"Optimization completed in {elapsed_time/60:.2f} minutes")
    print(f"Best composite score: {best_value:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    best_trial = study.best_trial
    best_trial_dir = os.path.join(
        CONFIG['output_dir'],
        f"{'ensemble_' if CONFIG['use_ensemble'] else ''}trial_{best_trial.number}"
    )
    
    best_results_path = os.path.join(best_trial_dir, 'results.json')
    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            best_results = json.load(f)
            
        print("\nBest trial detailed metrics:")
        individual_metrics = best_results.get('individual_metrics', {})
        for metric, value in individual_metrics.items():
            print(f"  {metric}: {value:.4f}")

    best_result = {
        'best_params': best_params,
        'best_value': best_value,
        'primary_metric': CONFIG['primary_metric'],
        'use_composite_score': CONFIG['use_composite_score'],
        'secondary_metrics': CONFIG['secondary_metrics'],
        'elapsed_time': elapsed_time,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'optimization_strategy': 'sample_based_composite_score'
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'best_params.json'), 'w') as f:
        json.dump(best_result, f, indent=4)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(CONFIG['output_dir'], 'trials.csv'), index=False)

    # 
    analyze_optimization_results(study, CONFIG)

    validate_f1_optimization(study, CONFIG, CONFIG['output_dir'])
    
    return best_params, best_value

def analyze_optimization_results(study, CONFIG):
    output_dir = CONFIG['output_dir']
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                'trial_number': trial.number,
                'composite_score': trial.value,
                **trial.params
            }

            trial_prefix = 'ensemble_' if CONFIG['use_ensemble'] else ''
            trial_dir = os.path.join(output_dir, f'{trial_prefix}trial_{trial.number}')
            results_path = os.path.join(trial_dir, 'results.json')
            
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                        individual_metrics = results.get('individual_metrics', {})
                        trial_data.update(individual_metrics)
                except:
                    pass
            
            trials_data.append(trial_data)
    
    if not trials_data:
        print("No completed trials found for analysis")
        return

    analysis_df = pd.DataFrame(trials_data)
    analysis_df.to_csv(os.path.join(output_dir, 'detailed_trials_analysis.csv'), index=False)

    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = analysis_df[numeric_cols].corr()

    correlation_matrix.to_csv(os.path.join(output_dir, 'parameter_metric_correlations.csv'))

    summary_stats = {
        'total_trials': len(trials_data),
        'best_composite_score': analysis_df['composite_score'].max(),
        'mean_composite_score': analysis_df['composite_score'].mean(),
        'std_composite_score': analysis_df['composite_score'].std(),
        'parameter_ranges': {}
    }
    
    for param in CONFIG['param_search'].keys():
        if param in analysis_df.columns:
            summary_stats['parameter_ranges'][param] = {
                'min': analysis_df[param].min(),
                'max': analysis_df[param].max(),
                'best_value': analysis_df.loc[analysis_df['composite_score'].idxmax(), param]
            }
    
    with open(os.path.join(output_dir, 'optimization_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4, default=str)
    
    print(f"Detailed analysis saved to {output_dir}")

def train_with_best_params(best_params, CONFIG):
    print("\n" + "="*60)
    print("Training final model with best parameters...")
    
    # Create final model directory
    final_dir = CONFIG['final_model_dir']

    os.makedirs(final_dir, exist_ok=True)
    
    
    params = {
        'encoding_dim': best_params.get('encoding_dim', 128),
        'learning_rate': best_params.get('learning_rate', 1e-4),
        'weight_decay': best_params.get('weight_decay', 1e-5),
        'decision_threshold': best_params.get('decision_threshold', 0.5),
        'precision_weight': best_params.get('precision_weight', 0.9),
        'max_active_signatures': best_params.get('max_active_signatures', 6),
        'optimize_thresholds': True,
        'n_models': CONFIG['n_models'] if CONFIG['use_ensemble'] else 1
    }

    with open(os.path.join(final_dir, 'final_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    if CONFIG['use_ensemble']:
        # Train ensemble
        model_dir = os.path.join(final_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_paths = train_ensemble(
            train_mutation_file=CONFIG['train_mutation'],
            train_exposure_file=CONFIG['train_exposure'],
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            model_dir=model_dir,
            n_models=params['n_models'],
            encoding_dim=params['encoding_dim'],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            exposure_threshold=CONFIG['exposure_threshold'],
            decision_threshold=params['decision_threshold'],
            optimize_thresholds=params['optimize_thresholds'],
            precision_weight=params['precision_weight'],
            max_active_signatures=params['max_active_signatures'],
            verbose=True
        )

        if model_paths:
            metrics = evaluate_ensemble(
                model_paths=model_paths,
                test_mutation_file=CONFIG['test_mutation'],
                test_exposure_file=CONFIG['test_exposure'],
                signature_file=CONFIG['signature_file'],
                output_dir=os.path.join(final_dir, 'evaluation'),
                max_active_signatures=params['max_active_signatures'],
                exposure_threshold=CONFIG['exposure_threshold']
            )
    else:
        model_path = os.path.join(final_dir, 'model.pth')
        metrics_path = os.path.join(final_dir, 'metrics.csv')
        
        classifier, autoencoder, scaler, signature_names = train_signature_classifier(
            train_mutation_file=CONFIG['train_mutation'],
            train_exposure_file=CONFIG['train_exposure'],
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            model_save_path=model_path,
            encoding_dim=params['encoding_dim'],
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            metrics_output_path=metrics_path,
            exposure_threshold=CONFIG['exposure_threshold'],
            decision_threshold=params['decision_threshold'],
            optimize_thresholds=params['optimize_thresholds'],
            precision_weight=params['precision_weight'],
            max_active_signatures=params['max_active_signatures'],
            verbose=True
        )

        metrics = evaluate_model(
            model_path=model_path,
            test_mutation_file=CONFIG['test_mutation'],
            test_exposure_file=CONFIG['test_exposure'],
            signature_file=CONFIG['signature_file'],
            output_dir=os.path.join(final_dir, 'evaluation'),
            max_active_signatures=params['max_active_signatures'],
            exposure_threshold=CONFIG['exposure_threshold']
        )

    with open(os.path.join(final_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4, default=str)

    final_composite_score = calculate_composite_score(metrics, CONFIG)
    
    print("\n" + "="*60)
    print("Final model training complete!")
    print(f"Model and evaluation results saved to {final_dir}")
    print("\nFinal Metrics (Sample-based):")
    print(f"  Sample-based Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Sample-based Precision: {metrics.get('sample_precision', 0.0):.4f}")
    print(f"  Sample-based Recall: {metrics.get('sample_recall', 0.0):.4f}")
    print(f"  Sample-based F1 Score: {metrics.get('sample_f1', 0.0):.4f}")
    print(f"  Sample-based Jaccard Similarity: {metrics.get('sample_jaccard', 0.0):.4f}")
    print(f"  Cosine Similarity: {metrics.get('cosine_similarity', 0.0):.4f}")
    print(f"  Traditional AUC: {metrics.get('auc', 0.0):.4f}")
    print(f"  Composite Score: {final_composite_score:.4f}")
    
    # Save final composite score
    final_result = {
        'final_metrics': metrics,
        'composite_score': final_composite_score,
        'optimization_config': CONFIG,
        'final_params': params
    }
    
    with open(os.path.join(final_dir, 'final_results.json'), 'w') as f:
        json.dump(final_result, f, indent=4, default=str)
    return final_dir

def validate_f1_optimization(study, config, output_dir):
    f1_scores = []
    trial_numbers = []
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_prefix = 'ensemble_' if config['use_ensemble'] else ''
            trial_dir = os.path.join(output_dir, f'{trial_prefix}trial_{trial.number}')
            results_path = os.path.join(trial_dir, 'results.json')
            
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                        f1_score = results.get('individual_metrics', {}).get('sample_f1', 0.0)
                        f1_scores.append(f1_score)
                        trial_numbers.append(trial.number)
                except:
                    continue
    
    if len(f1_scores) < 3:
        print("Insufficient trials for validation")
        return
    
    # Analysis
    f1_scores = np.array(f1_scores)
    trial_numbers = np.array(trial_numbers)
    
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_trial_num = trial_numbers[best_f1_idx]
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"F1 Score Statistics:")
    print(f"  Best F1 Score: {best_f1:.4f} (Trial {best_trial_num})")
    print(f"  Mean F1 Score: {mean_f1:.4f}")
    print(f"  Std F1 Score: {std_f1:.4f}")
    print(f"  F1 Score Range: {np.min(f1_scores):.4f} - {np.max(f1_scores):.4f}")
    
    # Check if best trial according to study matches best F1
    study_best_trial = study.best_trial.number
    study_best_objective = study.best_value
    
    print(f"\nOptimization Validation:")
    print(f"  Study best trial: {study_best_trial}")
    print(f"  Study best objective: {study_best_objective:.4f}")
    print(f"  Best F1 trial: {best_trial_num}")
    print(f"  Best F1 score: {best_f1:.4f}")

    validation_results = {
        'f1_scores': f1_scores.tolist(),
        'trial_numbers': trial_numbers.tolist(),
        'best_f1_score': float(best_f1),
        'best_f1_trial': int(best_trial_num),
        'mean_f1': float(mean_f1),
        'std_f1': float(std_f1),
        'study_best_trial': int(study_best_trial),
        'study_best_objective': float(study_best_objective),
        'f1_optimization_aligned': bool(study_best_trial == best_trial_num),
        'use_composite_score': bool(config['use_composite_score'])
    }
    
    with open(os.path.join(output_dir, 'f1_validation.json'), 'w') as f:
        json.dump(validation_results, f, indent=4)

def main():
    args = parse_args()

    
    CONFIG = build_config(args)

    # Step1: optimization
    best_params, best_value = run_optimization(CONFIG)

    # Step2: train final model
    model_dir = train_with_best_params(best_params, CONFIG)
    return model_dir


if __name__ == "__main__":
    main() 