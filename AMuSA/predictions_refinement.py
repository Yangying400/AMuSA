#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch
import argparse

from trainer import load_model, get_encoded_features
from data_loader import load_data
from utils import post_process_predictions


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Signature prediction pipeline")

    parser.add_argument("--test_mutation", type=str, required=True)
    parser.add_argument("--signature_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/optuna_optimization")

    parser.add_argument("--max_active_signatures", type=int, default=10)
    parser.add_argument("--exposure_threshold", type=float, default=0.05)

    return parser.parse_args()


# =========================================================
# CORE FUNCTION (original logic preserved)
# =========================================================
def extract_predictions(args):

    # ============================
    # Configuration (from CLI)
    # ============================
    CONFIG = {
        'test_mutation': args.test_mutation,
        'signature_file': args.signature_file,
        'output_dir': args.output_dir,
        'max_active_signatures': args.max_active_signatures,
        'exposure_threshold': args.exposure_threshold
    }

    # Output directory for predictions
    pred_output_dir = os.path.join(CONFIG['output_dir'], 'final_predictions')
    os.makedirs(pred_output_dir, exist_ok=True)

    # Load model directory
    ensemble_model_dir = os.path.join(CONFIG['output_dir'], 'final_model', 'models')
    model_paths = [
        os.path.join(ensemble_model_dir, f)
        for f in os.listdir(ensemble_model_dir)
        if f.endswith('.pth')
    ]

    # Load first model to extract scaler and signature names
    first_model = torch.load(model_paths[0], map_location='cpu')
    scaler = first_model['scaler']
    signature_names = first_model['signature_names']

    # Load mutation data (no exposure required)
    X_test, _, _, sample_ids, _ = load_data(
        mutation_file=CONFIG['test_mutation'],
        exposure_file=None,
        signature_file=CONFIG['signature_file'],
        scaler=scaler,
        train=False
    )

    # Load signature and mutation matrices (for reference only)
    signature_matrix = None
    mutation_data = None
    if os.path.exists(CONFIG['signature_file']):
        signature_df = pd.read_csv(CONFIG['signature_file'], index_col=0)
        signature_matrix = signature_df.values

        mutation_df = pd.read_csv(CONFIG['test_mutation'], index_col=0).T
        mutation_data = mutation_df.values

    # =====================================================
    # Ensemble prediction
    # =====================================================
    all_probs = []
    for path in model_paths:
        classifier, autoencoder, _, _ = load_model(path)
        X_test_encoded = get_encoded_features(autoencoder, X_test)

        classifier.eval()
        with torch.no_grad():
            probs, _, _ = classifier(
                torch.FloatTensor(X_test_encoded).to(classifier.thresholds.device)
            )
            all_probs.append(probs.cpu().numpy())

    # Average probabilities across ensemble
    avg_probs = np.mean(all_probs, axis=0)

    # Initialize thresholds
    avg_thresholds = np.ones(len(signature_names)) * CONFIG['exposure_threshold']

    # =====================================================
    # Threshold-based filtering
    # =====================================================
    active_preds = np.zeros_like(avg_probs)

    for i in range(avg_probs.shape[1]):
        active_preds[:, i] = (
            avg_probs[:, i] >= avg_thresholds[i]
        ).astype(float)

    # =====================================================
    # Limit number of active signatures
    # =====================================================
    if CONFIG['max_active_signatures'] is not None:
        active_preds = post_process_predictions(
            avg_probs,
            avg_thresholds,
            CONFIG['max_active_signatures']
        )

    # =====================================================
    # Linked signature rules (biological constraints)
    # =====================================================
    linked_groups = [
        ["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"],
        ["SBS2", "SBS13"],
        ["SBS10a", "SBS10b"],
        ["SBS17a", "SBS17b"]
    ]
    exclude_if_7_group = ["SBS3", "SBS8"]

    for i in range(active_preds.shape[0]):
        for group in linked_groups:
            group_idx = [j for j, sig in enumerate(signature_names) if sig in group]

            for idx in group_idx:
                if active_preds[i, idx] > 0:
                    # Activate all signatures within the group
                    for idx2 in group_idx:
                        active_preds[i, idx2] = 1.0

                    # Special rule: exclude SBS3/SBS8 if SBS7 group is active
                    if group == ["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"]:
                        for sig in exclude_if_7_group:
                            if sig in signature_names:
                                exclude_idx = signature_names.index(sig)
                                active_preds[i, exclude_idx] = 0.0
                    break

    # =====================================================
    # Convert to binary predictions
    # =====================================================
    active_preds_binary = active_preds.astype(int)

    # =====================================================
    # Save outputs
    # =====================================================
    predictions_df = pd.DataFrame(
        active_preds_binary.T,
        index=signature_names,
        columns=sample_ids
    )
    predictions_df.to_csv(os.path.join(pred_output_dir, 'predictions.csv'))

    probs_df = pd.DataFrame(
        avg_probs.T,
        index=signature_names,
        columns=sample_ids
    )
    probs_df.to_csv(os.path.join(pred_output_dir, 'probabilities.csv'))

    thresholds_df = pd.DataFrame({
        'signature': signature_names,
        'threshold': avg_thresholds
    })
    thresholds_df.to_csv(os.path.join(pred_output_dir, 'thresholds.csv'), index=False)

    return predictions_df


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    extract_predictions(args)