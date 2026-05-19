#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import torch

from AMuSA.trainer import load_model
from AMuSA.data_loader import load_data
from AMuSA.utils import post_process_predictions
from AMuSA.trainer import get_encoded_features

def extract_predictions(
    mutation_file,
    signature_file,
    base_model_dir,
    model_type,
    output_dir,
    config_path=None,
    max_active_signatures=7
):
    """
    Predict active mutational signatures using an ensemble of trained models.

    Parameters
    ----------
    mutation_file : str
        Path to mutation catalog file (features x samples)

    signature_file : str
        Path to signature matrix file (features x signatures)

    base_model_dir : str
        Base directory containing model subfolders (SBS_models, DBS_models, ID_models)

    model_type : str
        Model type to use ("SBS", "DBS", "ID")

    output_dir : str
        Base output directory

    max_active_signatures : int or None
        Maximum number of active signatures per sample.
        If None, no limit is applied.

    Returns
    -------
    predictions_df : pd.DataFrame
        Binary prediction matrix (signatures x samples)
    """

    # =============================
    # 1. Resolve model directory
    # =============================
    model_dir = os.path.join(base_model_dir, f"{model_type}_models")

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    # =============================
    # 2. Output directory (by type)
    # =============================
    pred_output_dir = os.path.join(
        output_dir,
        "final_predictions",
        model_type
    )
    os.makedirs(pred_output_dir, exist_ok=True)

    # =============================
    # 3. Collect model paths
    # =============================
    model_paths = sorted([
    os.path.join(model_dir, f)
    for f in os.listdir(model_dir)
    if f.endswith(".pth")
])

    if len(model_paths) == 0:
        raise ValueError(f"No .pth models found in {model_dir}")

    print(f"Using {len(model_paths)} models from: {model_dir}")

    # =============================
    # 4. Load scaler
    # =============================
    first_model = torch.load(model_paths[0], map_location="cpu",weights_only=False)
    scaler = first_model["scaler"]
    #signature_names = first_model['signature_names']
    # =============================
    # 5. Load input data
    # =============================
    X_test, _, _, sample_ids, _ = load_data(
        mutation_file=mutation_file,
        exposure_file=None,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )

    # =============================
    # 6. Load signatures
    # =============================
    signature_df = pd.read_csv(signature_file, index_col=0)
    mutation_df = pd.read_csv(mutation_file, index_col=0).T
    mutation_data = mutation_df.values
    signature_names = signature_df.columns.tolist()
    model_signature_names = first_model["signature_names"]
    # check missing signatures
    missing = set(signature_names) - set(model_signature_names)
    if len(missing) > 0:
        raise ValueError(f"Missing signatures in model: {missing}")
    
    # model signature order）
    
    sig_indices = [
        model_signature_names.index(s)
        for s in signature_names
    ]
    
    
    mutation_df = pd.read_csv(mutation_file, index_col=0).T
    mutation_data = mutation_df.values
    
    # =============================
    # 7. Ensemble prediction
    # =============================
    all_probs = []

    for path in model_paths:
        classifier, autoencoder, _, _ = load_model(path)

        X_test_encoded = get_encoded_features(autoencoder, X_test)

        classifier.eval()
        with torch.no_grad():
            probs, _, _ = classifier(
                torch.FloatTensor(X_test_encoded).to(classifier.thresholds.device)
            )

            probs = probs.cpu().numpy()
            probs = probs[:, sig_indices]
            all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)

    # =============================
    # 8. Threshold aggregation
    # =============================
    all_thresholds = []

    for path in model_paths:
        model_dict = torch.load(path, map_location="cpu",weights_only=False)
        if "thresholds" in model_dict:
            all_thresholds.append(model_dict["thresholds"])

    if all_thresholds:
        avg_thresholds = np.mean(all_thresholds, axis=0)[sig_indices]
    else:
        avg_thresholds = np.ones(len(sig_indices)) * 0.4

    avg_thresholds = np.minimum(avg_thresholds, 0.4)

    # =============================
    # 9. Binarization
    # =============================
    active_preds = np.zeros_like(avg_probs)

    for i in range(avg_probs.shape[1]):
        active_preds[:, i] = (avg_probs[:, i] >= avg_thresholds[i]).astype(float)

    # =============================
    # 10. Post-processing (optional)
    # =============================
    if max_active_signatures is not None:
        active_preds = post_process_predictions(
            avg_probs,
            avg_thresholds,
            max_active_signatures
        )

    active_preds_binary = active_preds.astype(int)

    # =============================
    # 11. Save results
    # =============================
    predictions_csv_file = os.path.join(pred_output_dir, "predictions.csv")
    predictions_df = pd.DataFrame(
        active_preds_binary.T,
        index=signature_names,
        columns=sample_ids
    )

    predictions_df.to_csv(predictions_csv_file)
    
    probs_df = pd.DataFrame(
        avg_probs.T,
        index=signature_names,
        columns=sample_ids
    )
    probs_df.to_csv(
        os.path.join(pred_output_dir, "probabilities.csv")
    )

    thresholds_df = pd.DataFrame({
        "signature": signature_names,
        "threshold": avg_thresholds
    })
    thresholds_df.to_csv(
        os.path.join(pred_output_dir, "thresholds.csv"),
        index=False
    )

    print(f"\n✅ Results saved to: {pred_output_dir}")

    return predictions_csv_file


# =============================
# CLI entry point
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Predict active mutational signatures"
    )

    parser.add_argument("--mutation_file", required=True)
    parser.add_argument("--signature_file", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--model_type", choices=["SBS", "DBS", "ID"], default="SBS")
    parser.add_argument("--output_dir", default="output")

    
    parser.add_argument(
        "--max_active_signatures",
        type=int,
        default=7,
        help="Maximum number of active signatures per sample (default:7)"
    )

    args = parser.parse_args()

    extract_predictions(
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        base_model_dir=args.base_model_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
        max_active_signatures=args.max_active_signatures
    )


if __name__ == "__main__":
    main()