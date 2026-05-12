#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch

from AMuSA.trainer import load_model, get_encoded_features
from AMuSA.data_loader import load_data
from AMuSA.utils import post_process_predictions


# =========================================================
# CORE FUNCTION
# =========================================================
def refine_low_cosine_predictions(
    low_cosine_catalog_file,
    mutation_signature_file,
    base_model_dir,
    model_type,
    output_dir,
    max_active_signatures=10,
    exposure_threshold=0.05
):

    # =====================================================
    # Output directory
    # =====================================================
    pred_output_dir = os.path.join(output_dir,model_type, "low_cosine_refinement")
    os.makedirs(pred_output_dir, exist_ok=True)

    # =====================================================
    # Load model ensemble
    # =====================================================
    model_dir = os.path.join(base_model_dir, f"{model_type}_models")

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")
    model_paths = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pth")
    ]

    if len(model_paths) == 0:
        raise ValueError(f"No model found in {model_dir}")

    # Load first model (for scaler + signature names)
    first_model = torch.load(model_paths[0], map_location="cpu",weights_only=False)
    scaler = first_model["scaler"]
    signature_names = first_model["signature_names"]

    # =====================================================
    # Load mutation data (low cosine samples)
    # =====================================================
    X_test, _, _, sample_ids, _ = load_data(
        mutation_file=low_cosine_catalog_file,
        exposure_file=None,
        signature_file=mutation_signature_file,
        scaler=scaler,
        train=False
    )

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

    avg_probs = np.mean(all_probs, axis=0)

    # =====================================================
    # Thresholding
    # =====================================================
    avg_thresholds = np.ones(len(signature_names)) * exposure_threshold

    active_preds = (avg_probs >= avg_thresholds).astype(float)

    # =====================================================
    # Limit active signatures
    # =====================================================
    if max_active_signatures is not None:
        active_preds = post_process_predictions(
            avg_probs,
            avg_thresholds,
            max_active_signatures
        )

    # =====================================================
    # Biological constraints
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

                    # activate group
                    for idx2 in group_idx:
                        active_preds[i, idx2] = 1.0

                    # special rule
                    if group == ["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"]:
                        for sig in exclude_if_7_group:
                            if sig in signature_names:
                                exclude_idx = signature_names.index(sig)
                                active_preds[i, exclude_idx] = 0.0
                    break

    active_preds_binary = active_preds.astype(int)

    # =====================================================
    # Outputs
    # =====================================================
    predictions_df = pd.DataFrame(
        active_preds_binary.T,
        index=signature_names,
        columns=sample_ids
    )

    probs_df = pd.DataFrame(
        avg_probs.T,
        index=signature_names,
        columns=sample_ids
    )

    thresholds_df = pd.DataFrame({
        "signature": signature_names,
        "threshold": avg_thresholds
    })

    predictions_file = os.path.join(pred_output_dir, "low_cosine_predictions.csv")
    probs_file       = os.path.join(pred_output_dir, "low_cosine_probabilities.csv")
    thresholds_file  = os.path.join(pred_output_dir, "low_cosine_thresholds.csv")

    # 
    predictions_df.to_csv(predictions_file)
    probs_df.to_csv(probs_file)
    thresholds_df.to_csv(thresholds_file, index=False)

    # 
    print(f"Saved predictions to: {predictions_file}")
    print(f"Saved probabilities to: {probs_file}")
    print(f"Saved thresholds to: {thresholds_file}")

    
    # =====================================================
    # Return (pipeline interface)
    # =====================================================
    return {
        "predictions": predictions_df,
        "probabilities": probs_df,
        "thresholds": thresholds_df
    }


