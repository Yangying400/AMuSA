#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.optimize import nnls as scipy_nnls


# =========================================================
# Step5: Adaptive Refinement (PURE PIPELINE FUNCTION)
# =========================================================
def refine_low_cosine_exposure(
    low_cosine_catalog_file,          
    signature_matrix,
    low_cosine_predictions_file,      
    model_type,
    output_dir,
    adaptive_threshold=0.02
):

    # =====================================================
    # 1. Inputs 
    # =====================================================
    mutation_catalog = pd.read_csv(low_cosine_catalog_file, index_col=0)
    predictions = pd.read_csv(low_cosine_predictions_file, index_col=0)

    signature_matrix = signature_matrix

    common_samples = [
        s for s in predictions.columns
        if s in mutation_catalog.columns
    ]

    common_signatures = [
        s for s in predictions.index
        if s in signature_matrix.columns
    ]

    if not common_samples:
        raise ValueError("No common samples found")

    if not common_signatures:
        raise ValueError("No common signatures found")

    print(f"[Step5] low-cosine samples: {len(common_samples)}")
    print(f"[Step5] signatures: {len(common_signatures)}")

    # =====================================================
    # 2. Prepare matrices
    # =====================================================
    X = mutation_catalog[common_samples].values
    W = signature_matrix[common_signatures].values

    exposures = np.zeros((len(common_signatures), len(common_samples)))

    # =====================================================
    # 3. First NNLS fitting
    # =====================================================
    for i, sample_name in enumerate(common_samples):

        sample_counts = X[:, i]

        if sample_counts.sum() == 0:
            continue

        active_mask = predictions[sample_name] == 1
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        active_sig_profiles = W[:, active_indices]

        weights = np.clip(
            1 / np.sqrt(sample_counts + 1e-6),
            0.001,
            0.1
        )

        A = active_sig_profiles * weights[:, np.newaxis]
        B = sample_counts * weights

        coeffs, _ = scipy_nnls(A, B)

        exposures[active_indices, i] = np.maximum(coeffs, 0)

    exposures_df = pd.DataFrame(
        exposures,
        index=common_signatures,
        columns=common_samples
    )

    first_raw_file = f"{output_dir}_first_refinement_raw.csv"
    exposures_df.to_csv(first_raw_file)

    print(f"[Step5] first refinement saved: {first_raw_file}")

    # =====================================================
    # 4. Normalize exposure
    # =====================================================
    col_sums = exposures_df.sum(axis=0).replace(0, np.nan)
    exposures_norm = exposures_df.div(col_sums, axis=1).fillna(0)

    mask = exposures_norm < adaptive_threshold
    exposures_norm[mask] = 0.0

    # =====================================================
    # 5. Biological linkage rules
    # =====================================================
    predictions_filtered = predictions.copy()
    predictions_filtered = predictions_filtered.where(~mask, 0.0)

    sig_names = list(predictions_filtered.index)

    linked_groups = [
        ["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"],
        ["SBS2", "SBS13"],
        ["SBS10a", "SBS10b"],
        ["SBS17a", "SBS17b"]
    ]

    exclude_if_7 = ["SBS3", "SBS8"]

    for sample in common_samples:

        vec = predictions_filtered[sample].values

        for group in linked_groups:

            idx = [
                sig_names.index(s)
                for s in group
                if s in sig_names
            ]

            if any(vec[i] > 0 for i in idx):

                for i in idx:
                    vec[i] = 1.0

                if set(group) == set(["SBS7a","SBS7b","SBS7c","SBS7d","SBS38"]):

                    for s in exclude_if_7:
                        if s in sig_names:
                            vec[sig_names.index(s)] = 0.0

        predictions_filtered[sample] = vec

    # =====================================================
    # 6. Second NNLS refitting
    # =====================================================
    exposures_refit = np.zeros_like(exposures_df.values)

    for i, sample_name in enumerate(common_samples):

        sample_counts = X[:, i]

        if sample_counts.sum() == 0:
            continue

        active_mask = predictions_filtered[sample_name] == 1
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        active_sig_profiles = W[:, active_indices]

        weights = np.clip(
            1 / np.sqrt(sample_counts + 1e-6),
            0.001,
            0.1
        )

        A = active_sig_profiles * weights[:, np.newaxis]
        B = sample_counts * weights

        coeffs, _ = scipy_nnls(A, B)

        exposures_refit[active_indices, i] = np.maximum(coeffs, 0)

    exposures_refit_df = pd.DataFrame(
        exposures_refit,
        index=common_signatures,
        columns=common_samples
    )

    # =====================================================
    # 7. Normalize + restore
    # =====================================================
    col_sums = exposures_refit_df.sum(axis=0).replace(0, np.nan)
    norm = exposures_refit_df.div(col_sums, axis=1).fillna(0)

    norm[norm < adaptive_threshold] = 0.0

    restored = norm.mul(
        exposures_refit_df.sum(axis=0),
        axis=1
    )

    # =====================================================
    # 8. Save output
    # =====================================================
    out_file = os.path.join(output_dir, f"{model_type}_low_cosine_refined_exposure.csv")
    restored.to_csv(out_file)

    print(f"[Step5] refined exposure saved: {out_file}")

    # =====================================================
    # 9. Return
    # =====================================================
    return {
        "refined_exposure": restored
    }