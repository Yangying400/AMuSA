#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import nnls as scipy_nnls


# =========================================================
# Core Function
# =========================================================
def estimate_exposure_from_predictions(
    mutation_catalog_file,
    signature_matrix_file,
    predictions_csv_file,
    output_prefix
):
    """
    Estimate signature exposures using NNLS with post-filtering and refitting.
    """

    # =============================
    # 1. Load data
    # =============================
    mutation_catalog = pd.read_csv(mutation_catalog_file, index_col=0)
    signature_matrix = pd.read_csv(signature_matrix_file, index_col=0)
    predictions = pd.read_csv(predictions_csv_file, index_col=0)

    common_samples = [s for s in predictions.columns if s in mutation_catalog.columns]
    common_signatures = [s for s in predictions.index if s in signature_matrix.columns]

    if not common_samples:
        raise ValueError("No common samples found")
    if not common_signatures:
        raise ValueError("No common signatures found")

    print(f"Common samples: {len(common_samples)}, signatures: {len(common_signatures)}")

    X = mutation_catalog[common_samples].values
    W = signature_matrix[common_signatures].values
    exposures = np.zeros((len(common_signatures), len(common_samples)))

    # =============================
    # 2. First NNLS fitting
    # =============================
    for i, sample_name in enumerate(common_samples):
        sample_counts = X[:, i]

        if sample_counts.sum() == 0:
            continue

        active_mask = predictions[sample_name] == 1
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        active_sig_profiles = W[:, active_indices]

        weights = 1 / np.sqrt(sample_counts + 1e-6)
        weights = np.clip(weights, 0.001, 0.1)

        A_weighted = active_sig_profiles * weights[:, np.newaxis]
        B_weighted = sample_counts * weights

        coeffs, _ = scipy_nnls(A_weighted, B_weighted)
        exposures[active_indices, i] = np.maximum(coeffs, 1)

    exposures_df = pd.DataFrame(exposures, index=common_signatures, columns=common_samples)

    raw_file_first = f"{output_prefix}_first_raw.dat"
    exposures_df.to_csv(raw_file_first)
    print(f"First NNLS raw result saved: {raw_file_first}")

    # =============================
    # 3. Filter low contribution (<2%)
    # =============================
    exposures_df_norm = exposures_df.div(exposures_df.sum(axis=0), axis=1)

    threshold = 0.02
    mask = exposures_df_norm < threshold
    exposures_df_norm[mask] = 0.0

    # =============================
    # 4. Apply linked signature rules
    # =============================
    predictions_filtered = predictions.copy()
    predictions_filtered = predictions_filtered.where(~mask, 0.0)

    signature_names = list(predictions_filtered.index)

    linked_groups = [
        ["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"],
        ["SBS2", "SBS13"],
        ["SBS10a", "SBS10b"],
        ["SBS17a", "SBS17b"]
    ]
    exclude_if_7_group = ["SBS3", "SBS8"]

    for sample_name in common_samples:
        active_preds = predictions_filtered[sample_name].values

        for group in linked_groups:
            group_idx = [signature_names.index(sig) for sig in group if sig in signature_names]

            if any(active_preds[idx] > 0 for idx in group_idx):

                # Activate entire group
                for idx in group_idx:
                    active_preds[idx] = 1.0

                # Exclusion rule for SBS7 group
                if set(group) == set(["SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS38"]):
                    for sig in exclude_if_7_group:
                        if sig in signature_names:
                            exclude_idx = signature_names.index(sig)
                            active_preds[exclude_idx] = 0.0

        predictions_filtered[sample_name] = active_preds

    # =============================
    # 5. Second NNLS fitting
    # =============================
    exposures_refit = np.zeros_like(exposures_df.values)

    for i, sample_name in enumerate(common_samples):
        sample_counts = X[:, i]

        active_mask = predictions_filtered[sample_name] == 1
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        active_sig_profiles = W[:, active_indices]

        weights = 1 / np.sqrt(sample_counts + 1e-6)
        weights = np.clip(weights, 0.001, 0.1)

        A_weighted = active_sig_profiles * weights[:, np.newaxis]
        B_weighted = sample_counts * weights

        coeffs, _ = scipy_nnls(A_weighted, B_weighted)
        exposures_refit[active_indices, i] = np.maximum(coeffs, 1)

    exposures_refit_df = pd.DataFrame(
        exposures_refit,
        index=common_signatures,
        columns=common_samples
    )

    exposures_refit_norm = exposures_refit_df.div(
        exposures_refit_df.sum(axis=0),
        axis=1
    )

    norm_file_second = f"{output_prefix}_second_normalized.dat"
    exposures_refit_norm.to_csv(norm_file_second)
    print(f"Second NNLS normalized result saved: {norm_file_second}")

    # =============================
    # 6. Second filtering (<2%)
    # =============================
    mask_refit = exposures_refit_norm < threshold
    exposures_refit_norm[mask_refit] = 0.0

    # =============================
    # 7. Restore to mutation counts
    # =============================
    col_sums = exposures_refit_df.sum(axis=0)
    exposures_refit_restored = exposures_refit_norm.mul(col_sums, axis=1)

    norm_restored_file = f"{output_prefix}_second_restored.dat"
    exposures_refit_restored.to_csv(norm_restored_file)
    print(f"Restored exposure (counts) saved: {norm_restored_file}")

    norm_filtered_file = f"{output_prefix}_second_normalized_filtered.dat"
    exposures_refit_norm.to_csv(norm_filtered_file)
    print(f"Filtered normalized exposure saved: {norm_filtered_file}")

    return exposures_refit_restored


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Exposure estimation from predictions")

    parser.add_argument("--mutation_catalog_file", required=True)
    parser.add_argument("--signature_matrix_file", required=True)
    parser.add_argument("--predictions_csv_file", required=True)
    parser.add_argument("--output_prefix", required=True)

    return parser.parse_args()


# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    estimate_exposure_from_predictions(
        mutation_catalog_file=args.mutation_catalog_file,
        signature_matrix_file=args.signature_matrix_file,
        predictions_csv_file=args.predictions_csv_file,
        output_prefix=args.output_prefix
    )