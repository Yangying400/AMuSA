#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import nnls as scipy_nnls


def estimate_exposure_from_predictions(
    mutation_catalog_file,
    signature_matrix_file,
    predictions_csv_file,
    output_dir,
    model_type,
    weight_min=0.001,
    weight_max=0.1,
    epsilon=1e-6
):
    """
    Estimate mutational signature exposures using weighted NNLS.

    Parameters
    ----------
    mutation_catalog_file : str
        Path to mutation catalog (features x samples)

    signature_matrix_file : str
        Path to signature matrix (features x signatures)

    predictions_csv_file : str
        Path to predicted binary signature matrix

    output_dir : str
        Base output directory

    model_type : str
        One of ["SBS", "DBS", "ID"]

    weight_min : float
        Minimum weight (to avoid extreme scaling)

    weight_max : float
        Maximum weight

    epsilon : float
        Small value to avoid division by zero and control weight smoothness
    """

    # =============================
    # 1. Load data
    # =============================
    mutation_catalog = pd.read_csv(mutation_catalog_file, index_col=0)
    signature_matrix = pd.read_csv(signature_matrix_file, index_col=0)
    predictions = pd.read_csv(predictions_csv_file, index_col=0)

    original_samples = predictions.columns.tolist()
    original_signatures = predictions.index.tolist()

    # =============================
    # 2. Find intersection
    # =============================
    common_samples = [s for s in original_samples if s in mutation_catalog.columns]
    if not common_samples:
        raise ValueError("No common samples found")

    common_signatures = [s for s in original_signatures if s in signature_matrix.columns]
    if not common_signatures:
        raise ValueError("No common signatures found")

    print("Common signatures:", common_signatures)

    # =============================
    # 3. Subset data
    # =============================
    mutation_catalog = mutation_catalog[common_samples]
    predictions = predictions.loc[common_signatures, common_samples]
    signature_matrix = signature_matrix[common_signatures]

    X = mutation_catalog.values
    W = signature_matrix.values

    exposures = np.zeros((len(common_signatures), len(common_samples)))

    # =============================
    # 4. Weighted NNLS per sample
    # =============================
    for i, sample_name in enumerate(common_samples):
        sample_counts = X[:, i]

        if sample_counts.sum() == 0:
            continue

        # Active signatures
        active_mask = predictions[sample_name] == 1
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        active_sig_profiles = W[:, active_indices]

        # -----------------------------
        # Weighted NNLS (core)
        # -----------------------------
        weights = 1.0 / np.sqrt(sample_counts + epsilon)
        weights = np.clip(weights, weight_min, weight_max)

        A_weighted = active_sig_profiles * weights[:, np.newaxis]
        B_weighted = sample_counts * weights

        coeffs, _ = scipy_nnls(A_weighted, B_weighted)

        exposures[active_indices, i] = np.maximum(coeffs, 0)

    # =============================
    # 5. Convert to integer
    # =============================
    exposures_df = pd.DataFrame(
        exposures,
        index=common_signatures,
        columns=common_samples
    )

    exposures_df = exposures_df.round(0).astype(int)

    # =============================
    # 6. Output path
    # =============================
    exposure_dir = os.path.join(
        output_dir,
        "results",
        model_type,
        
    )
    os.makedirs(exposure_dir, exist_ok=True)

    output_file = os.path.join(
        exposure_dir,
        f"{model_type}_exposure.csv"
    )

    # =============================
    # 7. Save
    # =============================
    exposures_df.to_csv(output_file)

    print(f"\n✅ Exposure saved to: {output_file}")

   # =========================================================

    
    return exposures_df


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(
        description="Estimate exposure using weighted NNLS"
    )

    parser.add_argument("--mutation_file", required=True)
    parser.add_argument("--signature_file", required=True)
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", choices=["SBS", "DBS", "ID"], required=True)

    # Weight parameters
    parser.add_argument("--weight_min", type=float, default=0.001)
    parser.add_argument("--weight_max", type=float, default=0.1)

    # Epsilon (NEW)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Small value to avoid division by zero"
    )

    args = parser.parse_args()

    estimate_exposure_from_predictions(
        mutation_catalog_file=args.mutation_file,
        signature_matrix_file=args.signature_file,
        predictions_csv_file=args.predictions_file,
        output_dir=args.output_dir,
        model_type=args.model_type,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        epsilon=args.epsilon
    )


if __name__ == "__main__":
    main()
    
