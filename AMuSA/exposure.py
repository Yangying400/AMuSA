#!/usr/bin/env python3
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import nnls as scipy_nnls


def compute_cosine(original_catalog, reconstructed_catalog):
    """Calculate sample-wise cosine similarity for aligned catalogs."""
    common_mutation_types = original_catalog.index.intersection(
        reconstructed_catalog.index
    )
    common_samples = original_catalog.columns.intersection(
        reconstructed_catalog.columns
    )

    cosine_values = {}

    for sample_name in common_samples:
        original_vector = pd.to_numeric(
            original_catalog.loc[common_mutation_types, sample_name],
            errors="coerce",
        ).fillna(0).values.astype(float)

        reconstructed_vector = pd.to_numeric(
            reconstructed_catalog.loc[common_mutation_types, sample_name],
            errors="coerce",
        ).fillna(0).values.astype(float)

        denominator = (
            np.linalg.norm(original_vector)
            * np.linalg.norm(reconstructed_vector)
        )

        cosine_values[sample_name] = (
            0.0
            if denominator == 0
            else float(
                np.dot(original_vector, reconstructed_vector) / denominator
            )
        )

    return cosine_values


def estimate_exposure_from_predictions(
    mutation_catalog_file,
    signature_matrix_file,
    predictions_csv_file,
    output_dir,
    model_type,
    cosine_threshold=0.95,
):
    """
    Estimate signature exposures with ordinary NNLS and perform cosine QC.

    The binary prediction matrix determines which signature profiles enter
    NNLS for each sample.
    """
    os.makedirs(output_dir, exist_ok=True)

    mutation_catalog = pd.read_csv(
        mutation_catalog_file,
        index_col=0
    )

    signature_matrix = pd.read_csv(
        signature_matrix_file,
        index_col=0
    )

    predictions = pd.read_csv(
        predictions_csv_file,
        index_col=0
    )

    original_samples = predictions.columns.tolist()
    original_signatures = predictions.index.tolist()

    common_samples = [
        sample_name
        for sample_name in original_samples
        if sample_name in mutation_catalog.columns
    ]

    if not common_samples:
        raise ValueError("No common samples found")

    common_signatures = [
        signature_name
        for signature_name in original_signatures
        if signature_name in signature_matrix.columns
    ]

    if not common_signatures:
        raise ValueError("No common signatures found")

    print("Common signatures:", common_signatures)

    mutation_catalog = mutation_catalog[common_samples]

    predictions = predictions.loc[
        common_signatures,
        common_samples
    ]

    signature_matrix = signature_matrix[
        common_signatures
    ]

    X = mutation_catalog.values
    W = signature_matrix.values

    exposures = np.zeros(
        (
            len(common_signatures),
            len(common_samples)
        )
    )

    for i, sample_name in enumerate(common_samples):
        sample_counts = X[:, i]

        if sample_counts.sum() == 0:
            continue

        active_mask = predictions[sample_name] == 1
        active_indices = np.where(active_mask)[0]

        if len(active_indices) == 0:
            continue

        active_sig_profiles = W[:, active_indices]

        coeffs, _ = scipy_nnls(
            active_sig_profiles,
            sample_counts
        )

        exposures[active_indices, i] = np.maximum(
            coeffs,
            0
        )

    exposures_float_df = pd.DataFrame(
        exposures,
        index=common_signatures,
        columns=common_samples
    )

    exposures_df = (
        exposures_float_df
        .round(0)
        .astype(int)
    )

    exposure_file = os.path.join(
        output_dir,
        f"{model_type}_exposures.csv"
    )

    exposures_df.to_csv(exposure_file)

    print(f"[Exposure] saved: {exposure_file}")

    reconstructed = np.dot(
        signature_matrix.values,
        exposures_df.values
    )

    recon_df = pd.DataFrame(
        reconstructed,
        index=signature_matrix.index,
        columns=common_samples
    )

    gt_df = mutation_catalog[common_samples]

    cosine_values = compute_cosine(
        gt_df,
        recon_df
    )

    low_cosine_samples = [
        sample_name
        for sample_name, cosine_value in cosine_values.items()
        if cosine_value < cosine_threshold
    ]

    print(
        f"[Cosine QC] low-confidence samples: "
        f"{len(low_cosine_samples)}"
    )

    cosine_df = pd.DataFrame.from_dict(
        cosine_values,
        orient="index",
        columns=["cosine"]
    )

    cosine_file = os.path.join(
        output_dir,
        f"{model_type}_cosine.csv"
    )

    cosine_df.to_csv(cosine_file)

    low_file = os.path.join(
        output_dir,
        f"{model_type}_low_cosine_samples.txt"
    )

    with open(low_file, "w") as file_obj:
        for sample_name in low_cosine_samples:
            file_obj.write(sample_name + "\n")

    low_cosine_catalog_file = os.path.join(
        output_dir,
        f"{model_type}_low_cosine_catalog.csv"
    )

    low_cosine_predictions_file = os.path.join(
        output_dir,
        f"{model_type}_low_cosine_predictions.csv"
    )

    if low_cosine_samples:
        low_cosine_catalog = mutation_catalog.loc[
            :,
            low_cosine_samples
        ]

        low_cosine_predictions = predictions.loc[
            :,
            low_cosine_samples
        ]
    else:
        low_cosine_catalog = mutation_catalog.iloc[:, 0:0]
        low_cosine_predictions = predictions.iloc[:, 0:0]

    low_cosine_catalog.to_csv(
        low_cosine_catalog_file
    )

    low_cosine_predictions.to_csv(
        low_cosine_predictions_file
    )

    print(f"[Cosine QC] saved: {cosine_file}")
    print(f"[Cosine QC] saved: {low_file}")
    print(f"[Cosine QC] saved: {low_cosine_catalog_file}")
    print(f"[Cosine QC] saved: {low_cosine_predictions_file}")

    return {
        "exposures": exposures_df,
        "exposures_float": exposures_float_df,
        "cosine_values": cosine_values,
        "low_cosine_samples": low_cosine_samples,
        "exposure_file": exposure_file,
        "cosine_file": cosine_file,
        "low_cosine_file": low_file,
        "low_cosine_catalog_file": low_cosine_catalog_file,
        "low_cosine_predictions_file": low_cosine_predictions_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate mutational signature exposures using ordinary NNLS "
            "and identify low-cosine samples."
        )
    )

    parser.add_argument(
        "--mutation_file",
        required=True
    )

    parser.add_argument(
        "--signature_file",
        required=True
    )

    parser.add_argument(
        "--predictions_file",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--model_type",
        choices=["SBS", "DBS", "ID"],
        required=True,
    )

    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.95,
    )

    args = parser.parse_args()

    estimate_exposure_from_predictions(
        mutation_catalog_file=args.mutation_file,
        signature_matrix_file=args.signature_file,
        predictions_csv_file=args.predictions_file,
        output_dir=args.output_dir,
        model_type=args.model_type,
        cosine_threshold=args.cosine_threshold,
    )


if __name__ == "__main__":
    main()