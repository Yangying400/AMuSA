#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import pandas as pd


# =========================================================
# PACKAGE PATHS
# =========================================================

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PACKAGE_DIR / "models"


# =========================================================
# DEFAULT PARAMETERS
# =========================================================

DEFAULT_MIN_IMPROVEMENT = {
    "SBS": 0.04,
    "DBS": 0.04,
    "ID": 0.03,
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _resolve_model_dir(base_model_dir=None):
    """
    Resolve the AMuSA pretrained model directory.

    If base_model_dir is not provided, models bundled with
    the installed AMuSA package are used.
    """
    if base_model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    else:
        model_dir = Path(base_model_dir).expanduser().resolve()

    if not model_dir.exists():
        raise FileNotFoundError(
            "AMuSA model directory was not found:\n"
            f"{model_dir}\n\n"
            "Please ensure that pretrained models are installed "
            "or provide base_model_dir explicitly."
        )

    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"AMuSA model path is not a directory: {model_dir}"
        )

    return str(model_dir)


def _resolve_min_improvement(model_type, min_improvement=None):
    """
    Resolve mutation-type-specific minimum cosine improvement.

    Default values:
        SBS = 0.04
        DBS = 0.04
        ID  = 0.03

    A user-specified value overrides these defaults.
    """
    model_type = str(model_type).upper()

    if model_type not in DEFAULT_MIN_IMPROVEMENT:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Expected one of: SBS, DBS, ID."
        )

    if min_improvement is None:
        return DEFAULT_MIN_IMPROVEMENT[model_type]

    min_improvement = float(min_improvement)

    if min_improvement < 0:
        raise ValueError(
            "min_improvement must be greater than or equal to 0."
        )

    return min_improvement


def _validate_fraction_parameter(name, value):
    """
    Validate parameters expected to be within [0, 1].
    """
    value = float(value)

    if not 0 <= value <= 1:
        raise ValueError(
            f"{name} must be between 0 and 1, got {value}."
        )

    return value


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(
    model_type,
    mutation_file,
    signature_file,
    output_dir,
    base_model_dir=None,
    cosine_threshold=0.95,
    probability_threshold=0.05,
    min_contribution=0.05,
    min_improvement=None,
    max_active_signatures=7,
):
    """
    Run the complete AMuSA mutational signature assignment pipeline.

    Parameters
    ----------
    model_type : {"SBS", "DBS", "ID"}
        Mutation type to analyze.

    mutation_file : str
        Path to the mutation catalog.

    signature_file : str
        Path to the reference signature matrix.

    output_dir : str
        Directory in which results will be saved.

    base_model_dir : str or None, optional
        Directory containing pretrained AMuSA models.
        If None, models bundled with the AMuSA package are used.

    cosine_threshold : float, optional
        Samples with reconstruction cosine similarity below this
        value enter the refinement procedure. Default: 0.95.

    probability_threshold : float, optional
        Minimum predicted probability required for a signature
        to enter the refinement candidate pool. Default: 0.05.

    min_contribution : float, optional
        Minimum NNLS contribution fraction retained during refinement.
        Default: 0.05.

    min_improvement : float or None, optional
        Minimum cosine improvement required to accept a refined
        assignment.

        If None, mutation-type-specific defaults are used:
            SBS = 0.04
            DBS = 0.04
            ID  = 0.03

    max_active_signatures : int, optional
        Maximum number of active signatures retained per sample.
        Default: 7.

    Returns
    -------
    dict
        Dictionary containing initial and final prediction/exposure
        results and corresponding output file paths.
    """

    model_type = str(model_type).upper()

    if model_type not in {"SBS", "DBS", "ID"}:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            "Expected one of: SBS, DBS, ID."
        )

    base_model_dir = _resolve_model_dir(base_model_dir)

    min_improvement = _resolve_min_improvement(
        model_type=model_type,
        min_improvement=min_improvement,
    )

    cosine_threshold = _validate_fraction_parameter(
        "cosine_threshold",
        cosine_threshold,
    )

    probability_threshold = _validate_fraction_parameter(
        "probability_threshold",
        probability_threshold,
    )

    min_contribution = _validate_fraction_parameter(
        "min_contribution",
        min_contribution,
    )

    max_active_signatures = int(max_active_signatures)

    if max_active_signatures < 1:
        raise ValueError(
            "max_active_signatures must be at least 1."
        )

    mutation_file = str(Path(mutation_file).expanduser())
    signature_file = str(Path(signature_file).expanduser())
    output_dir = str(Path(output_dir).expanduser())

    if not os.path.isfile(mutation_file):
        raise FileNotFoundError(
            f"Mutation catalog was not found: {mutation_file}"
        )

    if not os.path.isfile(signature_file):
        raise FileNotFoundError(
            f"Signature matrix was not found: {signature_file}"
        )

    os.makedirs(output_dir, exist_ok=True)

    print("\n========================================")
    print("AMuSA pipeline")
    print("========================================")
    print(f"Mutation type        : {model_type}")
    print(f"Mutation catalog     : {mutation_file}")
    print(f"Signature matrix     : {signature_file}")
    print(f"Model directory      : {base_model_dir}")
    print(f"Output directory     : {output_dir}")
    print(f"Cosine threshold     : {cosine_threshold}")
    print(f"Probability threshold: {probability_threshold}")
    print(f"Minimum contribution : {min_contribution}")
    print(f"Minimum improvement  : {min_improvement}")
    print(f"Max active signatures: {max_active_signatures}")
    print("========================================")

    from AMuSA.exposure import estimate_exposure_from_predictions
    from AMuSA.prediction import extract_predictions
    from AMuSA.refinement_predictions import (
        refine_low_cosine_predictions,
    )

    print("\nStep 1 -> Initial prediction")

    predictions_file = extract_predictions(
        mutation_file=mutation_file,
        signature_file=signature_file,
        base_model_dir=base_model_dir,
        model_type=model_type,
        output_dir=output_dir,
        max_active_signatures=max_active_signatures,
    )

    initial_predictions = pd.read_csv(predictions_file, index_col=0)
    initial_predictions.index = initial_predictions.index.astype(str)
    initial_predictions.columns = initial_predictions.columns.astype(str)

    print("\nStep 2 -> Initial exposure and cosine QC")

    step2_out = estimate_exposure_from_predictions(
        mutation_catalog_file=mutation_file,
        signature_matrix_file=signature_file,
        predictions_csv_file=predictions_file,
        output_dir=output_dir,
        model_type=model_type,
        cosine_threshold=cosine_threshold,
    )

    initial_exposure = step2_out["exposures"].copy()
    initial_exposure_float = step2_out["exposures_float"].copy()

    initial_exposure.index = initial_exposure.index.astype(str)
    initial_exposure.columns = initial_exposure.columns.astype(str)
    initial_exposure_float.index = initial_exposure_float.index.astype(str)
    initial_exposure_float.columns = initial_exposure_float.columns.astype(str)

    low_cosine_samples = [
        str(sample_name)
        for sample_name in step2_out["low_cosine_samples"]
    ]

    refinement_output = None
    refined_predictions = None
    refined_exposure_float = None
    improved_samples = []

    if not low_cosine_samples:
        print("\nNo low-cosine samples -> refinement skipped")
    else:
        print("\nStep 3 -> Low-cosine refinement")

        refinement_output = refine_low_cosine_predictions(
            low_cosine_catalog_file=step2_out["low_cosine_catalog_file"],
            mutation_signature_file=signature_file,
            original_exposures_float=initial_exposure_float,
            base_model_dir=base_model_dir,
            model_type=model_type,
            output_dir=output_dir,
            probability_threshold=probability_threshold,
            min_contribution=min_contribution,
            target_cosine=1.0,
            min_improvement=min_improvement,
            max_active_signatures=max_active_signatures,
        )

        print("\nStep 4 -> Obtain refined results")

        refined_predictions = refinement_output["predictions"].copy()
        refined_exposure_float = refinement_output["exposures_float"].copy()

        improved_samples = [
            str(sample_name)
            for sample_name in refinement_output["improved_samples"]
        ]

        refined_predictions.index = refined_predictions.index.astype(str)
        refined_predictions.columns = refined_predictions.columns.astype(str)
        refined_exposure_float.index = refined_exposure_float.index.astype(str)
        refined_exposure_float.columns = refined_exposure_float.columns.astype(str)

    print("\nStep 5 -> Replace improved samples and save final results")

    replaced_predictions = initial_predictions.copy()
    replaced_exposure_float = initial_exposure_float.copy()
    replaced_samples = []

    if refined_predictions is not None and refined_exposure_float is not None:
        for sample_name in improved_samples:
            missing_locations = []

            if sample_name not in replaced_predictions.columns:
                missing_locations.append("initial predictions")
            if sample_name not in replaced_exposure_float.columns:
                missing_locations.append("initial exposure")
            if sample_name not in refined_predictions.columns:
                missing_locations.append("refined predictions")
            if sample_name not in refined_exposure_float.columns:
                missing_locations.append("refined exposure")

            if missing_locations:
                print(
                    f"[Warning] Sample {sample_name} is missing from: "
                    f"{', '.join(missing_locations)}. Replacement skipped."
                )
                continue

            replaced_predictions.loc[:, sample_name] = (
                refined_predictions[sample_name]
                .reindex(replaced_predictions.index)
                .fillna(0)
            )

            replaced_exposure_float.loc[:, sample_name] = (
                refined_exposure_float[sample_name]
                .reindex(replaced_exposure_float.index)
                .fillna(0)
            )

            replaced_samples.append(sample_name)

    replaced_predictions = (
        replaced_predictions
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    replaced_exposure_float = (
        replaced_exposure_float
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )

    replaced_exposure = replaced_exposure_float.round(0).astype(int)

    print(
        f"Replaced {len(replaced_samples)} of "
        f"{len(low_cosine_samples)} low-cosine samples."
    )

    final_output_dir = os.path.join(
        output_dir,
        "results",
        model_type,
    )
    os.makedirs(final_output_dir, exist_ok=True)

    replaced_predictions_file = os.path.join(
        final_output_dir,
        f"{model_type}_replaced_predictions.csv",
    )

    replaced_exposure_float_file = os.path.join(
        final_output_dir,
        f"{model_type}_replaced_exposure_float.csv",
    )

    replaced_exposure_file = os.path.join(
        final_output_dir,
        f"{model_type}_replaced_exposure.csv",
    )

    replaced_predictions.to_csv(replaced_predictions_file)
    replaced_exposure_float.to_csv(replaced_exposure_float_file)
    replaced_exposure.to_csv(replaced_exposure_file)

    print("\n========================================")
    print("Pipeline completed")
    print("========================================")
    print(f"Original Step 1 result kept at: {predictions_file}")
    print(f"Final predictions saved to: {replaced_predictions_file}")
    print(
        "Final floating-point exposures saved to: "
        f"{replaced_exposure_float_file}"
    )
    print(f"Final rounded exposures saved to: {replaced_exposure_file}")

    return {
        "predictions_file": predictions_file,
        "initial_predictions": initial_predictions,
        "initial_exposure": initial_exposure,
        "initial_exposure_float": initial_exposure_float,
        "replaced_predictions": replaced_predictions,
        "replaced_exposure": replaced_exposure,
        "replaced_exposure_float": replaced_exposure_float,
        "replaced_predictions_file": replaced_predictions_file,
        "replaced_exposure_file": replaced_exposure_file,
        "replaced_exposure_float_file": replaced_exposure_float_file,
        "low_cosine_samples": low_cosine_samples,
        "improved_samples": replaced_samples,
        "refinement": refinement_output,
    }


# =========================================================
# COMMAND-LINE INTERFACE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="AMuSA mutational signature assignment pipeline"
    )

    parser.add_argument(
        "--type",
        dest="model_type",
        choices=["SBS", "DBS", "ID"],
        required=True,
        help="Mutation type.",
    )

    parser.add_argument(
        "--mutation_file",
        required=True,
        help="Input mutation catalog.",
    )

    parser.add_argument(
        "--signature_file",
        required=True,
        help="Reference signature matrix.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--base_model_dir",
        default=None,
        help=(
            "Directory containing pretrained AMuSA models. "
            "If omitted, models bundled with the installed "
            "AMuSA package are used."
        ),
    )

    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.95,
        help=(
            "Refine samples whose cosine similarity "
            "is below this value. Default: 0.95."
        ),
    )

    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum model probability for a signature "
            "to enter the candidate pool. Default: 0.05."
        ),
    )

    parser.add_argument(
        "--min_contribution",
        type=float,
        default=0.05,
        help=(
            "Minimum NNLS contribution fraction retained "
            "during refinement. Default: 0.05."
        ),
    )

    parser.add_argument(
        "--min_improvement",
        type=float,
        default=None,
        help=(
            "Minimum cosine improvement required to accept "
            "a refined assignment. "
            "If omitted, AMuSA uses 0.04 for SBS/DBS "
            "and 0.03 for ID."
        ),
    )

    parser.add_argument(
        "--max_active_signatures",
        type=int,
        default=7,
        help=(
            "Maximum number of active signatures "
            "retained per sample. Default: 7."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        model_type=args.model_type,
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        output_dir=args.output_dir,
        base_model_dir=args.base_model_dir,
        cosine_threshold=args.cosine_threshold,
        probability_threshold=args.probability_threshold,
        min_contribution=args.min_contribution,
        min_improvement=args.min_improvement,
        max_active_signatures=args.max_active_signatures,
    )


if __name__ == "__main__":
    main()