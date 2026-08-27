
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd


# =========================================================
# PIPELINE
# =========================================================
def run_pipeline(
    model_type,
    mutation_file,
    signature_file,
    base_model_dir,
    output_dir,
    cosine_threshold=0.95,
    probability_threshold=0.05,
    min_contribution=0.05,
    min_improvement=1e-2,
    max_active_signatures=7,
):
    """Run AMuSA prediction, refinement, replacement, and saving."""

    from AMuSA.exposure import estimate_exposure_from_predictions
    from AMuSA.prediction import extract_predictions
    from AMuSA.refinement_predictions import refine_low_cosine_predictions

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # Step 1: Initial signature prediction
    # =====================================================
    print("\nStep 1 -> Initial prediction")

    predictions_file = extract_predictions(
        mutation_file=mutation_file,
        signature_file=signature_file,
        base_model_dir=base_model_dir,
        model_type=model_type,
        output_dir=output_dir,
        max_active_signatures=max_active_signatures,
    )

    # Read and preserve the original Step 1 result.
    initial_predictions = pd.read_csv(predictions_file, index_col=0)
    initial_predictions.index = initial_predictions.index.astype(str)
    initial_predictions.columns = initial_predictions.columns.astype(str)

    # =====================================================
    # Step 2: Initial NNLS exposure and cosine QC
    # =====================================================
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

    # =====================================================
    # Step 3: Refine only low-cosine samples
    # =====================================================
    if not low_cosine_samples:
        print("\nNo low-cosine samples -> refinement skipped")
    else:
        print("\nStep 3 -> Low-cosine refinement")

        refinement_output = refine_low_cosine_predictions(
            low_cosine_catalog_file=step2_out["low_cosine_catalog_file"],
            mutation_signature_file=signature_file,

            # 直接使用 Step 2 第一次 NNLS 的浮点 exposure。
            # exposure > 0 的 signature 视为第一次真正使用，
            # 其余 signature 只要概率超过阈值即可进入候选池。
            original_exposures_float=initial_exposure_float,

            base_model_dir=base_model_dir,
            model_type=model_type,
            output_dir=output_dir,
            probability_threshold=probability_threshold,
            min_contribution=min_contribution,
            target_cosine=cosine_threshold,
            min_improvement=min_improvement,
            max_active_signatures=max_active_signatures,
        )

        # =================================================
        # Step 4: Obtain the refined results
        # =================================================
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

    # =====================================================
    # Step 5: Replace samples in a copy of the initial result
    # =====================================================
    print("\nStep 5 -> Replace improved samples and save as new results")

    # Copy the initial results. The Step 1 and Step 2 files remain unchanged.
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

            # Replace the entire prediction column in the copied Step 1 result.
            # Signatures absent from the refined result are filled with zero.
            replaced_predictions.loc[:, sample_name] = (
                refined_predictions[sample_name]
                .reindex(replaced_predictions.index)
                .fillna(0)
            )

            # Replace the entire exposure column in the copied Step 2 result.
            replaced_exposure_float.loc[:, sample_name] = (
                refined_exposure_float[sample_name]
                .reindex(replaced_exposure_float.index)
                .fillna(0)
            )

            replaced_samples.append(sample_name)

    # Standardize output data types.
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

    # Save to new files without overwriting the original results.
    final_output_dir = os.path.join(output_dir, "results", model_type)
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

    print("\nPipeline completed")
    print(f"Original Step 1 result kept at: {predictions_file}")
    print(f"Replaced predictions saved to: {replaced_predictions_file}")
    print(
        "Replaced exposure float saved to: "
        f"{replaced_exposure_float_file}"
    )
    print(f"Replaced exposure saved to: {replaced_exposure_file}")

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
    parser = argparse.ArgumentParser(description="AMuSA pipeline runner")

    parser.add_argument(
        "--type",
        dest="model_type",
        choices=["SBS", "DBS", "ID"],
        required=True,
    )
    parser.add_argument("--mutation_file", required=True)
    parser.add_argument("--signature_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_dir", required=True)

    parser.add_argument(
        "--cosine_threshold",
        type=float,
        default=0.95,
        help="Refine samples whose cosine similarity is below this value.",
    )
    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum model probability for a signature to enter "
            "the candidate pool."
        ),
    )
    parser.add_argument(
        "--min_contribution",
        type=float,
        default=0.05,
        help=(
            "Minimum NNLS contribution fraction retained in "
            "the refined solution."
        ),
    )
    parser.add_argument(
        "--min_improvement",
        type=float,
        default=1e-2,
        help=(
            "Minimum cosine improvement required to replace "
            "the original result."
        ),
    )
    parser.add_argument(
        "--max_active_signatures",
        type=int,
        default=7,
        help="Maximum number of active signatures in each final sample.",
    )

    args = parser.parse_args()

    run_pipeline(
        model_type=args.model_type,
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        base_model_dir=args.base_model_dir,
        output_dir=args.output_dir,
        cosine_threshold=args.cosine_threshold,
        probability_threshold=args.probability_threshold,
        min_contribution=args.min_contribution,
        min_improvement=args.min_improvement,
        max_active_signatures=args.max_active_signatures,
    )


if __name__ == "__main__":
    main()