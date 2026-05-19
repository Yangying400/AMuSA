#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
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
    cosine_threshold=0.9,
    filtering_threshold=0.02,
    weight_min=0.001,
    weight_max=0.1,
    epsilon=1e-6,
    max_active_signatures=7,
    refine_thresholds=0.1,
    config_path=None
):
    from AMuSA.prediction import extract_predictions
    from AMuSA.exposure import estimate_exposure_from_predictions
    from AMuSA.refinement_predictions import refine_low_cosine_predictions
    from AMuSA.refinement_exposure import refine_low_cosine_exposure

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # Step2
    # =====================================================
    print("\nStep2 -> Prediction")

    predictions_file = extract_predictions(
        mutation_file=mutation_file,
        signature_file=signature_file,
        base_model_dir=base_model_dir,
        model_type=model_type,
        output_dir=output_dir,
        config_path=config_path
    )

    # =====================================================
    # Step3
    # =====================================================
    print("\nStep3 -> Exposure + QC")

    step3_out = estimate_exposure_from_predictions(
        mutation_catalog_file=mutation_file,
        signature_matrix_file=signature_file,
        predictions_csv_file=predictions_file,
        output_dir=output_dir,
        model_type=model_type,
        cosine_threshold=cosine_threshold
    )

    initial_exposure = step3_out["exposures"]

    low_cosine_catalog_file = step3_out["low_cosine_catalog_file"]

    # =====================================================
    # No low cosine case
    # =====================================================
    if step3_out["low_cosine_samples"] is None or len(step3_out["low_cosine_samples"]) == 0:

        print("\nNo low-cosine samples -> skip Step4/5")

        final_exposure = initial_exposure.copy()

    else:

        # =================================================
        # Step4
        # =================================================
        print("\nStep4 -> Re-prediction")

        step4_out = refine_low_cosine_predictions(
            low_cosine_catalog_file=low_cosine_catalog_file,
            mutation_signature_file=signature_file,
            base_model_dir=base_model_dir,
            model_type=model_type,
            output_dir=output_dir
        )

        low_cosine_predictions = step4_out["predictions"]

        # =================================================
        # Step5
        # =================================================
        print("\nStep5 -> Refinement")

        refined_exposure = refine_low_cosine_exposure(
            low_cosine_catalog_file=step3_out["low_cosine_catalog_file"],   
            signature_matrix=pd.read_csv(signature_file, index_col=0),
            low_cosine_predictions_file=predictions_file,
            model_type=model_type,
            output_dir=output_dir,
            filtering_threshold=filtering_threshold
        )

        # =================================================
        # Step6
        # =================================================
        print("\nStep6 -> Reintegration")
        refined_file_path = os.path.join(output_dir, f"{model_type}_low_cosine_refined_exposure.csv")
        refined_exposure_df = pd.read_csv(refined_file_path, index_col=0)
        final_exposure = initial_exposure.copy()
        final_exposure[refined_exposure_df.columns] = refined_exposure_df
        
    # =====================================================
    # Save
    # =====================================================
    out_file = os.path.join(output_dir, f"{model_type}_final_exposure.csv")
    final_exposure.to_csv(out_file)

    print("\nPipeline completed")
    print(out_file)


# =========================================================
# MAIN
# =========================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="AMuSA pipeline runner")

    # =========================
    # required args
    # =========================
    parser.add_argument("--type", choices=["SBS", "DBS", "ID"], required=True)
    parser.add_argument("--mutation_file", required=True)
    parser.add_argument("--signature_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_dir", required=True)

    # =========================
    # thresholds
    # =========================
    parser.add_argument("--cosine_threshold", type=float, default=0.9)
    parser.add_argument("--filtering_threshold", type=float, default=0.02)
    parser.add_argument("--weight_min", type=float, default=0.001)
    parser.add_argument("--weight_max", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--max_active_signatures", type=int, default=7)
    parser.add_argument("--refine_thresholds", type=float, default=0.1)

    args = parser.parse_args()

    # =========================
    # run pipeline 
    # =========================
    run_pipeline(
        model_type=args.type,
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        base_model_dir=args.base_model_dir,
        output_dir=args.output_dir,
        cosine_threshold=args.cosine_threshold,
        filtering_threshold=args.filtering_threshold,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        epsilon=args.epsilon,
        max_active_signatures=args.max_active_signatures,
        refine_thresholds=args.refine_thresholds,
        config_path=None
    )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()