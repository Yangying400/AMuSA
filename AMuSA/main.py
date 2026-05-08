#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse


# =========================================================
# STEP2 + STEP3 PIPELINE
# =========================================================
def run_pipeline(
    model_type,
    mutation_file,
    signature_file,
    model_dir,
    output_dir,
    config_path=None
):
    print("\nRunning Step2 + Step3...")

    from AMuSA.prediction import extract_predictions
    from AMuSA.exposure import estimate_exposure_from_predictions

    print(f"\n[Step2] Using model_dir: {model_dir}")

    if not model_dir or not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    print("\n[Step2] Prediction...")

    # Step2: prediction
    predictions_file = extract_predictions(
        mutation_file=mutation_file,
        signature_file=signature_file,
        base_model_dir=model_dir,
        model_type=model_type,
        output_dir=output_dir,
        config_path=None  
    )

    print(f"Prediction saved: {predictions_file}")

    print("\n[Step3] Exposure...")

    # Step3: exposure estimation
    estimate_exposure_from_predictions(
        mutation_catalog_file=mutation_file,
        signature_matrix_file=signature_file,
        predictions_csv_file=predictions_file,
        output_dir=output_dir,
        model_type=model_type
    )

    print("\n[Step2+3 DONE]")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="AMuSA Pipeline (Inference Only)")

    # required inputs
    parser.add_argument("--type", choices=["SBS", "DBS", "ID"], required=True)
    parser.add_argument("--mutation_file", required=True)
    parser.add_argument("--signature_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--cosine_threshold", type=float, default=0.9)
    
    # optional config (for your postprocess pipeline)
    parser.add_argument("--config", default=None)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AMuSA Pipeline (Inference Only)")
    print("=" * 60)

    run_pipeline(
        model_type=args.type,
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        model_dir=args.base_model_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )

    print("\nALL DONE!")


if __name__ == "__main__":
    main()