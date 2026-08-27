#!/usr/bin/env python3
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd
import torch

from AMuSA.data_loader import load_data
from AMuSA.trainer import get_encoded_features, load_model
from AMuSA.utils import post_process_predictions


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
    Predict all signatures used by the trained model first,
    then select and reorder signatures according to signature_file.
    """

    # =============================
    # 1. Resolve model directory
    # =============================
    model_dir = os.path.join(
        base_model_dir,
        f"{model_type}_models"
    )

    if not os.path.exists(model_dir):
        raise ValueError(
            f"Model directory not found: {model_dir}"
        )

    # =============================
    # 2. Create output directory
    # =============================
    pred_output_dir = os.path.join(
        output_dir,
        "final_predictions",
        model_type
    )

    os.makedirs(
        pred_output_dir,
        exist_ok=True
    )

    # =============================
    # 3. Collect model paths
    # =============================
    model_paths = sorted([
        os.path.join(model_dir, file_name)
        for file_name in os.listdir(model_dir)
        if file_name.endswith(".pth")
    ])

    if not model_paths:
        raise ValueError(
            f"No .pth models found in {model_dir}"
        )

    print(
        f"Using {len(model_paths)} models from: "
        f"{model_dir}"
    )

    # =============================
    # 4. Load first model metadata
    # =============================
    first_model = torch.load(
        model_paths[0],
        map_location="cpu",
        weights_only=False
    )

    if "scaler" not in first_model:
        raise KeyError(
            f"No scaler found in model: {model_paths[0]}"
        )

    if "signature_names" not in first_model:
        raise KeyError(
            f"No signature_names found in model: "
            f"{model_paths[0]}"
        )

    scaler = first_model["scaler"]

    # Complete signature names used during model training,
    # for example, 51 SBS signatures.
    model_signature_names = list(
        first_model["signature_names"]
    )

    # =============================
    # 5. Load target signatures
    # =============================
    signature_df = pd.read_csv(
        signature_file,
        index_col=0
    )

    # Signatures contained in the current reference file,
    # for example, 44 SBS signatures.
    signature_names = (
        signature_df.columns.tolist()
    )

    if not signature_names:
        raise ValueError(
            "No signatures found in signature file"
        )

    missing_signatures = [
        signature_name
        for signature_name in signature_names
        if signature_name not in model_signature_names
    ]

    if missing_signatures:
        raise ValueError(
            "The following signatures are missing "
            f"from the model: {missing_signatures}"
        )

    # Positions of the selected 44 signatures
    # in the complete 51-signature model output.
    selected_indices = [
        model_signature_names.index(signature_name)
        for signature_name in signature_names
    ]

    print(
        f"Model signatures: "
        f"{len(model_signature_names)}"
    )

    print(
        f"Selected signatures: "
        f"{len(signature_names)}"
    )

    # =============================
    # 6. Load mutation data
    # =============================
    X_test, _, _, sample_ids, _ = load_data(
        mutation_file=mutation_file,
        exposure_file=None,
        signature_file=signature_file,
        scaler=scaler,
        train=False
    )

    sample_ids = list(sample_ids)

    # =============================
    # 7. Ensemble prediction
    # =============================
    all_probs = []
    all_thresholds = []

    for path in model_paths:
        model_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        if "signature_names" not in model_dict:
            raise KeyError(
                f"No signature_names found in model: "
                f"{path}"
            )

        current_signature_names = list(
            model_dict["signature_names"]
        )

        # Check whether the current model contains
        # all signatures from the first model.
        missing_in_current_model = [
            signature_name
            for signature_name in model_signature_names
            if signature_name not in current_signature_names
        ]

        if missing_in_current_model:
            raise ValueError(
                f"Model {path} is missing signatures: "
                f"{missing_in_current_model}"
            )

        # Align the current model output to the
        # signature order of the first model.
        alignment_indices = [
            current_signature_names.index(
                signature_name
            )
            for signature_name in model_signature_names
        ]

        classifier, autoencoder, _, _ = load_model(
            path
        )

        X_test_encoded = get_encoded_features(
            autoencoder,
            X_test
        )

        classifier.eval()

        with torch.no_grad():
            probs, _, _ = classifier(
                torch.FloatTensor(
                    X_test_encoded
                ).to(
                    classifier.thresholds.device
                )
            )

        probs = (
            probs
            .detach()
            .cpu()
            .numpy()
        )

        if probs.ndim != 2:
            raise ValueError(
                f"Unexpected probability shape "
                f"in {path}: {probs.shape}"
            )

        if probs.shape[1] != len(
            current_signature_names
        ):
            raise ValueError(
                f"Model output dimension mismatch "
                f"in {path}: "
                f"output={probs.shape[1]}, "
                f"signatures="
                f"{len(current_signature_names)}"
            )

        # Keep the complete 51-signature prediction
        # and align its order.
        probs = probs[:, alignment_indices]

        all_probs.append(probs)

        # =============================
        # Collect model thresholds
        # =============================
        if "thresholds" in model_dict:
            thresholds = model_dict["thresholds"]

            if isinstance(
                thresholds,
                torch.Tensor
            ):
                thresholds = (
                    thresholds
                    .detach()
                    .cpu()
                    .numpy()
                )

            thresholds = np.asarray(
                thresholds,
                dtype=float
            ).reshape(-1)

            if len(thresholds) != len(
                current_signature_names
            ):
                raise ValueError(
                    f"Threshold length mismatch "
                    f"in {path}: "
                    f"thresholds={len(thresholds)}, "
                    f"signatures="
                    f"{len(current_signature_names)}"
                )

            thresholds = thresholds[
                alignment_indices
            ]

            all_thresholds.append(
                thresholds
            )

    # =============================
    # 8. Average full model outputs
    # =============================
    avg_full_probs = np.mean(
        np.stack(
            all_probs,
            axis=0
        ),
        axis=0
    )

    if all_thresholds:
        avg_full_thresholds = np.mean(
            np.stack(
                all_thresholds,
                axis=0
            ),
            axis=0
        )
    else:
        print(
            "No model thresholds found. "
            "Using default threshold 0.5."
        )

        avg_full_thresholds = np.full(
            len(model_signature_names),
            0.5,
            dtype=float
        )

    # =============================
    # 9. Select 44 signatures
    # =============================
    avg_probs = avg_full_probs[
        :,
        selected_indices
    ]

    avg_thresholds = avg_full_thresholds[
        selected_indices
    ]

    if avg_probs.shape[1] != len(
        signature_names
    ):
        raise ValueError(
            "Selected probability columns do not "
            "match the signature names"
        )

    if len(avg_thresholds) != len(
        signature_names
    ):
        raise ValueError(
            "Selected thresholds do not match "
            "the signature names"
        )

    print(
        "Selected probability matrix shape:",
        avg_probs.shape
    )

    print(
        "Selected threshold number:",
        len(avg_thresholds)
    )

    # =============================
    # 10. Binarization
    # =============================
    active_preds = (
        avg_probs
        >= avg_thresholds.reshape(1, -1)
    ).astype(float)

    # =============================
    # 11. Post-processing
    # =============================
    if max_active_signatures is not None:
        active_preds = post_process_predictions(
            avg_probs,
            avg_thresholds,
            max_active_signatures
        )

    active_preds_binary = (
        active_preds.astype(int)
    )

    # =============================
    # 12. Save 44-signature results
    # =============================
    predictions_csv_file = os.path.join(
        pred_output_dir,
        "predictions.csv"
    )

    predictions_df = pd.DataFrame(
        active_preds_binary.T,
        index=signature_names,
        columns=sample_ids
    )

    predictions_df.to_csv(
        predictions_csv_file
    )

    probabilities_csv_file = os.path.join(
        pred_output_dir,
        "probabilities.csv"
    )

    probabilities_df = pd.DataFrame(
        avg_probs.T,
        index=signature_names,
        columns=sample_ids
    )

    probabilities_df.to_csv(
        probabilities_csv_file
    )

    thresholds_csv_file = os.path.join(
        pred_output_dir,
        "thresholds.csv"
    )

    thresholds_df = pd.DataFrame({
        "signature": signature_names,
        "threshold": avg_thresholds
    })

    thresholds_df.to_csv(
        thresholds_csv_file,
        index=False
    )

    print(
        f"Results saved to: "
        f"{pred_output_dir}"
    )

    print(
        f"Predictions saved: "
        f"{predictions_csv_file}"
    )

    return predictions_csv_file


# =============================
# CLI entry point
# =============================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict active mutational signatures"
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
        "--base_model_dir",
        required=True
    )

    parser.add_argument(
        "--model_type",
        choices=["SBS", "DBS", "ID"],
        default="SBS"
    )

    parser.add_argument(
        "--output_dir",
        default="output"
    )

    parser.add_argument(
        "--max_active_signatures",
        type=int,
        default=7,
        help=(
            "Maximum number of active signatures "
            "per sample"
        )
    )

    args = parser.parse_args()

    extract_predictions(
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        base_model_dir=args.base_model_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
        max_active_signatures=(
            args.max_active_signatures
        )
    )


if __name__ == "__main__":
    main()