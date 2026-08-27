#!/usr/bin/env python3
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

from AMuSA.data_loader import load_data
from AMuSA.trainer import get_encoded_features, load_model


def cosine_similarity(x, y):
    """Calculate cosine similarity between two one-dimensional arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return 0.0

    return float(np.dot(x, y) / denominator)


def fit_nnls(signature_matrix, sample_counts, selected):
    """Fit ordinary NNLS using only the selected signature columns."""
    selected = list(selected)

    if not selected:
        reconstruction = np.zeros_like(sample_counts, dtype=float)
        return np.array([], dtype=float), reconstruction, 0.0

    profiles = signature_matrix[:, selected]
    coefficients, _ = nnls(profiles, sample_counts)
    reconstruction = profiles @ coefficients
    cosine = cosine_similarity(sample_counts, reconstruction)

    return coefficients, reconstruction, cosine


def contribution_fractions(coefficients):
    """Convert NNLS coefficients to fractions of the total exposure."""
    coefficients = np.asarray(coefficients, dtype=float)
    total = coefficients.sum()

    if total <= 0:
        return np.zeros_like(coefficients, dtype=float)

    return coefficients / total


def fit_and_filter(
    signature_matrix,
    sample_counts,
    selected,
    min_contribution,
    max_active_signatures=None,
    protected=None,
):
    """
    Repeatedly fit NNLS while protecting the original signatures.

    Parameters
    ----------
    protected
        Signature indices originating from the original Step 2 result.
        These signatures never participate in contribution-based deletion
        and are not removed to satisfy ``max_active_signatures``.

    Only signatures added during refinement can be removed when:
    1. Their contribution fraction is below ``min_contribution``; or
    2. The selected-signature count exceeds ``max_active_signatures``.

    NNLS is re-run after every removal and once more at the end.
    """
    selected = list(dict.fromkeys(selected))
    protected_set = set(protected or []).intersection(selected)
    deleted = []

    while selected:
        coefficients, _, _ = fit_nnls(
            signature_matrix,
            sample_counts,
            selected,
        )
        fractions = contribution_fractions(coefficients)

        # Only signatures added during refinement can be removed.
        removable_positions = [
            position
            for position, signature_index in enumerate(selected)
            if signature_index not in protected_set
        ]

        # If only original signatures remain, stop filtering.
        if not removable_positions:
            break

        # Among later-added signatures, inspect the lowest contributor.
        lowest_removable_position = min(
            removable_positions,
            key=lambda position: float(fractions[position]),
        )

        exceeds_limit = (
            max_active_signatures is not None
            and len(selected) > max_active_signatures
        )
        below_threshold = (
            float(fractions[lowest_removable_position])
            < min_contribution
        )

        if not exceeds_limit and not below_threshold:
            break

        deleted.append(selected.pop(lowest_removable_position))

    coefficients, reconstruction, cosine = fit_nnls(
        signature_matrix,
        sample_counts,
        selected,
    )
    fractions = contribution_fractions(coefficients)

    return selected, coefficients, fractions, reconstruction, cosine, deleted


def refine_sample(
    sample_name,
    sample_counts,
    signature_matrix,
    signature_names,
    probabilities,
    original_exposures,
    probability_threshold,
    min_contribution,
    target_cosine,
    min_improvement,
    max_active_signatures,
):
    """
    Run forward candidate addition for one sample.

    Original Step 2 signatures are protected and never participate in
    contribution-based deletion. Only signatures added during refinement
    can be removed.
    """
    sample_counts = np.asarray(sample_counts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    original_exposures = np.asarray(original_exposures, dtype=float)

    if sample_counts.sum() <= 0:
        return {
            "original_cosine": 0.0,
            "attempted_final_cosine": 0.0,
            "final_cosine": 0.0,
            "improvement": 0.0,
            "use_refined": False,
            "selected": [],
            "coefficients": np.array([], dtype=float),
            "fractions": np.array([], dtype=float),
            "reconstruction": np.zeros_like(sample_counts, dtype=float),
            "log": [],
        }

    # -----------------------------------------------------
    # Original Step 2 NNLS exposure
    # -----------------------------------------------------
    # Only signatures with a positive floating-point exposure in Step 2
    # are treated as signatures that were actually used initially.
    exposure_epsilon = 1e-12
    original_selected = np.where(
        original_exposures > exposure_epsilon
    )[0].tolist()

    # Preserve the original Step 2 NNLS coefficients instead of fitting them
    # again from the Step 1 binary prediction matrix.
    original_coefficients = original_exposures[original_selected].copy()

    if original_selected:
        original_reconstruction = (
            signature_matrix[:, original_selected] @ original_coefficients
        )
        original_cosine = cosine_similarity(
            sample_counts,
            original_reconstruction,
        )
    else:
        original_reconstruction = np.zeros_like(
            sample_counts,
            dtype=float,
        )
        original_cosine = 0.0

    selected = original_selected.copy()
    coefficients = original_coefficients.copy()
    reconstruction = original_reconstruction.copy()
    current_cosine = original_cosine

    # Only signatures actually used by Step 2 are excluded.
    # Every other signature with probability strictly greater than the
    # threshold enters the candidate pool, including signatures predicted
    # in Step 1 but assigned zero exposure by Step 2 NNLS.
    candidates = [
        index
        for index, probability in enumerate(probabilities)
        if probability > probability_threshold and index not in selected
    ]

    blocked = set()
    log_rows = []
    step = 0

    # -----------------------------------------------------
    # Forward addition; backward deletion applies only to added signatures
    # -----------------------------------------------------
    while current_cosine < target_cosine:
        available = [
            index
            for index in candidates
            if index not in selected and index not in blocked
        ]

        if not available:
            break

        best_candidate = None
        best_result = None
        failed_candidates = []
        round_rows = []

        for candidate in available:
            trial_selected = selected + [candidate]

            trial_coefficients, _, _ = fit_nnls(
                signature_matrix,
                sample_counts,
                trial_selected,
            )
            trial_fractions = contribution_fractions(trial_coefficients)
            contribution_before = float(trial_fractions[-1])

            result = fit_and_filter(
                signature_matrix=signature_matrix,
                sample_counts=sample_counts,
                selected=trial_selected,
                min_contribution=min_contribution,
                max_active_signatures=max_active_signatures,
                protected=original_selected,
            )

            (
                filtered_selected,
                _,
                filtered_fractions,
                _,
                tested_cosine,
                deleted,
            ) = result

            survived = candidate in filtered_selected
            contribution_after = (
                float(filtered_fractions[filtered_selected.index(candidate)])
                if survived
                else 0.0
            )

            if not survived:
                failed_candidates.append(candidate)

            round_rows.append({
                "sample": sample_name,
                "step": step + 1,
                "candidate": signature_names[candidate],
                "probability": float(probabilities[candidate]),
                "contribution_before": contribution_before,
                "contribution_after": contribution_after,
                "cosine_before": current_cosine,
                "cosine_after": tested_cosine,
                "deleted_signatures": ";".join(
                    signature_names[index] for index in deleted
                ),
                "action": "rejected" if not survived else "tested",
            })

            if survived and (
                best_result is None
                or tested_cosine > best_result[4]
            ):
                best_candidate = candidate
                best_result = result

        # Candidates removed by NNLS filtering are not tested again.
        blocked.update(failed_candidates)

        if best_result is None:
            log_rows.extend(round_rows)
            break

        best_improvement = best_result[4] - current_cosine
        if best_improvement < min_improvement:
            log_rows.extend(round_rows)
            break

        step += 1
        previous_selected = set(selected)
        (
            selected,
            coefficients,
            _,
            reconstruction,
            current_cosine,
            _,
        ) = best_result

        # A signature deleted during an accepted step cannot return later.
        blocked.update(previous_selected - set(selected))

        for row in round_rows:
            if row["candidate"] == signature_names[best_candidate]:
                row["action"] = "accepted"
            log_rows.append(row)

        print(
            f"{sample_name} | step {step} | "
            f"add {signature_names[best_candidate]} | "
            f"cosine {current_cosine:.6f}"
        )

    # -----------------------------------------------------
    # Final filtering of added signatures and final NNLS refit
    # -----------------------------------------------------
    (
        filtered_selected,
        filtered_coefficients,
        filtered_fractions,
        filtered_reconstruction,
        filtered_cosine,
        final_deleted,
    ) = fit_and_filter(
        signature_matrix=signature_matrix,
        sample_counts=sample_counts,
        selected=selected,
        min_contribution=min_contribution,
        max_active_signatures=max_active_signatures,
        protected=original_selected,
    )

    for deleted_index in final_deleted:
        log_rows.append({
            "sample": sample_name,
            "step": step + 1,
            "candidate": signature_names[deleted_index],
            "probability": float(probabilities[deleted_index]),
            "contribution_before": np.nan,
            "contribution_after": 0.0,
            "cosine_before": current_cosine,
            "cosine_after": filtered_cosine,
            "deleted_signatures": signature_names[deleted_index],
            "action": "final_filter_removed",
        })

    attempted_improvement = filtered_cosine - original_cosine
    use_refined = attempted_improvement >= min_improvement

    # Preserve the original solution when refinement does not improve cosine.
    if use_refined:
        final_selected = filtered_selected
        final_coefficients = filtered_coefficients
        final_fractions = filtered_fractions
        final_reconstruction = filtered_reconstruction
        final_cosine = filtered_cosine
    else:
        final_selected = original_selected
        final_coefficients = original_coefficients
        final_fractions = contribution_fractions(original_coefficients)
        final_reconstruction = original_reconstruction
        final_cosine = original_cosine

    return {
        "original_cosine": original_cosine,
        "attempted_final_cosine": filtered_cosine,
        "final_cosine": final_cosine,
        "improvement": attempted_improvement,
        "use_refined": use_refined,
        "selected": final_selected,
        "coefficients": final_coefficients,
        "fractions": final_fractions,
        "reconstruction": final_reconstruction,
        "log": log_rows,
    }


def predict_probabilities(
    mutation_file,
    signature_file,
    base_model_dir,
    model_type,
    signature_names,
):
    """Predict mean signature probabilities from the model ensemble."""
    model_dir = os.path.join(base_model_dir, f"{model_type}_models")

    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    model_paths = sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(".pth")
    )

    if not model_paths:
        raise ValueError(f"No .pth model found in {model_dir}")

    all_probabilities = []
    sample_ids = None

    for model_path in model_paths:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )
        if "scaler" not in checkpoint:
            raise ValueError(f"Scaler not found in model: {model_path}")

        X_test, _, _, model_sample_ids, _ = load_data(
            mutation_file=mutation_file,
            exposure_file=None,
            signature_file=signature_file,
            scaler=checkpoint["scaler"],
            train=False,
        )
        model_sample_ids = [str(sample_id) for sample_id in model_sample_ids]

        if sample_ids is None:
            sample_ids = model_sample_ids
        elif model_sample_ids != sample_ids:
            raise ValueError(
                f"Sample order from {model_path} does not match the ensemble."
            )

        model_names = [str(name) for name in checkpoint["signature_names"]]

        missing_signatures = [
            name for name in signature_names if name not in model_names
        ]
        if missing_signatures:
            raise ValueError(
                f"Model {model_path} is missing signatures: "
                f"{missing_signatures}"
            )

        signature_order = [model_names.index(name) for name in signature_names]

        classifier, autoencoder, _, _ = load_model(model_path)
        encoded = get_encoded_features(autoencoder, X_test)
        classifier.eval()
        device = next(classifier.parameters()).device

        with torch.no_grad():
            probabilities, _, _ = classifier(
                torch.as_tensor(
                    encoded,
                    dtype=torch.float32,
                    device=device,
                )
            )

        model_probabilities = probabilities.detach().cpu().numpy()
        all_probabilities.append(model_probabilities[:, signature_order])

    mean_probabilities = np.mean(all_probabilities, axis=0)
    return mean_probabilities, sample_ids


def refine_low_cosine_predictions(
    low_cosine_catalog_file,
    mutation_signature_file,
    original_exposures_float,
    base_model_dir,
    model_type,
    output_dir,
    probability_threshold=0.05,
    min_contribution=0.05,
    target_cosine=1.0,
    min_improvement=1e-2,
    max_active_signatures=7,
):
    """Refine low-cosine samples using the Step 2 floating NNLS exposure."""
    catalog = pd.read_csv(low_cosine_catalog_file, index_col=0)
    signatures = pd.read_csv(mutation_signature_file, index_col=0)

    # Pipeline calls may pass the Step 2 DataFrame directly. Standalone use
    # may pass the path of the Step 2 floating-point exposure CSV.
    if isinstance(original_exposures_float, pd.DataFrame):
        original_exposures_float = original_exposures_float.copy()
    elif isinstance(original_exposures_float, (str, os.PathLike)):
        original_exposures_float = pd.read_csv(
            original_exposures_float,
            index_col=0,
        )
    else:
        raise TypeError(
            "original_exposures_float must be a pandas DataFrame "
            "or a CSV file path."
        )

    catalog.index = catalog.index.astype(str)
    catalog.columns = catalog.columns.astype(str)
    signatures.index = signatures.index.astype(str)
    signatures.columns = signatures.columns.astype(str)
    original_exposures_float.index = (
        original_exposures_float.index.astype(str)
    )
    original_exposures_float.columns = (
        original_exposures_float.columns.astype(str)
    )

    missing_features = signatures.index.difference(catalog.index)
    if len(missing_features) > 0:
        raise ValueError(
            f"Low-cosine catalog is missing mutation types: "
            f"{missing_features.tolist()}"
        )

    catalog = catalog.reindex(signatures.index)
    signature_names = signatures.columns.tolist()
    signature_matrix = signatures.values.astype(float)

    probabilities, sample_ids = predict_probabilities(
        mutation_file=low_cosine_catalog_file,
        signature_file=mutation_signature_file,
        base_model_dir=base_model_dir,
        model_type=model_type,
        signature_names=signature_names,
    )

    missing_samples = [
        sample_name for sample_name in sample_ids
        if sample_name not in catalog.columns
    ]
    if missing_samples:
        raise ValueError(
            f"Model output contains samples absent from the low-cosine catalog: "
            f"{missing_samples}"
        )

    catalog = catalog.reindex(columns=sample_ids)
    original_exposures_float = original_exposures_float.reindex(
        index=signature_names,
        columns=sample_ids,
        fill_value=0,
    ).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if (original_exposures_float < 0).any().any():
        raise ValueError(
            "Step 2 exposure contains negative values, which is invalid "
            "for NNLS exposure."
        )

    shape = (len(signature_names), len(sample_ids))
    final_predictions = np.zeros(shape, dtype=int)
    final_exposures_float = np.zeros(shape, dtype=float)
    final_contributions = np.zeros(shape, dtype=float)
    summary_rows = []
    log_rows = []

    for sample_position, sample_name in enumerate(sample_ids):
        result = refine_sample(
            sample_name=sample_name,
            sample_counts=catalog[sample_name].values.astype(float),
            signature_matrix=signature_matrix,
            signature_names=signature_names,
            probabilities=probabilities[sample_position],
            original_exposures=original_exposures_float[
                sample_name
            ].values.astype(float),
            probability_threshold=probability_threshold,
            min_contribution=min_contribution,
            target_cosine=target_cosine,
            min_improvement=min_improvement,
            max_active_signatures=max_active_signatures,
        )

        selected = result["selected"]
        final_predictions[selected, sample_position] = 1
        final_exposures_float[selected, sample_position] = result["coefficients"]
        final_contributions[selected, sample_position] = result["fractions"]

        summary_rows.append({
            "sample": sample_name,
            "original_cosine": result["original_cosine"],
            "attempted_final_cosine": result["attempted_final_cosine"],
            "final_cosine": result["final_cosine"],
            "improvement": result["improvement"],
            "use_refined": result["use_refined"],
            "final_signature_count": len(selected),
            "final_signatures": ";".join(
                signature_names[index] for index in selected
            ),
        })
        log_rows.extend(result["log"])

    predictions_df = pd.DataFrame(
        final_predictions,
        index=signature_names,
        columns=sample_ids,
    )
    exposures_float_df = pd.DataFrame(
        final_exposures_float,
        index=signature_names,
        columns=sample_ids,
    )
    exposures_df = exposures_float_df.round(0).astype(int)
    contributions_df = pd.DataFrame(
        final_contributions,
        index=signature_names,
        columns=sample_ids,
    )
    summary_df = pd.DataFrame(summary_rows)
    log_df = pd.DataFrame(log_rows)

    improved_samples = summary_df.loc[
        summary_df["use_refined"],
        "sample",
    ].tolist()

    output_path = os.path.join(
        output_dir,
        "results",
        model_type,
        "low_cosine_refinement",
    )
    os.makedirs(output_path, exist_ok=True)

    predictions_file = os.path.join(output_path, "final_predictions.csv")
    exposures_float_file = os.path.join(output_path, "final_exposures_float.csv")
    exposures_file = os.path.join(output_path, "final_exposures.csv")
    contributions_file = os.path.join(output_path, "final_contributions.csv")
    summary_file = os.path.join(output_path, "cosine_summary.csv")
    log_file = os.path.join(output_path, "refinement_log.csv")

    predictions_df.to_csv(predictions_file)
    exposures_float_df.to_csv(exposures_float_file)
    exposures_df.to_csv(exposures_file)
    contributions_df.to_csv(contributions_file)
    summary_df.to_csv(summary_file, index=False)
    log_df.to_csv(log_file, index=False)

    print(f"Refinement results saved to: {output_path}")
    print(
        f"Improved samples: {len(improved_samples)} / {len(sample_ids)}"
    )

    return {
        "predictions": predictions_df,
        "exposures": exposures_df,
        "exposures_float": exposures_float_df,
        "contributions": contributions_df,
        "summary": summary_df,
        "log": log_df,
        "improved_samples": improved_samples,
        "predictions_file": predictions_file,
        "exposures_file": exposures_file,
        "exposures_float_file": exposures_float_file,
        "contributions_file": contributions_file,
        "summary_file": summary_file,
        "log_file": log_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Refine low-cosine AMuSA predictions"
    )
    parser.add_argument("--low_cosine_catalog_file", required=True)
    parser.add_argument("--mutation_signature_file", required=True)
    parser.add_argument(
        "--original_exposures_float",
        required=True,
        help=(
            "Step 2 floating-point NNLS exposure CSV. "
            "Only signatures with exposure > 0 are excluded "
            "from the candidate pool."
        ),
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument(
        "--model_type",
        choices=["SBS", "DBS", "ID"],
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--probability_threshold", type=float, default=0.05)
    parser.add_argument("--min_contribution", type=float, default=0.05)
    parser.add_argument("--target_cosine", type=float, default=1.0)
    parser.add_argument("--min_improvement", type=float, default=1e-2)
    parser.add_argument("--max_active_signatures", type=int, default=7)

    refine_low_cosine_predictions(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()