#!/usr/bin/env python3
# coding: utf-8

"""
Search AMuSA refinement thresholds on an independent validation/test dataset.

What this script does
---------------------
1. Loads the mutation catalog, signature matrix, and ground-truth exposures.
2. Obtains ensemble signature probabilities from AMuSA models.
3. Loads the original prediction matrix if supplied; otherwise constructs it
   from ensemble probabilities and checkpoint thresholds.
4. Re-fits the original selected signatures with NNLS and calculates cosine.
5. Only samples with original cosine < 0.95 enter refinement.
6. Grid-searches:
      probability_threshold
      min_contribution
      min_improvement
7. Evaluates every parameter combination using:
      Precision, Recall, F1
      sample-macro F1
      TAE and normalized TAE
      mean reconstruction cosine
      rare-signature recall
8. Selects the best parameters primarily by low-sample F1 and saves all results.

Recommended location
--------------------
Save this file as:
    /home/yangying/AMuSA_SBS_final/threshold_grid_search.py

Run from the project root:
    cd /home/yangying/AMuSA_SBS_final
    python threshold_grid_search.py
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

from AMuSA.data_loader import load_data
from AMuSA.trainer import get_encoded_features, load_model


# =========================================================
# Default paths
# =========================================================
DEFAULT_MUTATION_FILE = (
    "/home/yangying/AMuSA_SBS_final/data/test_sbs_catalog.csv"
)
DEFAULT_SIGNATURE_FILE = (
    "/home/yangying/AMuSA_SBS_final/data/ground.truth.syn.sigs.SBS96.csv"
)
DEFAULT_GROUND_TRUTH_EXPOSURES = (
    "/home/yangying/AMuSA_SBS_final/data/test_sbs_exposures.csv"
)
DEFAULT_BASE_MODEL_DIR = (
    "/home/yangying/AMuSA_SBS_final/AMuSA/models"
)
DEFAULT_OUTPUT_DIR = (
    "/home/yangying/AMuSA_SBS_final/output/SBS_threshold_search"
)


# =========================================================
# Numerical threshold
# =========================================================
EXPOSURE_EPSILON = 1e-8


# =========================================================
# Basic calculations
# =========================================================
def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate cosine similarity between two one-dimensional arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return 0.0

    return float(np.dot(x, y) / denominator)


def fit_nnls(
    signature_matrix: np.ndarray,
    sample_counts: np.ndarray,
    selected: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit ordinary NNLS using only selected signature columns."""
    selected = list(selected)
    sample_counts = np.asarray(sample_counts, dtype=float)

    if not selected:
        reconstruction = np.zeros_like(sample_counts, dtype=float)
        return np.array([], dtype=float), reconstruction, 0.0

    profiles = signature_matrix[:, selected]
    coefficients, _ = nnls(profiles, sample_counts)
    reconstruction = profiles @ coefficients
    cosine = cosine_similarity(sample_counts, reconstruction)

    return coefficients, reconstruction, cosine


def contribution_fractions(coefficients: np.ndarray) -> np.ndarray:
    """Convert NNLS coefficients to fractions of total exposure."""
    coefficients = np.asarray(coefficients, dtype=float)
    total = float(coefficients.sum())

    if total <= 0:
        return np.zeros_like(coefficients, dtype=float)

    return coefficients / total


def fit_and_filter(
    signature_matrix: np.ndarray,
    sample_counts: np.ndarray,
    selected: Sequence[int],
    min_contribution: float,
    max_active_signatures: Optional[int],
    protected: Optional[Sequence[int]] = None,
) -> Tuple[
    List[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    List[int],
]:
    """
    Repeatedly fit NNLS, but filter only newly added signatures.

    Signatures in ``protected`` are the signatures selected by the original
    prediction step. They are never removed by either the contribution
    threshold or ``max_active_signatures``.

    Only an unprotected (later-added) signature can be removed when:
    1. Its relative contribution is below ``min_contribution``; or
    2. The total number of selected signatures exceeds
       ``max_active_signatures``.

    If the protected signatures alone already reach or exceed the maximum,
    every newly added candidate will be removed instead of deleting an
    original signature.
    """
    selected = list(dict.fromkeys(selected))
    protected_set = set(protected or []).intersection(selected)
    deleted: List[int] = []

    while selected:
        coefficients, _, _ = fit_nnls(
            signature_matrix,
            sample_counts,
            selected,
        )
        fractions = contribution_fractions(coefficients)

        removable_positions = [
            position
            for position, signature_index in enumerate(selected)
            if signature_index not in protected_set
        ]

        # There are no later-added signatures left to filter.
        if not removable_positions:
            break

        lowest_removable_position = min(
            removable_positions,
            key=lambda position: float(fractions[position]),
        )

        active_count = int(
            np.sum(coefficients > EXPOSURE_EPSILON)
        )

        exceeds_limit = (
            max_active_signatures is not None
            and active_count > max_active_signatures
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

    return (
        selected,
        coefficients,
        fractions,
        reconstruction,
        cosine,
        deleted,
    )


# =========================================================
# Single-sample refinement
# =========================================================
def refine_sample(
    sample_counts: np.ndarray,
    signature_matrix: np.ndarray,
    probabilities: np.ndarray,
    original_predictions: np.ndarray,
    probability_threshold: float,
    min_contribution: float,
    target_cosine: float,
    min_improvement: float,
    max_active_signatures: int,
) -> Dict[str, object]:
    """
    Refine one sample using forward candidate addition and protected filtering.

    The signatures selected by the original prediction are fixed members of
    the model. Contribution filtering and the active-signature limit apply
    only to signatures added during refinement.

    Each round:
    1. Tests every available candidate separately.
    2. Runs NNLS and filters only later-added signatures.
    3. Selects the surviving candidate with the highest cosine.
    4. Accepts it only when cosine improvement >= min_improvement.
    """
    sample_counts = np.asarray(sample_counts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    original_predictions = np.asarray(original_predictions, dtype=float)

    original_selected = np.where(original_predictions > 0)[0].tolist()
    (
        original_coefficients,
        original_reconstruction,
        original_cosine,
    ) = fit_nnls(
        signature_matrix,
        sample_counts,
        original_selected,
    )

    if sample_counts.sum() <= 0:
        return {
            "original_cosine": 0.0,
            "attempted_final_cosine": 0.0,
            "final_cosine": 0.0,
            "improvement": 0.0,
            "use_refined": False,
            "selected": original_selected,
            "coefficients": original_coefficients,
            "reconstruction": original_reconstruction,
        }

    selected = original_selected.copy()
    coefficients = original_coefficients.copy()
    reconstruction = original_reconstruction.copy()
    current_cosine = original_cosine

    candidates = [
        index
        for index, probability in enumerate(probabilities)
        if probability >= probability_threshold
        and index not in selected
    ]

    blocked = set()

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
        failed_candidates: List[int] = []

        for candidate in available:
            trial_selected = selected + [candidate]

            result = fit_and_filter(
                signature_matrix=signature_matrix,
                sample_counts=sample_counts,
                selected=trial_selected,
                min_contribution=min_contribution,
                max_active_signatures=max_active_signatures,
                protected=original_selected,
            )

            filtered_selected = result[0]
            tested_cosine = result[4]
            survived = candidate in filtered_selected

            if not survived:
                failed_candidates.append(candidate)
                continue

            if best_result is None or tested_cosine > best_result[4]:
                best_candidate = candidate
                best_result = result

        # A candidate removed by filtering is not tested again.
        blocked.update(failed_candidates)

        if best_result is None or best_candidate is None:
            break

        best_improvement = float(best_result[4] - current_cosine)

        if best_improvement < min_improvement:
            break

        previous_selected = set(selected)

        (
            selected,
            coefficients,
            _,
            reconstruction,
            current_cosine,
            _,
        ) = best_result

        # Only previously added candidates can be deleted here.
        # Original signatures are protected and can never enter this set.
        blocked.update(previous_selected - set(selected))

    # Mandatory final filtering.
    (
        filtered_selected,
        filtered_coefficients,
        _,
        filtered_reconstruction,
        filtered_cosine,
        _,
    ) = fit_and_filter(
        signature_matrix=signature_matrix,
        sample_counts=sample_counts,
        selected=selected,
        min_contribution=min_contribution,
        max_active_signatures=max_active_signatures,
        protected=original_selected,
    )

    attempted_improvement = float(filtered_cosine - original_cosine)
    use_refined = attempted_improvement >= min_improvement

    if use_refined:
        final_selected = filtered_selected
        final_coefficients = filtered_coefficients
        final_reconstruction = filtered_reconstruction
        final_cosine = filtered_cosine
    else:
        final_selected = original_selected
        final_coefficients = original_coefficients
        final_reconstruction = original_reconstruction
        final_cosine = original_cosine

    return {
        "original_cosine": float(original_cosine),
        "attempted_final_cosine": float(filtered_cosine),
        "final_cosine": float(final_cosine),
        "improvement": attempted_improvement,
        "use_refined": bool(use_refined),
        "selected": final_selected,
        "coefficients": final_coefficients,
        "reconstruction": final_reconstruction,
    }


# =========================================================
# Matrix alignment
# =========================================================
def read_csv_matrix(path: str) -> pd.DataFrame:
    """Read a CSV matrix and convert row/column labels to strings."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)

    return frame


def align_signature_sample_matrix(
    frame: pd.DataFrame,
    signature_names: Sequence[str],
    sample_ids: Sequence[str],
    matrix_name: str,
) -> pd.DataFrame:
    """
    Align a signature-by-sample matrix.

    If the supplied matrix is sample-by-signature, it is transposed
    automatically.
    """
    signature_names = [str(x) for x in signature_names]
    sample_ids = [str(x) for x in sample_ids]

    index_has_signatures = set(signature_names).issubset(set(frame.index))
    columns_have_samples = set(sample_ids).issubset(set(frame.columns))

    if index_has_signatures and columns_have_samples:
        aligned = frame.reindex(
            index=signature_names,
            columns=sample_ids,
            fill_value=0,
        )
        return aligned.apply(pd.to_numeric, errors="coerce").fillna(0)

    columns_have_signatures = set(signature_names).issubset(
        set(frame.columns)
    )
    index_has_samples = set(sample_ids).issubset(set(frame.index))

    if columns_have_signatures and index_has_samples:
        aligned = frame.T.reindex(
            index=signature_names,
            columns=sample_ids,
            fill_value=0,
        )
        return aligned.apply(pd.to_numeric, errors="coerce").fillna(0)

    raise ValueError(
        f"{matrix_name} cannot be aligned as signature x sample.\n"
        f"Expected signatures such as: {signature_names[:5]}\n"
        f"Expected samples such as: {sample_ids[:5]}"
    )


# =========================================================
# Model ensemble prediction
# =========================================================
def predict_ensemble_probabilities_and_thresholds(
    mutation_file: str,
    signature_file: str,
    base_model_dir: str,
    model_type: str,
    signature_names: Sequence[str],
    fallback_prediction_threshold: float,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Predict mean signature probabilities from all .pth models.

    The original binary predictions can be generated from:
    - Mean checkpoint thresholds, if all models contain thresholds; or
    - fallback_prediction_threshold, otherwise.
    """
    model_dir = os.path.join(base_model_dir, f"{model_type}_models")

    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    model_paths = sorted(
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(".pth")
    )

    if not model_paths:
        raise ValueError(f"No .pth model found in: {model_dir}")

    all_probabilities: List[np.ndarray] = []
    all_thresholds: List[np.ndarray] = []
    sample_ids: Optional[List[str]] = None

    for model_path in model_paths:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        if "scaler" not in checkpoint:
            raise ValueError(f"Scaler not found in model: {model_path}")
        if "signature_names" not in checkpoint:
            raise ValueError(
                f"signature_names not found in model: {model_path}"
            )

        X_test, _, _, model_sample_ids, _ = load_data(
            mutation_file=mutation_file,
            exposure_file=None,
            signature_file=signature_file,
            scaler=checkpoint["scaler"],
            train=False,
        )

        model_sample_ids = [str(x) for x in model_sample_ids]

        if sample_ids is None:
            sample_ids = model_sample_ids
        elif model_sample_ids != sample_ids:
            raise ValueError(
                f"Sample order from {model_path} does not match "
                "the other ensemble models."
            )

        model_names = [
            str(name) for name in checkpoint["signature_names"]
        ]

        missing_signatures = [
            name for name in signature_names
            if name not in model_names
        ]
        if missing_signatures:
            raise ValueError(
                f"Model {model_path} is missing signatures: "
                f"{missing_signatures}"
            )

        signature_order = [
            model_names.index(name) for name in signature_names
        ]

        classifier, autoencoder, _, _ = load_model(model_path)
        encoded = get_encoded_features(autoencoder, X_test)
        classifier.eval()
        device = next(classifier.parameters()).device

        with torch.no_grad():
            output = classifier(
                torch.as_tensor(
                    encoded,
                    dtype=torch.float32,
                    device=device,
                )
            )

        probabilities = output[0] if isinstance(output, tuple) else output
        model_probabilities = probabilities.detach().cpu().numpy()
        all_probabilities.append(
            model_probabilities[:, signature_order]
        )

        if "thresholds" in checkpoint:
            thresholds = np.asarray(
                checkpoint["thresholds"],
                dtype=float,
            ).reshape(-1)

            if len(thresholds) != len(model_names):
                raise ValueError(
                    f"Threshold length does not match signature_names "
                    f"in model: {model_path}"
                )

            all_thresholds.append(thresholds[signature_order])

    if sample_ids is None:
        raise RuntimeError("No sample IDs were obtained from the models.")

    mean_probabilities = np.mean(all_probabilities, axis=0)

    if len(all_thresholds) == len(model_paths):
        mean_thresholds = np.mean(all_thresholds, axis=0)
        threshold_source = "mean checkpoint thresholds"
    else:
        mean_thresholds = np.full(
            len(signature_names),
            fallback_prediction_threshold,
            dtype=float,
        )
        threshold_source = (
            f"fallback fixed threshold "
            f"{fallback_prediction_threshold}"
        )

    print(
        f"[Model] Loaded {len(model_paths)} models; "
        f"original prediction threshold source: {threshold_source}"
    )

    return mean_probabilities, sample_ids, mean_thresholds


# =========================================================
# Original NNLS baseline
# =========================================================
def calculate_original_nnls(
    catalog: pd.DataFrame,
    signature_matrix: np.ndarray,
    original_predictions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Re-fit every original selected signature set using NNLS."""
    signature_names = original_predictions.index.tolist()
    sample_ids = original_predictions.columns.tolist()

    exposure_values = np.zeros(
        (len(signature_names), len(sample_ids)),
        dtype=float,
    )
    cosines: Dict[str, float] = {}

    for sample_position, sample_name in enumerate(sample_ids):
        selected = np.where(
            original_predictions[sample_name].values.astype(float) > 0
        )[0].tolist()

        coefficients, _, cosine = fit_nnls(
            signature_matrix,
            catalog[sample_name].values.astype(float),
            selected,
        )

        if selected:
            exposure_values[selected, sample_position] = coefficients

        cosines[sample_name] = cosine

    exposures = pd.DataFrame(
        exposure_values,
        index=signature_names,
        columns=sample_ids,
    )

    return exposures, pd.Series(cosines, name="original_cosine")


# =========================================================
# Evaluation
# =========================================================
def safe_precision_recall_f1(
    ground_truth_binary: np.ndarray,
    prediction_binary: np.ndarray,
) -> Tuple[float, float, float, int, int, int]:
    """Calculate binary precision, recall and F1."""
    truth = np.asarray(ground_truth_binary, dtype=bool)
    prediction = np.asarray(prediction_binary, dtype=bool)

    tp = int(np.logical_and(truth, prediction).sum())
    fp = int(np.logical_and(~truth, prediction).sum())
    fn = int(np.logical_and(truth, ~prediction).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, tp, fp, fn


def calculate_macro_sample_f1(
    ground_truth_binary: pd.DataFrame,
    prediction_binary: pd.DataFrame,
) -> float:
    """Calculate F1 for each sample, then average over samples."""
    sample_f1_values: List[float] = []

    for sample_name in ground_truth_binary.columns:
        truth = ground_truth_binary[sample_name].values > 0
        prediction = prediction_binary[sample_name].values > 0

        _, _, f1, _, _, _ = safe_precision_recall_f1(
            truth,
            prediction,
        )
        sample_f1_values.append(f1)

    return (
        float(np.mean(sample_f1_values))
        if sample_f1_values
        else 0.0
    )


def calculate_reconstruction_cosines(
    catalog: pd.DataFrame,
    signatures: pd.DataFrame,
    exposures: pd.DataFrame,
) -> pd.Series:
    """Calculate sample-wise reconstruction cosine."""
    reconstruction = signatures.values.astype(float) @ exposures.values

    reconstruction_df = pd.DataFrame(
        reconstruction,
        index=signatures.index,
        columns=exposures.columns,
    )

    values = {
        sample_name: cosine_similarity(
            catalog[sample_name].values.astype(float),
            reconstruction_df[sample_name].values.astype(float),
        )
        for sample_name in exposures.columns
    }

    return pd.Series(values, name="final_cosine")


def evaluate_predictions(
    ground_truth_exposures: pd.DataFrame,
    predicted_exposures: pd.DataFrame,
    final_cosines: pd.Series,
    rare_signature_max_occurrences: int,
) -> Dict[str, float]:
    """Calculate all evaluation metrics for one parameter combination."""
    ground_truth_exposures = ground_truth_exposures.astype(float)
    predicted_exposures = predicted_exposures.astype(float)

    gt_binary = ground_truth_exposures > 0
    pred_binary = predicted_exposures > EXPOSURE_EPSILON

    (
        precision,
        recall,
        f1,
        tp,
        fp,
        fn,
    ) = safe_precision_recall_f1(
        gt_binary.values,
        pred_binary.values,
    )

    macro_f1 = calculate_macro_sample_f1(
        gt_binary.astype(int),
        pred_binary.astype(int),
    )

    absolute_error = (
        predicted_exposures - ground_truth_exposures
    ).abs()

    sample_tae = absolute_error.sum(axis=0)
    ground_truth_totals = ground_truth_exposures.sum(axis=0)

    normalized_sample_tae = sample_tae / ground_truth_totals.replace(
        0,
        np.nan,
    )
    normalized_sample_tae = normalized_sample_tae.fillna(0)

    occurrence_counts = gt_binary.sum(axis=1)
    rare_signatures = occurrence_counts[
        (occurrence_counts > 0)
        & (occurrence_counts <= rare_signature_max_occurrences)
    ].index.tolist()

    if rare_signatures:
        rare_truth = gt_binary.loc[rare_signatures].values
        rare_prediction = pred_binary.loc[rare_signatures].values
        (
            rare_precision,
            rare_recall,
            rare_f1,
            _,
            _,
            _,
        ) = safe_precision_recall_f1(
            rare_truth,
            rare_prediction,
        )
    else:
        rare_precision = np.nan
        rare_recall = np.nan
        rare_f1 = np.nan

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_sample_f1": float(macro_f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mean_tae": float(sample_tae.mean()),
        "median_tae": float(sample_tae.median()),
        "mean_normalized_tae": float(
            normalized_sample_tae.mean()
        ),
        "median_normalized_tae": float(
            normalized_sample_tae.median()
        ),
        "mean_final_cosine": float(final_cosines.mean()),
        "median_final_cosine": float(final_cosines.median()),
        "rare_signature_count": int(len(rare_signatures)),
        "rare_precision": float(rare_precision),
        "rare_recall": float(rare_recall),
        "rare_f1": float(rare_f1),
    }


# =========================================================
# Run one parameter combination
# =========================================================
def run_parameter_combination(
    catalog: pd.DataFrame,
    signatures: pd.DataFrame,
    ground_truth_exposures: pd.DataFrame,
    probabilities: np.ndarray,
    original_predictions: pd.DataFrame,
    original_exposures: pd.DataFrame,
    original_cosines: pd.Series,
    entry_cosine_threshold: float,
    probability_threshold: float,
    min_contribution: float,
    min_improvement: float,
    target_cosine: float,
    max_active_signatures: int,
    rare_signature_max_occurrences: int,
) -> Tuple[
    Dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run refinement for a single pair of thresholds.

    Samples with original cosine >= entry_cosine_threshold keep their
    original NNLS result unchanged.
    """
    signature_names = signatures.columns.tolist()
    sample_ids = catalog.columns.tolist()
    signature_matrix = signatures.values.astype(float)

    final_exposures = original_exposures.copy()
    sample_rows: List[Dict[str, object]] = []

    low_cosine_samples = original_cosines[
        original_cosines < entry_cosine_threshold
    ].index.tolist()

    improved_count = 0

    sample_to_position = {
        sample_name: position
        for position, sample_name in enumerate(sample_ids)
    }

    for sample_name in sample_ids:
        original_cosine = float(original_cosines[sample_name])

        if sample_name not in low_cosine_samples:
            sample_rows.append({
                "sample": sample_name,
                "entered_refinement": False,
                "original_cosine": original_cosine,
                "attempted_final_cosine": original_cosine,
                "final_cosine": original_cosine,
                "improvement": 0.0,
                "use_refined": False,
            })
            continue

        sample_position = sample_to_position[sample_name]

        result = refine_sample(
            sample_counts=catalog[sample_name].values.astype(float),
            signature_matrix=signature_matrix,
            probabilities=probabilities[sample_position],
            original_predictions=(
                original_predictions[sample_name].values.astype(float)
            ),
            probability_threshold=probability_threshold,
            min_contribution=min_contribution,
            target_cosine=target_cosine,
            min_improvement=min_improvement,
            max_active_signatures=max_active_signatures,
        )

        final_exposures[sample_name] = 0.0
        selected = result["selected"]

        if selected:
            final_exposures.loc[
                [signature_names[index] for index in selected],
                sample_name,
            ] = np.asarray(result["coefficients"], dtype=float)

        if bool(result["use_refined"]):
            improved_count += 1

        sample_rows.append({
            "sample": sample_name,
            "entered_refinement": True,
            "original_cosine": original_cosine,
            "attempted_final_cosine": result[
                "attempted_final_cosine"
            ],
            "final_cosine": result["final_cosine"],
            "improvement": result["improvement"],
            "use_refined": result["use_refined"],
        })

    final_predictions = (final_exposures > 0).astype(int)

    final_cosines = calculate_reconstruction_cosines(
        catalog=catalog,
        signatures=signatures,
        exposures=final_exposures,
    )

    metrics = evaluate_predictions(
        ground_truth_exposures=ground_truth_exposures,
        predicted_exposures=final_exposures,
        final_cosines=final_cosines,
        rare_signature_max_occurrences=(
            rare_signature_max_occurrences
        ),
    )

    low_sample_set = set(low_cosine_samples)
    low_sample_ids = [
        sample_name
        for sample_name in sample_ids
        if sample_name in low_sample_set
    ]

    if low_sample_ids:
        low_metrics = evaluate_predictions(
            ground_truth_exposures=(
                ground_truth_exposures[low_sample_ids]
            ),
            predicted_exposures=final_exposures[low_sample_ids],
            final_cosines=final_cosines[low_sample_ids],
            rare_signature_max_occurrences=(
                rare_signature_max_occurrences
            ),
        )
    else:
        low_metrics = {
            "f1": np.nan,
            "macro_sample_f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "mean_normalized_tae": np.nan,
            "mean_final_cosine": np.nan,
        }

    metrics.update({
        "entry_cosine_threshold": entry_cosine_threshold,
        "probability_threshold": probability_threshold,
        "min_contribution": min_contribution,
        "min_improvement": min_improvement,
        "target_cosine": target_cosine,
        "max_active_signatures": max_active_signatures,
        "total_samples": len(sample_ids),
        "low_cosine_samples": len(low_cosine_samples),
        "refined_samples": improved_count,
        "low_sample_precision": low_metrics["precision"],
        "low_sample_recall": low_metrics["recall"],
        "low_sample_f1": low_metrics["f1"],
        "low_sample_macro_f1": low_metrics["macro_sample_f1"],
        "low_sample_mean_normalized_tae": (
            low_metrics["mean_normalized_tae"]
        ),
        "low_sample_mean_cosine": (
            low_metrics["mean_final_cosine"]
        ),
    })

    sample_summary = pd.DataFrame(sample_rows)

    return metrics, final_exposures, sample_summary


# =========================================================
# Best-parameter selection
# =========================================================
def choose_best_result(
    results: pd.DataFrame,
    selection_metric: str,
) -> pd.Series:
    """
    Select the best parameter row.

    The default primary metric is ``low_sample_f1`` because refinement is
    applied only to samples whose original cosine is below the entry threshold.

    Tie-break order for low_sample_f1:
    1. Higher low_sample_f1
    2. Lower low_sample_mean_normalized_tae
    3. Higher low_sample_mean_cosine
    4. Higher overall f1
    5. Lower overall mean_normalized_tae
    6. Higher overall mean_final_cosine
    """
    if selection_metric not in results.columns:
        raise ValueError(
            f"Unknown selection metric: {selection_metric}"
        )

    if selection_metric == "low_sample_f1":
        ordered = results.sort_values(
            by=[
                "low_sample_f1",
                "low_sample_mean_normalized_tae",
                "low_sample_mean_cosine",
                "f1",
                "mean_normalized_tae",
                "mean_final_cosine",
            ],
            ascending=[
                False,
                True,
                False,
                False,
                True,
                False,
            ],
            na_position="last",
        )
        return ordered.iloc[0]

    ascending_main = selection_metric in {
        "mean_tae",
        "median_tae",
        "mean_normalized_tae",
        "median_normalized_tae",
        "low_sample_mean_normalized_tae",
    }

    ordered = results.sort_values(
        by=[
            selection_metric,
            "mean_normalized_tae",
            "f1",
            "mean_final_cosine",
        ],
        ascending=[
            ascending_main,
            True,
            False,
            False,
        ],
        na_position="last",
    )

    return ordered.iloc[0]


# =========================================================
# Main grid search
# =========================================================
def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated list of floats."""
    try:
        values = [
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float list: {value}"
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one numeric value is required."
        )

    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search contribution and cosine-improvement "
            "thresholds for AMuSA refinement."
        )
    )

    parser.add_argument(
        "--mutation_file",
        default=DEFAULT_MUTATION_FILE,
    )
    parser.add_argument(
        "--signature_file",
        default=DEFAULT_SIGNATURE_FILE,
    )
    parser.add_argument(
        "--ground_truth_exposures",
        default=DEFAULT_GROUND_TRUTH_EXPOSURES,
    )
    parser.add_argument(
        "--original_predictions_file",
        default="",
        help=(
            "Optional original signature-by-sample prediction CSV. "
            "When omitted, predictions are generated from ensemble "
            "probabilities and checkpoint thresholds."
        ),
    )
    parser.add_argument(
        "--base_model_dir",
        default=DEFAULT_BASE_MODEL_DIR,
    )
    parser.add_argument(
        "--model_type",
        choices=["SBS", "DBS", "ID"],
        default="SBS",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--entry_cosine_threshold",
        type=float,
        default=0.95,
        help="Only original cosine below this value enters refinement.",
    )
    parser.add_argument(
        "--probability_values",
        type=parse_float_list,
        default=[0.05],
        help=(
            "Comma-separated candidate-pool probability thresholds "
            "to search."
        ),
    )
    parser.add_argument(
        "--target_cosine",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_active_signatures",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--fallback_prediction_threshold",
        type=float,
        default=0.40,
        help=(
            "Used only when checkpoint thresholds are unavailable "
            "and no original prediction file is supplied."
        ),
    )

    parser.add_argument(
        "--contribution_values",
        type=parse_float_list,
        default=parse_float_list(
            "0.005,0.01,0.02,0.03,0.04,0.05"
        ),
    )
    parser.add_argument(
        "--improvement_values",
        type=parse_float_list,
        default=parse_float_list(
            "0.005,0.01,0.015,0.02,0.03"
        ),
    )
    parser.add_argument(
        "--selection_metric",
        default="low_sample_f1",
        choices=[
            "f1",
            "macro_sample_f1",
            "precision",
            "recall",
            "mean_tae",
            "mean_normalized_tae",
            "mean_final_cosine",
            "rare_f1",
            "rare_recall",
            "low_sample_f1",
            "low_sample_mean_normalized_tae",
        ],
    )
    parser.add_argument(
        "--rare_signature_max_occurrences",
        type=int,
        default=5,
        help=(
            "A signature appearing in at most this many samples is "
            "treated as rare."
        ),
    )
    parser.add_argument(
        "--save_all_sample_summaries",
        action="store_true",
        help=(
            "Save a sample-level summary for every threshold pair. "
            "This creates many files."
        ),
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Read and align catalog/signature matrices
    # -----------------------------------------------------
    catalog = read_csv_matrix(args.mutation_file)
    signatures = read_csv_matrix(args.signature_file)

    missing_mutation_types = signatures.index.difference(
        catalog.index
    )
    if len(missing_mutation_types) > 0:
        raise ValueError(
            "Mutation catalog is missing mutation types required by "
            f"the signature matrix: {missing_mutation_types.tolist()}"
        )

    catalog = catalog.reindex(signatures.index)
    catalog = catalog.apply(pd.to_numeric, errors="coerce").fillna(0)
    signatures = signatures.apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)

    signature_names = signatures.columns.tolist()

    # -----------------------------------------------------
    # Run model ensemble only once
    # -----------------------------------------------------
    (
        probabilities,
        model_sample_ids,
        mean_prediction_thresholds,
    ) = predict_ensemble_probabilities_and_thresholds(
        mutation_file=args.mutation_file,
        signature_file=args.signature_file,
        base_model_dir=args.base_model_dir,
        model_type=args.model_type,
        signature_names=signature_names,
        fallback_prediction_threshold=(
            args.fallback_prediction_threshold
        ),
    )

    missing_model_samples = [
        sample_name
        for sample_name in model_sample_ids
        if sample_name not in catalog.columns
    ]
    if missing_model_samples:
        raise ValueError(
            "Model output contains samples absent from the mutation "
            f"catalog: {missing_model_samples}"
        )

    catalog = catalog.reindex(columns=model_sample_ids)
    sample_ids = catalog.columns.tolist()

    # -----------------------------------------------------
    # Align ground-truth exposures
    # -----------------------------------------------------
    ground_truth_raw = read_csv_matrix(
        args.ground_truth_exposures
    )
    ground_truth_exposures = align_signature_sample_matrix(
        frame=ground_truth_raw,
        signature_names=signature_names,
        sample_ids=sample_ids,
        matrix_name="Ground-truth exposures",
    )

    # -----------------------------------------------------
    # Load or generate original predictions
    # -----------------------------------------------------
    if args.original_predictions_file:
        original_raw = read_csv_matrix(
            args.original_predictions_file
        )
        original_predictions = align_signature_sample_matrix(
            frame=original_raw,
            signature_names=signature_names,
            sample_ids=sample_ids,
            matrix_name="Original predictions",
        )
        original_predictions = (
            original_predictions > 0
        ).astype(int)
        original_prediction_source = (
            args.original_predictions_file
        )
    else:
        original_binary = (
            probabilities
            >= mean_prediction_thresholds.reshape(1, -1)
        ).astype(int)

        original_predictions = pd.DataFrame(
            original_binary.T,
            index=signature_names,
            columns=sample_ids,
        )
        original_prediction_source = (
            "generated from ensemble probabilities and model thresholds"
        )

    print(
        f"[Input] Original predictions: "
        f"{original_prediction_source}"
    )

    original_predictions.to_csv(
        os.path.join(
            args.output_dir,
            "original_predictions_used.csv",
        )
    )

    # -----------------------------------------------------
    # Calculate original NNLS baseline
    # -----------------------------------------------------
    (
        original_exposures,
        original_cosines,
    ) = calculate_original_nnls(
        catalog=catalog,
        signature_matrix=signatures.values.astype(float),
        original_predictions=original_predictions,
    )

    low_cosine_count = int(
        (original_cosines < args.entry_cosine_threshold).sum()
    )

    print(
        f"[Baseline] Samples: {len(sample_ids)}; "
        f"cosine < {args.entry_cosine_threshold}: "
        f"{low_cosine_count}"
    )

    original_exposures.to_csv(
        os.path.join(
            args.output_dir,
            "original_nnls_exposures.csv",
        )
    )
    original_cosines.to_csv(
        os.path.join(
            args.output_dir,
            "original_cosines.csv",
        ),
        header=True,
    )

    baseline_cosines = calculate_reconstruction_cosines(
        catalog=catalog,
        signatures=signatures,
        exposures=original_exposures,
    )
    baseline_metrics = evaluate_predictions(
        ground_truth_exposures=ground_truth_exposures,
        predicted_exposures=original_exposures,
        final_cosines=baseline_cosines,
        rare_signature_max_occurrences=(
            args.rare_signature_max_occurrences
        ),
    )

    pd.DataFrame([baseline_metrics]).to_csv(
        os.path.join(
            args.output_dir,
            "baseline_metrics.csv",
        ),
        index=False,
    )

    # -----------------------------------------------------
    # Grid search
    # -----------------------------------------------------
    total_combinations = (
        len(args.probability_values)
        * len(args.contribution_values)
        * len(args.improvement_values)
    )

    print(
        f"[Search] Probability values: "
        f"{args.probability_values}"
    )
    print(
        f"[Search] Contribution values: "
        f"{args.contribution_values}"
    )
    print(
        f"[Search] Improvement values: "
        f"{args.improvement_values}"
    )
    print(
        f"[Search] Total combinations: "
        f"{total_combinations}"
    )

    all_metrics: List[Dict[str, float]] = []
    best_outputs: Dict[
        Tuple[float, float, float],
        Tuple[pd.DataFrame, pd.DataFrame],
    ] = {}

    summaries_dir = os.path.join(
        args.output_dir,
        "all_sample_summaries",
    )
    if args.save_all_sample_summaries:
        os.makedirs(summaries_dir, exist_ok=True)

    combination_number = 0

    for probability_threshold in args.probability_values:
        for min_contribution in args.contribution_values:
            for min_improvement in args.improvement_values:
                combination_number += 1

                print(
                    f"[{combination_number}/{total_combinations}] "
                    f"probability_threshold={probability_threshold}, "
                    f"min_contribution={min_contribution}, "
                    f"min_improvement={min_improvement}"
                )

                (
                    metrics,
                    final_exposures,
                    sample_summary,
                ) = run_parameter_combination(
                    catalog=catalog,
                    signatures=signatures,
                    ground_truth_exposures=ground_truth_exposures,
                    probabilities=probabilities,
                    original_predictions=original_predictions,
                    original_exposures=original_exposures,
                    original_cosines=original_cosines,
                    entry_cosine_threshold=(
                        args.entry_cosine_threshold
                    ),
                    probability_threshold=probability_threshold,
                    min_contribution=min_contribution,
                    min_improvement=min_improvement,
                    target_cosine=args.target_cosine,
                    max_active_signatures=(
                        args.max_active_signatures
                    ),
                    rare_signature_max_occurrences=(
                        args.rare_signature_max_occurrences
                    ),
                )

                all_metrics.append(metrics)
                key = (
                    probability_threshold,
                    min_contribution,
                    min_improvement,
                )
                best_outputs[key] = (
                    final_exposures,
                    sample_summary,
                )

                if args.save_all_sample_summaries:
                    summary_name = (
                        f"summary_probability_{probability_threshold:g}"
                        f"_contribution_{min_contribution:g}"
                        f"_improvement_{min_improvement:g}.csv"
                    )
                    sample_summary.to_csv(
                        os.path.join(
                            summaries_dir,
                            summary_name,
                        ),
                        index=False,
                    )

    results = pd.DataFrame(all_metrics)

    results_file = os.path.join(
        args.output_dir,
        "threshold_grid_search_results.csv",
    )
    results.to_csv(results_file, index=False)

    # -----------------------------------------------------
    # Select and save best result
    # -----------------------------------------------------
    best_row = choose_best_result(
        results=results,
        selection_metric=args.selection_metric,
    )

    best_probability = float(
        best_row["probability_threshold"]
    )
    best_contribution = float(best_row["min_contribution"])
    best_improvement = float(best_row["min_improvement"])
    best_key = (
        best_probability,
        best_contribution,
        best_improvement,
    )

    best_exposures, best_sample_summary = best_outputs[best_key]
    best_predictions = (best_exposures > EXPOSURE_EPSILON).astype(int)

    best_dir = os.path.join(args.output_dir, "best_result")
    os.makedirs(best_dir, exist_ok=True)

    best_exposures.to_csv(
        os.path.join(best_dir, "best_exposures_float.csv")
    )
    best_exposures.round(0).astype(int).to_csv(
        os.path.join(best_dir, "best_exposures.csv")
    )
    best_predictions.to_csv(
        os.path.join(best_dir, "best_predictions.csv")
    )
    best_sample_summary.to_csv(
        os.path.join(best_dir, "best_sample_summary.csv"),
        index=False,
    )
    pd.DataFrame([best_row]).to_csv(
        os.path.join(best_dir, "best_metrics.csv"),
        index=False,
    )

    settings = {
        "mutation_file": args.mutation_file,
        "signature_file": args.signature_file,
        "ground_truth_exposures": (
            args.ground_truth_exposures
        ),
        "original_predictions_file": (
            args.original_predictions_file
        ),
        "base_model_dir": args.base_model_dir,
        "model_type": args.model_type,
        "entry_cosine_threshold": (
            args.entry_cosine_threshold
        ),
        "probability_values": args.probability_values,
        "target_cosine": args.target_cosine,
        "max_active_signatures": (
            args.max_active_signatures
        ),
        "selection_metric": args.selection_metric,
        "best_probability_threshold": best_probability,
        "best_min_contribution": best_contribution,
        "best_min_improvement": best_improvement,
    }

    with open(
        os.path.join(best_dir, "best_settings.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)

    # Sorted table for convenient inspection.
    if args.selection_metric == "low_sample_f1":
        sorted_results = results.sort_values(
            by=[
                "low_sample_f1",
                "low_sample_mean_normalized_tae",
                "low_sample_mean_cosine",
                "f1",
                "mean_normalized_tae",
                "mean_final_cosine",
            ],
            ascending=[
                False,
                True,
                False,
                False,
                True,
                False,
            ],
            na_position="last",
        )
    else:
        sorted_results = results.sort_values(
            by=[
                args.selection_metric,
                "mean_normalized_tae",
                "f1",
                "mean_final_cosine",
            ],
            ascending=[
                args.selection_metric in {
                    "mean_tae",
                    "median_tae",
                    "mean_normalized_tae",
                    "median_normalized_tae",
                    "low_sample_mean_normalized_tae",
                },
                True,
                False,
                False,
            ],
            na_position="last",
        )

    sorted_results.to_csv(
        os.path.join(
            args.output_dir,
            "threshold_grid_search_results_sorted.csv",
        ),
        index=False,
    )

    print("\n" + "=" * 70)
    print("Grid search completed")
    print("=" * 70)
    print(f"Results: {results_file}")
    print(
        f"Selection metric: {args.selection_metric}"
    )
    print(
        f"Best probability_threshold: {best_probability}"
    )
    print(
        f"Best min_contribution: {best_contribution}"
    )
    print(
        f"Best min_improvement: {best_improvement}"
    )
    print(
        f"Best low-sample F1: "
        f"{best_row['low_sample_f1']:.6f}"
    )
    print(
        f"Best low-sample Precision: "
        f"{best_row['low_sample_precision']:.6f}"
    )
    print(
        f"Best low-sample Recall: "
        f"{best_row['low_sample_recall']:.6f}"
    )
    print(
        "Best low-sample mean normalized TAE: "
        f"{best_row['low_sample_mean_normalized_tae']:.6f}"
    )
    print(
        "Best low-sample mean cosine: "
        f"{best_row['low_sample_mean_cosine']:.6f}"
    )
    print(
        f"Overall F1 of the selected row: "
        f"{best_row['f1']:.6f}"
    )
    print(
        f"Overall Precision of the selected row: "
        f"{best_row['precision']:.6f}"
    )
    print(
        f"Overall Recall of the selected row: "
        f"{best_row['recall']:.6f}"
    )
    print(f"Best files saved in: {best_dir}")


if __name__ == "__main__":
    main()