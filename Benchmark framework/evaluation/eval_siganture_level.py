#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

EPSILON = 1e-8


def evaluate_signatures_per_signature(
    true_res,
    df,
    muts,
    output_path,
    extra_col=None
):
    """
    Evaluate per-signature performance metrics for mutational
    signature assignment.

    Parameters
    ----------
    true_res : pd.DataFrame
        Ground truth exposure matrix.
        Rows represent signatures and columns represent samples.

    df : pd.DataFrame
        Predicted exposure matrix.
        Rows represent signatures and columns represent samples.

    muts : pd.Series
        Total mutation counts for each sample.

    output_path : str
        Directory for saving output results.

    extra_col : str, optional
        Extra label appended to the output filename.

    Returns
    -------
    pd.DataFrame
        Per-signature evaluation metrics.
    """

    results = []

    # ==========================================================
    # Fill missing signature rows
    # ==========================================================
    for sig in df.index:
        if sig not in true_res.index:
            true_res.loc[sig] = 0

    for sig in true_res.index:
        if sig not in df.index:
            df.loc[sig] = 0

    # ==========================================================
    # Evaluate each signature
    # ==========================================================
    for sig in true_res.index:

        # Initialize counters
        TP = FP = FN = TN = 0

        MAE_sig = 0

        true_vals = []
        pred_vals = []

        # ------------------------------------------------------
        # Iterate through samples
        # ------------------------------------------------------
        for sample in true_res.columns:

            true_val = true_res.loc[sig, sample]
            pred_val = df.loc[sig, sample]

            true_vals.append(true_val)
            pred_vals.append(pred_val)

            # Active signature
            if true_val > 0:

                MAE_sig += (
                    np.abs(pred_val - true_val) /
                    muts[sample]
                )

                if pred_val == 0:
                    FN += 1
                else:
                    TP += 1

            # Inactive signature
            else:

                if pred_val > 0:
                    FP += 1
                else:
                    TN += 1

        # ------------------------------------------------------
        # Average MAE across samples
        # ------------------------------------------------------
        MAE_sig /= len(true_res.columns)

        # ------------------------------------------------------
        # Classification metrics
        # ------------------------------------------------------
        Precision = (
            TP / (TP + FP)
            if (TP + FP) > 0 else 0
        )

        Recall = (
            TP / (TP + FN)
            if (TP + FN) > 0 else 0
        )

        F1 = (
            2 * Precision * Recall /
            (Precision + Recall)
            if (Precision + Recall) > 0 else 0
        )

        MCC = (
            (TP * TN - FP * FN) /
            np.sqrt(
                (TP + FP) *
                (TP + FN) *
                (TN + FP) *
                (TN + FN)
            )
            if (
                (TP + FP) *
                (TP + FN) *
                (TN + FP) *
                (TN + FN)
            ) > 0
            else 0
        )

        # ------------------------------------------------------
        # False positive / false negative weights
        # ------------------------------------------------------
        wT_FP = sum(
            df.loc[sig, s] / muts[s]
            for s in true_res.columns
            if (
                true_res.loc[sig, s] == 0 and
                df.loc[sig, s] > 0
            )
        )

        wT_FN = sum(
            true_res.loc[sig, s] / muts[s]
            for s in true_res.columns
            if (
                true_res.loc[sig, s] > 0 and
                df.loc[sig, s] == 0
            )
        )

        # ------------------------------------------------------
        # Pearson correlation
        # ------------------------------------------------------
        if (
            np.std(true_vals) > EPSILON and
            np.std(pred_vals) > EPSILON
        ):
            Pearson = pearsonr(
                true_vals,
                pred_vals
            )[0]
        else:
            Pearson = np.nan

        # ------------------------------------------------------
        # TAE / nRMSE
        # ------------------------------------------------------
        err = (
            np.abs(
                np.array(pred_vals) -
                np.array(true_vals)
            ) /
            (2 * muts[true_res.columns].values)
        )

        TAE = err.mean()

        TAE_std = err.std()

        nRMSE = np.sqrt(
            np.mean(err ** 2)
        )

        # ------------------------------------------------------
        # Store metrics
        # ------------------------------------------------------
        results.append({
            "Signature": sig,
            "MAE": MAE_sig,
            "TAE": TAE,
            "TAE_std": TAE_std,
            "nRMSE": nRMSE,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1,
            "MCC": MCC,
            "wT_FP": wT_FP,
            "wT_FN": wT_FN,
            "Pearson": Pearson
        })

    # ==========================================================
    # Convert results to DataFrame
    # ==========================================================
    results_df = pd.DataFrame(results)

    # Sort by F1 score
    results_df = results_df.sort_values(
        by="F1",
        ascending=False
    )

    # Reset index
    results_df = results_df.reset_index(drop=True)

    # ==========================================================
    # Create output directory
    # ==========================================================
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # ==========================================================
    # Define output file
    # ==========================================================
    outfile = os.path.join(
        output_path,
        f"per_signature_metrics"
        f"{('_' + extra_col) if extra_col else ''}.tsv"
    )

    # ==========================================================
    # Save results
    # ==========================================================
    results_df.to_csv(
        outfile,
        sep='\t',
        index=False
    )

    print(
        f"Saved per-signature metrics to: {outfile}"
    )

    return results_df