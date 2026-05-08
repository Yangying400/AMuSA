#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

EPSILON = 1e-8

# Ensure that the file with results starts with a header
def check_open(oname, extra_col):
    if not os.path.isfile(oname):
        with open(oname, 'w') as obj:
            if extra_col == 'real_data':
                obj.write('run\tsamples\tmuts\tTAE\tTAE_std\twT\tn_FP\twT_FP\tn_FN\twT_FN\tP\tP_std\tR\tR_std\tF1\tF1_std\tMCC\tMCC_std\n')
            else:
                base = '\t'.join(cfg.header_line_full.split('\t')[1:])
                if extra_col is None:
                    obj.write('sig_weights\t' + base + '\n')
                else:
                    obj.write('cancer_type\tweights\t' + base + '\tr_S1\tr_S2\tr_S3\tr_S4\tr_S5\tr_S6\n')
    else:
        obj = open(oname, 'a')
    return obj

def make_info_label(tool, code_name):
    return f"{tool}-{code_name}"

def shorten_string(s, max_len=15):
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + '...'

def evaluate_signatures(info_label, true_res, df, muts, ref_sigs_path=None, extra_col=None, line_start='', recommended=False, 
                        output_path="/home/yangying/evaluate_signatures/", top_sigs=None, code_name="default_code"):
    output_file_path = os.path.join(output_path, f"results-WGS-{info_label}-{code_name}.dat")

    # Read reference signature file if provided
    if ref_sigs_path is not None:
        try:
            tmp = pd.read_csv(ref_sigs_path, sep='\t', index_col=0)
        except Exception as e:
            print(f"Warning: Failed to read reference signature file: {e}")
            tmp = None
        if tmp is not None:
            for sig in tmp.columns:
                if sig not in true_res.index:
                    true_res.loc[sig] = 0
                if sig not in df.index:
                    df.loc[sig] = 0

    # Fill missing signature rows
    for ix in df.index:
        if ix not in true_res.index:
            true_res.loc[ix] = 0
    for ix in true_res.index:
        if ix not in df.index:
            df.loc[ix] = 0

    # Calculate error metrics
    err = df.sub(true_res, axis=0).abs().sum() / (2 * muts)
    TAE = err.mean()
    TAE_std = err.std()
    nRMSE = np.sqrt(np.power(err, 2).mean())

    # Initialize variables for metrics
    wtot_FP = wtot_FP_squared = num_FP_sigs = 0
    wtot_FN = num_FN_sigs = MAE_active = 0
    pearson_vals = []
    P_vals, R_vals, S_vals, F_vals, MCC_vals = [], [], [], [], []

    # Iterate over each sample
    for sample in true_res.columns:
        either_pos = (true_res[sample] > 0) | (df[sample] > 0)  # Identify signatures with positive true or estimated weight
        if either_pos.sum() >= 3:  # At least 3 positive signatures to compute Pearson correlation
            if np.std(true_res[sample][either_pos]) > EPSILON and np.std(df[sample][either_pos]) > EPSILON:
                pearson_vals.append(pearsonr(true_res[sample][either_pos], df[sample][either_pos])[0])

        # Reset counters
        wtot_FP_one_sample, num_TP, num_TN, num_FP, num_FN = 0, 0, 0, 0, 0

        # Iterate over each signature for the current sample
        for sig in true_res.index:
            true_val = true_res.loc[sig, sample]
            pred_val = df.loc[sig, sample]
            if true_val > 0:  # Active signature
                MAE_active += np.abs(pred_val - true_val) / muts[sample]  # Calculate MAE for active signatures
                if pred_val == 0:  # False negative
                    wtot_FN += true_val / muts[sample]
                    num_FN_sigs += 1
                    num_FN += 1
                else:
                    num_TP += 1
            else:  # Inactive signature
                if pred_val > 0:  # False positive
                    wtot_FP += pred_val / muts[sample]
                    wtot_FP_one_sample += pred_val
                    num_FP_sigs += 1
                    num_FP += 1
                else:  # True negative
                    num_TN += 1

        # Normalize false positive weight
        wtot_FP_one_sample /= muts[sample]
        wtot_FP_squared += wtot_FP_one_sample * wtot_FP_one_sample

        # Calculate Precision, Recall, Specificity, F1, and MCC
        Psample = num_TP / (num_TP + num_FP) if (num_TP + num_FP) > 0 else 0
        Rsample = num_TP / (num_TP + num_FN) if (num_TP + num_FN) > 0 else 0
        Ssample = num_TN / (num_TN + num_FP) if (num_TN + num_FP) > 0 else 0
        Fsample = 2 * Psample * Rsample / (Psample + Rsample) if (Psample + Rsample) > 0 else 0
        MCCsample = (num_TP * num_TN - num_FP * num_FN) / np.sqrt(
            (num_TP + num_FP) * (num_TP + num_FN) * (num_TN + num_FP) * (num_TN + num_FN)
        ) if (Psample + Rsample) > 0 and (num_TN + num_FP) > 0 and (num_TN + num_FN) > 0 else 0

        # Store metrics for each sample
        P_vals.append(Psample)
        R_vals.append(Rsample)
        S_vals.append(Ssample)
        F_vals.append(Fsample)
        MCC_vals.append(MCCsample)

    # Final metrics calculations
    MAE_active /= (2 * df.shape[1])  # normalize all metrics
    wtot_FP /= df.shape[1]
    wtot_FP_squared /= df.shape[1]
    wtot_FP_squared -= wtot_FP * wtot_FP
    num_FP_sigs /= df.shape[1]  # average number of false positive signatures per sample
    wtot_FN /= df.shape[1]
    num_FN_sigs /= df.shape[1]  # average number of false negative signatures per sample
    weight_tot = (df.sum() / muts).mean()  # sum of the assigned weights, averaged over samples, normalized

    # Calculate effective number of signatures per sample
    w_norm = df / df.sum()
    n_eff = np.ma.masked_invalid(1 / np.power(w_norm, 2).sum()).mean()
    if np.ma.is_masked(n_eff): 
        n_eff = np.nan

    # Pearson correlation calculation for samples
    pearson = np.nanmean(pearson_vals) if len(pearson_vals) >= 3 else np.nan

    # Prepare correlation output for top signatures if extra_col is provided
    corr_out = ''
    if extra_col is not None and extra_col in cfg.top_sigs.keys():
        for sig in cfg.top_sigs[extra_col]:
            if sig not in true_res.index: true_res.loc[sig] = 0  # Top signature missing in true result
            if sig not in df.index: df.loc[sig] = 0  # Top signature missing in results
            if true_res.loc[sig].std() > 0 and df.loc[sig].std() > 0:
                pearson_value = pearsonr(true_res.loc[sig], df.loc[sig])[0]
            else:
                pearson_value = np.nan
            corr_out += f'\t{pearson_value:.4f}'

    # Prepare output strings for results
    mean_num_muts = f'{muts.mean():.0f}'
    full_string = ('sig_weights \tsamples \tmuts\tTAE\tTAE_std\tRMSE\twT\tn_eff\tTAE_TP\twT_FP \twT_FP_std\tn_FP\twT_FN\tn_FN\tP\tP_std\tR\tR_std \tS \tS_std\tF1                    \tF1_std\tMCC\tMCC_std\tPearson\n'
        f'{line_start}\t{df.shape[1]}\t{mean_num_muts}\t{TAE:.4f}\t{TAE_std:.4f}\t{nRMSE:.4f}\t{weight_tot:.4f}\t{n_eff:.4f}\t'
        f'{MAE_active:.4f}\t{wtot_FP:.4f}\t{np.sqrt(wtot_FP_squared):.4f}\t{num_FP_sigs:.4f}\t{wtot_FN:.4f}\t{num_FN_sigs:.4f}\t'
        f'{np.mean(P_vals):.4f}\t{np.std(P_vals):.4f}\t{np.mean(R_vals):.4f}\t{np.std(R_vals):.4f}\t{np.mean(S_vals):.4f}\t{np.std(S_vals):.4f}\t'
        f'{np.mean(F_vals):.4f}\t{np.std(F_vals):.4f}\t{np.mean(MCC_vals):.4f}\t{np.std(MCC_vals):.4f}\t{pearson:.4f}{corr_out}\n'
    )

    short_string = (
        f'{line_start}\t{df.shape[1]}\t{mean_num_muts}\t{TAE:.4f}\t{MAE_active:.4f}\t{n_eff:.4f}\t{weight_tot:.4f}\t'
        f'{wtot_FP:.4f}\t{num_FP_sigs:.4f}\t{wtot_FN:.4f}\t{num_FN_sigs:.4f}\t'
        f'{np.mean(P_vals):.4f}\t{np.mean(R_vals):.4f}\t{np.mean(S_vals):.4f}\t{np.mean(F_vals):.4f}\t{np.mean(MCC_vals):.4f}\t{pearson:.4f}'
    )

    # Output the results
    if output_path is not None:
        if recommended:
            print(short_string + '(recommended settings)')
        else:
            print(short_string)
        with open(output_file_path, 'w') as f:
            f.write(full_string)
    else:
        if recommended:
            print(f'{shorten_string(extra_col)}\t' + short_string + '(recommended settings)')
        else:
            print(f'{shorten_string(extra_col)}\t' + short_string)
        
        # Append results to the output file
        with open(output_file_path, 'a') as output_file:
            output_file.write(extra_col + '\t' + full_string)
