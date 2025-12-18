from collections.abc import Iterable

import joblib
import numpy as np
import pandas as pd
from lightautoml.dataset.np_pd_dataset import NumpyDataset
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    average_precision_score, 
    matthews_corrcoef
)

from src.chem import calculate_physchem
from src.descriptors import calculate_aac, compute_ngram, build_aap_scores, calculate_aap


def dict_to_fasta(d, fasta_file):
    with open(fasta_file, "w") as f:
        for key, val in d.items():
            f.write(f">{key}\n")
            f.write(val)
            f.write("\n")

def fasta_to_dict(fasta_file):
    with open(fasta_file) as f:
        seqs = dict()
        for line in f:
            if line.startswith(">"):
                prot_id = line.rstrip()[1:]
            else:
                seqs[prot_id] = line.rstrip()
    return seqs

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) 
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def to_npy_array(data):
    """Convert LAML NumpyDataset to np.array, if applicable"""
    if isinstance(data, NumpyDataset):
        data = data.data
    return data

def save_training_results(
    filename, 
    trained_model, 
    metrics_dict, 
    input_features, 
    test_preds=None,
    aap_dipeptide_scores=None,
):
    """Save trained models dict, metrics, input features and test predictions using joblib."""
    to_save = {
        "model": trained_model, 
        "metrics": metrics_dict,
        "input_features": input_features,
    }
    if test_preds is not None:
        to_save["test_preds"] = test_preds
    if aap_dipeptide_scores is not None:
        to_save["aap_scores"] = aap_dipeptide_scores
    joblib.dump(to_save, filename)


def propeptide_cleave_proba(row, neighborhood_size=5):
    """
    Aggregate propeptide cleave probability predicted by DeepPeptide for a single 
    peptide.

    Parameters
    ----------
    row: dict-like
        Object that has peptide information given by the following keys: 
        * row["propeptide_proba"] is a np.array of per-residue probabilities of being 
        part of a propeptide,
        * row["epi_start"] is the index of peptide start (inclusive),
        * row["epi_end"] is the index of peptide end (exclusive).
    neighborhood_size: int, optional
        Size of the neighborhood used to average the scores.

    Returns 
    -------
    float
        Mean probability of being a propeptide for the N-terminal peptide neighborhood.
    """
    if np.isscalar(row["propeptide_proba"]):
        return 0.

    left_start = max(0, row["epi_start"] - neighborhood_size)
    left_end = row["epi_start"]

    if left_start == left_end:
        mean_proba = 0.
    else:
        mean_proba = row["propeptide_proba"][left_start:left_end].mean()
    return mean_proba


def peptide_cleave_proba(row):
    """
    Aggregate cleave probability predicted by DeepPeptide for a single peptide.

    Parameters
    ----------
    row: dict-like
        Object that has peptide information given by the following keys: 
        * row["peptide_proba"] is a np.array of per-residue probabilities of being 
        part of a peptide,
        * row["epi_start"] is the index of peptide start (inclusive),
        * row["epi_end"] is the index of peptide end (exclusive).

    Returns 
    -------
    float
        Mean probability of being cleaved for a given peptide.
    """
    if np.isscalar(row["peptide_proba"]):
        return 0.

    mean_proba = row["peptide_proba"][row["epi_start"]:row["epi_end"]].mean()
    return mean_proba


def load_scores(
    df, # DataFrame with peptides to keep. If None, `mhcnuggets_peptides` will be used
    mhcnuggets_scores, 
    il_scores=None, 
    dp_probas=None, 
    scores_columns=None,
    aap_dipeptide_scores=None,
):
    """Load and merge all available scores."""
    use_all_scores = False
    if scores_columns is None:
        use_all_scores = True
    
    # MHCNuggets scores
    if df is None:
        scores = mhcnuggets_scores
    else:
        scores = pd.merge(
            df,
            mhcnuggets_scores,
            how="left",
            on=[col for col in df if col in mhcnuggets_scores],
        )
    if use_all_scores:
        scores_columns = [col for col in scores if col.startswith("HLA")]

    # IL scores
    if il_scores is not None:
        scores = pd.merge(
            scores, 
            il_scores, 
            how="left",
            left_on=["prot_id", "epi_start", "epi_end"], 
            right_on=["Protein ID", "Starting Position", "Ending Position"]
        )
        if use_all_scores:
            scores_columns.extend([col for col in scores if col.startswith("Score_")])

    # DeepPeptide scores
    if dp_probas is not None:
        # calculate mean cleaving of and around the peptide
        scores = pd.merge(
            scores, 
            dp_probas, 
            how="left", 
            on="prot_id", 
        )
        scores["peptide_proba"] = scores.apply(peptide_cleave_proba, 1)
        scores["propeptide_proba"] = scores.apply(propeptide_cleave_proba, 1)
        if use_all_scores:
            scores_columns.extend(["peptide_proba", "propeptide_proba"])

    # phys-chem scores
    chem_scores = scores["epi_seq"].apply(calculate_physchem).apply(pd.Series)
    if use_all_scores:
        scores_columns.extend(chem_scores.columns)
    scores = pd.concat((scores, chem_scores), axis=1)

    # classical sequence descriptors
    # add DPC
    dpc_scores = scores["epi_seq"].apply(compute_ngram).apply(pd.Series) 
    dpc_scores.columns = [f"DPC_{col}" for col in dpc_scores.columns]
    scores = pd.concat((scores, dpc_scores), axis=1)
    # add AAC
    aac_scores = scores["epi_seq"].apply(calculate_aac).apply(pd.Series)
    aac_scores.columns = [f"AAC_{col}" for col in aac_scores.columns]
    scores = pd.concat((scores, aac_scores), axis=1)
    # add AAP
    if aap_dipeptide_scores is None:
        positive_seqs = scores[(scores["split"] == "train") & (scores["immuno_target"] == 1)]["epi_seq"].unique()
        negative_seqs = scores[(scores["split"] == "train") & (scores["immuno_target"] == 0)]["epi_seq"].unique()
        aap_dipeptide_scores = build_aap_scores(positive_seqs, negative_seqs)
    aap_scores = scores["epi_seq"].apply(lambda x: calculate_aap(x, aap_dipeptide_scores)).apply(pd.Series)
    aap_scores.columns = [f"AAP_{col}" for col in aap_scores.columns]
    scores = pd.concat((scores, aap_scores), axis=1)
    if use_all_scores:
        scores_columns.extend(dpc_scores.columns)
        scores_columns.extend(aac_scores.columns)
        scores_columns.extend(aap_scores.columns)
    
    scores = scores[~(scores[scores_columns].isna().any(axis=1))]
    return scores, scores_columns, aap_dipeptide_scores


def print_train_test_metrics(train_metrics, test_metrics=None):
    for metric, val in train_metrics.items():
        if isinstance(val, Iterable):
            continue
        metric_str = f"{metric}:\t{val:.3f}"
        if test_metrics is not None:
            test_val = test_metrics[metric]
            metric_str = f"{metric_str} / {test_val:.3f}"
        print(metric_str)


def sliding_window_peptides(seq, pep_lengths=[8]):
    """Return all peptide substrings of a given length."""
    indexed_peptides = dict()
    for pep_length in pep_lengths:
        for i in range(len(seq) - pep_length + 1):
            left = i
            right = min(len(seq), i + pep_length)
            peptide = seq[left:right]
            indexed_peptides[(left, right)] = peptide
    return indexed_peptides


def calculate_metrics(targets, preds, thresholds=[0.5]):
    metrics_dict = {
        "ROC AUC": roc_auc_score(targets, preds),
        "PR AUC": average_precision_score(targets, preds),
    }
    for threshold in thresholds:
        metrics_dict[f"Recall @ {threshold}"] = recall_score(targets, preds > threshold)
        metrics_dict[f"Precision @ {threshold}"] = precision_score(targets, preds > threshold)
        metrics_dict[f"F1 @ {threshold}"] = f1_score(targets, preds > threshold)
        metrics_dict[f"MCC @ {threshold}"] = matthews_corrcoef(targets, preds > threshold)
    return metrics_dict