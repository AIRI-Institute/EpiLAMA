import os
import json
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.utils import softmax


def process_multifile_preds(dir):
    """Process predictions for multiple proteins stored as single files within one directory."""
    proba_preds = []
    for prot_id in os.listdir(dir):
        pred_id = ">" + prot_id
        seq_path = os.path.join(dir, prot_id, "sequence_outputs.json")

        with open(seq_path) as f:
            seq_out = json.load(f)
        prot_preds = np.array(seq_out["PREDICTIONS"][pred_id]["probabilities"])
        prot_probas = softmax(prot_preds)
        proba_preds.append({
            "prot_id": prot_id,
            "peptide_proba": prot_probas[:, 1], # probability of peptide
            "propeptide_proba": prot_probas[:, 2] # probability of propeptide
        })
    proba_preds = pd.DataFrame(proba_preds)
    return proba_preds

def process_single_file_preds(filepath):
    """Process predictions for multiple proteins stored within one file."""
    with open(filepath) as f:
        seq_out = json.load(f)
    
    prot_ids = seq_out["PREDICTIONS"].keys()
    proba_preds = []
    for prot_id in prot_ids:
        prot_preds = np.array(seq_out["PREDICTIONS"][prot_id]["probabilities"])
        prot_probas = softmax(prot_preds)
        proba_preds.append({
            "prot_id": prot_id,
            "peptide_proba": prot_probas[:, 1], # probability of peptide
            "propeptide_proba": prot_probas[:, 2] # probability of propeptide
        })
    proba_preds = pd.DataFrame(proba_preds)
    return proba_preds

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Fasta with proteins to run predictions for",
    )
    parser.add_argument(
        "-od", "--output_dir",
        type=str,
        help="Dir for raw DeepPeptide outputs"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=True,
        help="File to save per-residue probability predictions to",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cwd = os.getcwd()
    os.chdir("../DeepPeptide/predictor")
    output_dir = os.path.join(cwd, args.output_dir)
    input_file = os.path.join(cwd, args.input_file)
    os.system(f"python predict.py -ff {input_file} -od {output_dir}")
    os.chdir(cwd)
    
    proba_preds = process_single_file_preds(os.path.join(output_dir, "sequence_outputs.json"))
    pd.Series(proba_preds, name="cleave_proba").to_pickle(args.output_file)
    