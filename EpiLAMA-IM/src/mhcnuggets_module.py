import io
from argparse import ArgumentParser
from contextlib import redirect_stdout

import pandas as pd
from tqdm import tqdm
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mhcnuggets.src.predict import predict


def get_allele_preds(
    mhc_class, 
    peptides_path, 
    mhc_allele,
):
    """Get MHCnuggets predictions for peptides stored in a given path and a given allele."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        predict(
            class_=mhc_class,
            peptides_path=peptides_path, 
            mhc=mhc_allele
        )
    start_str = "peptide,ic50"
    pointer_idx = buffer.getvalue().find(start_str)
    buffer.seek(pointer_idx)
    data = pd.read_csv(buffer)
    return data

def pivot_and_score(mhcnuggets_peptides):
    """Generate multiple per-peptide [0, 1] scores from MHCNuggets-style output of 
    predicted IC50 values."""
    pivot_index_cols = [col for col in mhcnuggets_peptides.columns if col not in ("allele", "ic50")]
    peptides_pivot = mhcnuggets_peptides.pivot_table(index=pivot_index_cols, columns='allele', values='ic50', aggfunc='first')
    peptides_pivot = (1 - peptides_pivot / 5000).clip(0, 1)
    peptides_pivot = peptides_pivot.reset_index()

    if "peptide_indices" in peptides_pivot:
        peptides_pivot[["epi_start", "epi_end"]] = peptides_pivot["peptide_indices"].apply(eval).apply(pd.Series)
        peptides_pivot = peptides_pivot.drop(columns="peptide_indices")

    peptides_pivot["epi_seq"] = peptides_pivot["peptide"]
    peptides_pivot = peptides_pivot.drop(columns="peptide")
    return peptides_pivot


def calculate_mhcnuggets_scores(
    peptides_file, 
    mhcI_alleles_file=None, 
    mhcII_alleles_file=None
):
    """Calculate MHCNuggets predictions for a file with a list of peptides 
    and files with lists of MHC-I and MHC-II alleles."""
    all_preds = []
    
    # Calculate for MHC-I alleles
    if mhcI_alleles_file is not None:
        with open(mhcI_alleles_file) as f:
            mhcI_alleles = [allele.rstrip() for allele in f.readlines()]
        for allele in tqdm(mhcI_alleles):
            preds = get_allele_preds("I", peptides_file, mhc_allele=allele)
            preds["allele"] = allele
            all_preds.append(preds)
    
    # Calculate for MHC-II alleles
    if mhcII_alleles_file is not None:
        with open(mhcII_alleles_file) as f:
            mhcII_alleles = [allele.rstrip() for allele in f.readlines()]
        for allele in tqdm(mhcII_alleles):
            preds = get_allele_preds("II", peptides_file, mhc_allele=allele)
            preds["allele"] = allele
            all_preds.append(preds)
    all_preds = pd.concat(all_preds).reset_index(drop=True)

    # Pivot predictions, so they appear as multiple scores per peptide
    all_preds = pivot_and_score(all_preds)
    return all_preds

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--peptides_file",
        type=str,
        required=True,
        help="File with a list of peptides to run",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File to save predictions to",
    )

    parser.add_argument(
        "--mhcI_alleles",
        type=str,
        default="custom_data/mhcI/human_alleles_with_trained_models.txt",
        help="File with MHC-I alleles"
    )
    parser.add_argument(
        "--mhcII_alleles",
        type=str,
        default="custom_data/mhcII/human_alleles_with_trained_models.txt",
        help="File with MHC-II alleles"
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()

    all_preds = calculate_mhcnuggets_scores(
        args.peptides_file, 
        args.mhcI_alleles, 
        args.mhcII_alleles
    )
    all_preds.to_csv(args.output_file, index=None)