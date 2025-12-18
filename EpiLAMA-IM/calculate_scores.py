import os
import tempfile
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.mhcnuggets_module import calculate_mhcnuggets_scores
from src.deeppeptide_module import process_single_file_preds
from src.utils import dict_to_fasta


def mhcnuggets_scores(df, mhcII_alleles_file):
    """Run MHCNuggets and turn predictions into [0, 1] scores."""
    # Write temporary file with peptides
    peptides = df["epi_seq"].unique()
    temp = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    np.savetxt(temp.name, peptides, fmt="%s")

    # Calculate scores
    mhcnuggets_preds = calculate_mhcnuggets_scores(
        temp.name, 
        mhcI_alleles_file=None, 
        mhcII_alleles_file=mhcII_alleles_file,
    )
    os.unlink(temp.name)
    return mhcnuggets_preds

def deeppeptide_scores(df, deeppeptide_output_dir):
    """Run DeepPeptide and turn predictions into per-residue cleave probabilities."""
    if not os.path.exists(deeppeptide_output_dir):
        os.makedirs(deeppeptide_output_dir)

    # Write temporary fasta input file
    proteins = df[["prot_id", "prot_seq"]].drop_duplicates().set_index("prot_id")["prot_seq"].to_dict()
    temp = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    dict_to_fasta(proteins, temp.name)
    
    # Run DeepPeptide
    cwd = os.getcwd()
    os.chdir("DeepPeptide/predictor")
    output_dir = os.path.join(cwd, args.deeppeptide_output_dir)
    os.system(f"python predict.py -ff {temp.name} -od {output_dir}")
    os.unlink(temp.name)
    os.chdir(cwd)
    
    # Process outputs
    dp_probas = process_single_file_preds(os.path.join(args.deeppeptide_output_dir, "sequence_outputs.json"))
    return dp_probas

def epilama_il_scores(df, cuda_device=0):
    """Run EpiLAMA-IL."""
    il_input = df[["prot_id", "prot_seq", "epi_start", "epi_end"]].drop_duplicates()
    il_input.columns = ["Protein ID", "Protein Seq", "Starting Position", "Ending Position"]

    temp_input = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    temp_output = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    il_input.to_csv(temp_input.name, index=False)
    
    cwd = os.getcwd()
    os.chdir("../EpiLAMA-IL/scripts/inference")
    os.system(f"LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m SWAT.scripts.predict -i {temp_input.name} -o {temp_output.name} --cuda {str(cuda_device)}")
    os.chdir(cwd)

    preds = pd.read_csv(temp_output.name)
    os.unlink(temp_input.name)
    os.unlink(temp_output.name)
    return preds

def calculate_scores(
    input_file, 
    mhcII_alleles_file,
    deeppeptide_output_dir,
    cuda_device=0,
    output_dir=None,
    exclude_mhcnuggets=False,
    exclude_deeppeptide=False,
    exclude_il=False,
):
    """
    Run calculation of all scores by pre-processing input data, running external tools,
    and post-processing the outputs.
    """
    if (output_dir is not None) and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(input_file)
    return_scores = []

    if not exclude_mhcnuggets:
        mhc_scores = mhcnuggets_scores(df, mhcII_alleles_file)
        if output_dir is not None:
            output_file = os.path.join(output_dir, "mhcnuggets_scores.csv")
            mhc_scores.to_csv(output_file, index=False)
        return_scores.append(mhc_scores)

    if not exclude_deeppeptide:
        dp_scores = deeppeptide_scores(df, deeppeptide_output_dir)
        if output_dir is not None:
            output_file = os.path.join(output_dir, "deeppeptide_probas.pkl")
            dp_scores.to_pickle(output_file)
        return_scores.append(dp_scores)
    
    if not exclude_il:
        il_scores = epilama_il_scores(df, cuda_device)
        if output_dir is not None:
            output_file = os.path.join(output_dir, "il_scores.csv")
            il_scores.to_csv(output_file, index=False)
        return_scores.append(il_scores)
    return return_scores

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="CSV file with parent protein sequences, peptide coordinates and peptides",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="File to save predictions to",
    )

    parser.add_argument(
        "--mhcII_alleles",
        type=str,
        default="custom_data/mhcII/human_alleles_with_trained_models.txt",
        help="File with MHC-II alleles"
    )
    parser.add_argument(
        "--deeppeptide_output_dir",
        type=str,
        default="DeepPeptide_outputs",
        help="Directory to save raw DeepPeptide outputs to",
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="CUDA device ordinal to use",
    )

    parser.add_argument(
        "--exclude_mhcnuggets",
        action='store_true', 
        default=False,
        help='Whether to exclude MHCNuggets scores from calculation',
    )
    parser.add_argument(
        "--exclude_deeppeptide",
        action='store_true', 
        default=False,
        help='Whether to exclude DeepPeptide scores from calculation',
    )
    parser.add_argument(
        "--exclude_il",
        action='store_true', 
        default=False,
        help='Whether to exclude EpiLAMA-IL scores from calculation',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scores = calculate_scores(
        input_file=args.input_file, 
        mhcII_alleles_file=args.mhcII_alleles,
        deeppeptide_output_dir=args.deeppeptide_output_dir,
        cuda_device=args.cuda_device,
        output_dir=args.output_dir,
        exclude_mhcnuggets=args.exclude_mhcnuggets,
        exclude_deeppeptide=args.exclude_deeppeptide,
        exclude_il=args.exclude_il,
    )
