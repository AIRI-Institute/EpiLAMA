from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE

from src.utils import load_scores
from src.utils import calculate_metrics, print_train_test_metrics, save_training_results
from src.utils import to_npy_array

RANDOM_STATE = 14


def parse_args():
    """Create parser and parse arguments."""
    parser = ArgumentParser()

    # input files
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        help="CSV file with input proteins",
    )
    parser.add_argument(
        "--mhcnuggets",
        type=str,
        help="CSV with MHCNuggets predictions",
    )
    parser.add_argument(
        "--il_scores",
        type=str,
        help="CSV with IL scores predictions",
    )
    parser.add_argument(
        "--deeppeptide_probas",
        type=str,
        help="PKL with processed DeepPeptide per-residue probabilities",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Pre-trained .joblib model path",
    )

    # output files
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        help="PKL to save trained models, metrics and test predictions to"
    )

    # script options
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Column with target values. Will be used to calculate metrics, if provided."
    )
    return parser.parse_args()


def predict(
    input_file,
    il_scores_file,
    deeppeptide_file,
    mhcnuggets_file,
    model_file,
    output_file,
    target_col=None,
):
    """
    Run model inference and save results.
    
    Parameters
    ----------
    input_file: str
        CSV file with peptides, train/test split info ("split" column), 
        and corresponding targets ("immuno_target" column).
    il_scores_file: str
        CSV file with EpiLAMA-IL generated scores.
    deeppeptide_file: str
        PKL file with DeepPeptide-predicted probabilities.
    mhcnuggets_file: str
        CSV file with MHCNuggets generated scores.
    model_file: str
        JOBLIB file with pre-trained model, input features and AAP dipeptide scores.
    output_file: str
        CSV file to save scores and predictions to.
    target_col: str, optional
        If specified, this column will be used to calculate metrics.
    """
    # Load model
    model_dict = joblib.load(model_file)
    scores_columns = model_dict["input_features"]
    model = model_dict["model"]
    aap_dipeptide_scores = model_dict["aap_scores"]

    # load scores
    mhc_data = pd.read_csv(input_file, index_col=0)
    il_scores = pd.read_csv(il_scores_file)
    dp_probas = pd.read_pickle(deeppeptide_file)
    mhcnuggets_scores = pd.read_csv(mhcnuggets_file)
    
    # create a single dataset
    mhc_scores, scores_columns, _ = load_scores(
        mhc_data, 
        mhcnuggets_scores=mhcnuggets_scores,
        il_scores=il_scores,
        dp_probas=dp_probas,
        scores_columns=scores_columns,
        aap_dipeptide_scores=aap_dipeptide_scores,
    )

    # make and save predictions
    preds = to_npy_array(model.predict(mhc_scores[scores_columns]))[:, 0]
    mhc_scores["preds"] = preds
    mhc_scores.to_csv(output_file, index=False)

    # calculate and print metrics
    if target_col is not None:
        metrics_dict = calculate_metrics(mhc_scores[target_col], preds, thresholds=[0.3])
        print_train_test_metrics(metrics_dict)


if __name__ == "__main__":  
    args = parse_args()

    predict(
        args.input_file,
        args.il_scores,
        args.deeppeptide_probas,
        args.mhcnuggets,
        args.model,
        args.output_file,
        args.target_col,
    )
