from argparse import ArgumentParser

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
        help="CSV file with input proteins and train/test split columns",
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

    # output files
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        help="PKL to save trained models, metrics and test predictions to"
    )

    # model training parameters
    parser.add_argument(
        "--score_names",
        type=str,
        default=None,
        help="TXT with a list of score names to use for prediction",
    )
    parser.add_argument(
        "--smote",
        type=str,
        default=None,
        help="Specify `cv` or `train` to add SMOTE data to each CV train fold or for the whole train"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="auc",
        help="Metric to use for LAML training",
    )
    return parser.parse_args()


def main_train(
    input_file,
    il_scores_file,
    deeppeptide_file,
    mhcnuggets_file,
    output_file,
    score_names=None,
    smote_type=None,
    metric="auc",
):
    """
    Run model training and save results.
    
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
    output_file: str
        JOBLIB file to save a dictionary of trained model, metrics, test predictions 
        and input features to.
    score_names: str, optional
        TXT file with a list of score names to include in the model. If None, all scores
        will be used Default: None.
    smote_type: {"cv", "train", None}, optional.
        What type of SMOTE dataset oversampling to use. If "cv", for each CV fold train
        will be augmented independently, test will remain unchanged. If "train",
        the whole train dataset will be oversampled, then CV splits will be formed. 
        If None, no oversampling will be applied. Default: None.
    metric: str, optional.
        Metric to use for model selection during CV training. If not "auc", 
        `eval(metric)` function will be used. Default: "auc".
    """

    # load scores
    mhc_data = pd.read_csv(input_file, index_col=0)
    il_scores = pd.read_csv(il_scores_file)
    dp_probas = pd.read_pickle(deeppeptide_file)
    mhcnuggets_scores = pd.read_csv(mhcnuggets_file)
    
    # load a list of score names to use
    scores_columns = None
    if score_names is not None:
        with open(score_names) as f:
            scores_columns = list(map(lambda x: x.rstrip(), f.readlines()))
    
    # create a single dataset
    mhc_scores, scores_columns, aap_dipeptide_scores = load_scores(
        mhc_data, 
        mhcnuggets_scores=mhcnuggets_scores,
        il_scores=il_scores,
        dp_probas=dp_probas,
        scores_columns=scores_columns
    )
    train_data = mhc_scores[mhc_scores["split"] == "train"]
    test_data = mhc_scores[mhc_scores["split"] == "test"]
        
    # generate CV splits
    cv_splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
        train_data, train_data["immuno_target"], train_data["prot_id"],
    ))
    train_to_save = train_data.copy()
    train_data = train_data[scores_columns + ["immuno_target"]]

    if smote_type == "cv":
        # add SMOTE to each train but not test part of the CV split
        smote = SMOTE(random_state=RANDOM_STATE)
        
        synthetic_cv_splits = []
        for train_idx, test_idx in cv_splits:
            # Apply SMOTE to fold train
            fold_train = train_data.iloc[train_idx]
            n_original = fold_train.shape[0]

            # Convert np.array back to DataFrame
            fold_X_smote, fold_y_smote = smote.fit_resample(fold_train[scores_columns], fold_train["immuno_target"])
            fold_X_smote = pd.DataFrame(fold_X_smote, columns=scores_columns)
            fold_y_smote = pd.Series(fold_y_smote, name="immuno_target", dtype=fold_y_smote.dtype)
            fold_train = pd.concat([fold_X_smote, fold_y_smote], axis=1)

            # add indices of synthetic data to CV indices
            synthetic_data = fold_train.iloc[n_original:]
            n_synthetic = synthetic_data.shape[0]
            synthetic_train_idx = list(range(train_data.shape[0], train_data.shape[0] + n_synthetic))
            train_data = pd.concat([train_data, synthetic_data], axis=0)

            synthetic_cv_splits.append((np.concatenate((train_idx, synthetic_train_idx)), test_idx))
            # print(n_original, n_synthetic, len(synthetic_cv_splits[-1][0]), train_data.shape)
            # print(train_data["immuno_target"].isna().sum())
        cv_splits = synthetic_cv_splits
    elif smote_type == "train":
        # apply SMOTE to train 
        smote = SMOTE(random_state=RANDOM_STATE)
        train_X_smote, train_y_smote = smote.fit_resample(train_data[scores_columns], train_data["immuno_target"])

        cv_splits = list(
            StratifiedKFold(shuffle=True, random_state=RANDOM_STATE).split(
                train_data, train_data["immuno_target"],
            ),
        )
    
    # if smote_type != "cv":
    #     for i, (train_idx, test_idx) in enumerate(cv_splits):
    #         train_to_save.loc[train_to_save.iloc[train_idx].index, f"fold_{i}"] = "train"
    #         train_to_save.loc[train_to_save.iloc[test_idx].index, f"fold_{i}"] = "test"
    #     train_to_save.to_csv("actual_train.csv", index=None)
    
    # train model
    if metric != "auc":
        metric = eval(metric)
    task = Task(name='binary', metric=metric, greater_is_better=True)

    automl = TabularAutoML(
        task=task,
        gpu_ids="0,1",
        general_params={
            "weighted_blender_max_nonzero_coef": 0.0,
            "use_algos": [["lgb_tuned", "lgb", "linear_l2", "cb"]],
        }, 
    )

    train_preds = automl.fit_predict(
        train_data, 
        roles={"target": "immuno_target"},
        cv_iter=cv_splits,
        verbose=3,
    ).data[:, 0] # returns nans for CV-SMOTE data splits

    # re-predict on train to obtain predictions on all data, not just non-synthetic
    train_preds = automl.predict(train_data[scores_columns])
    train_preds = to_npy_array(train_preds)[:, 0]
    test_preds = to_npy_array(automl.predict(test_data[scores_columns]))[:, 0]
    

    # calculate and print metrics
    train_metrics = calculate_metrics(train_data["immuno_target"], train_preds, thresholds=[0.3, 0.5, 0.7])
    test_metrics = calculate_metrics(test_data["immuno_target"], test_preds, thresholds=[0.3, 0.5, 0.7])
    print("Train / Test")
    print_train_test_metrics(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )
    
    # save trained model, metrics, predictions and input features
    train_metrics["split"] = "train"
    test_metrics["split"] = "test"
    metrics_dict = pd.DataFrame([train_metrics, test_metrics])

    test_data["laml_preds"] = test_preds
    save_training_results(
        output_file, 
        automl, 
        metrics_dict, 
        scores_columns, 
        test_preds=test_data,
        aap_dipeptide_scores=aap_dipeptide_scores,
    )


if __name__ == "__main__":  
    args = parse_args()

    main_train(
        args.input_file,
        args.il_scores,
        args.deeppeptide_probas,
        args.mhcnuggets,
        args.output_file,
        args.score_names,
        args.smote,
        args.metric,
    )
