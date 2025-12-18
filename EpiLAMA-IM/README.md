# EpiLAMA-IM

EpiLAMA-IM is an immunogenicity prediction model.

## Environment
To run EpiLAMA-IM, create an environment using the following script:

```bash
source create_environment.sh
```

## Training

Scores for train and test datasets described in the paper are already pre-computed and stored in `train_data/`. To run training on another dataset, first generate the necessary scores:

```bash
python calculate_scores.py --input_file train_data/proliferationII_dataset_negs5.csv --output_dir train_data/
```

To train the best model described in the paper run:
```bash
python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_desc_il.txt \
    --metric average_precision_score \
    -o trained/laml_allele_desc_il.joblib
```

To run ablation experiments with all possible combinations of groups of scores:
```bash
source train_ap_combinations.sh
```

## Inference

To run a pre-trained model on another dataset, calculate external scores first (via `calculate_scores.py`), and then run inference using the following command:

```bash
python epilama_im_inference.py \
    -i train_data/proliferationII_dataset_negs5_test.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --model trained/laml_allele_desc_il.joblib \
    --target_col "immuno_target" \
    -o test_results.csv
```

The model reported in the paper should be downloaded and placed in the `trained/` folder first:

```bash
wget -P trained/ LINK/TO/MODEL/laml_allele_desc_il.joblib
```