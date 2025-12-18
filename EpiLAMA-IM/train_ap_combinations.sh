time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_desc.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_desc.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_desc.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_desc.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_desc.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_desc.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_desc_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_desc_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_desc_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_desc_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_desc.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_desc.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_desc_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_desc_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_desc_il.txt \
    --metric average_precision_score \
    -o trained/latest_laml_allele_desc_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_desc_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_desc_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_desc_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_desc_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_desc_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_desc_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_desc_dp.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_desc_dp.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_desc_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_desc_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_desc_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_desc_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_chem_desc_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_chem_desc_dp_il.joblib

time python epilama_im_train.py \
    -i train_data/proliferationII_dataset_negs5.csv \
    --mhcnuggets train_data/mhcnuggets_scores.csv \
    --il_scores train_data/il_scores.csv \
    --deeppeptide_probas train_data/deeppeptide_probas.pkl \
    --score_names train_data/scores_files/scores_allele_chem_desc_dp_il.txt \
    --metric average_precision_score \
    -o trained/combinations/laml_allele_chem_desc_dp_il.joblib

