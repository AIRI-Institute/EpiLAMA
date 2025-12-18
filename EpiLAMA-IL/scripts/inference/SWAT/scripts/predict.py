import argparse
from pathlib import Path
from collections import defaultdict, Counter
import itertools
import joblib
from typing import Any
from tqdm import tqdm
import yaml
import pandas as pd
import numpy as np
import torch
from SWAT.src.models.esmc import ESMCfloat32


LENGTH_LIMIT = 2048
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DIPEPTIDES = [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]


def compute_ngram(sequence: str, n: int = 2) -> dict:
    """
    Compute Dipeptide Composition (DPC) frequencies for a given peptide sequence.

    Parameters:
        sequence (str): The input peptide sequence consisting of 1-letter amino acid codes.

    Returns:
        dict: A dictionary of dipeptide frequencies (values normalized by total number of dipeptides).
    """
    ngrams = [''.join(dp) for dp in itertools.product(AMINO_ACIDS, repeat=n)]
    ngram_counts = dict.fromkeys(ngrams, 0)

    total_ngrams = len(sequence) - n + 1
    if total_ngrams <= 0:
        return dict.fromkeys(ngrams, 0.0)

    for i in range(total_ngrams):
        ngram = sequence[i:i+n]
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1

    # Normalize counts to get frequencies
    for ngram in ngram_counts:
        ngram_counts[ngram] /= total_ngrams

    return ngram_counts


def calculate_aac(sequence):
    """
    Calculate Amino Acid Composition (AAC) for a protein sequence.

    Parameters:
        sequence (str): Protein sequence using 20 standard amino acids.

    Returns:
        dict: Frequencies of each amino acid (keys are 'A', 'C', ..., 'Y').
    """
    sequence = sequence.upper()
    total_len = len(sequence)
    aa_counts = Counter(sequence)

    aac_vector = {aa: aa_counts.get(aa, 0) / total_len for aa in AMINO_ACIDS}
    return aac_vector


def calculate_aap(sequence, aap_scores):
    """
    Возвращает AAP-вектор длиной 400: значения лог-оценок дипептидов из последовательности.
    Если дипептид не встречается, значение = 0.

    Parameters:
        sequence (str): Пептидная последовательность
        aap_scores (dict): Предварительно рассчитанные лог-оценки дипептидов

    Returns:
        dict: Словарь из 400 дипептидов с их значениями AAP (0 если не встречаются)
    """
    # Получаем дипептиды из последовательности
    pairs = [sequence[i:i+2] for i in range(len(sequence) - 1)]

    # Считаем частоты дипептидов в данной последовательности
    pair_counts = Counter(pairs)
    total_pairs = len(pairs)

    # Строим AAP-вектор: частота дипептида * log-оценка
    aap_vector = {
        dp: (pair_counts.get(dp, 0) / total_pairs) * aap_scores.get(dp, 0.0)
        if total_pairs > 0 else 0.0
        for dp in DIPEPTIDES
    }

    return aap_vector


def compute_features(df, fn, *args, **kwargs):
    data_dict = {}
    for idx, sequence in enumerate(df["Epitope Seq"]):
        features_seq = fn(sequence, *args, **kwargs)
        data_dict[idx] = features_seq
    return pd.DataFrame.from_dict(data_dict, orient='index')



def correct_epitopes_df(df: pd.DataFrame, length_limit: int = LENGTH_LIMIT, start_index: int = 0) -> pd.DataFrame:
    """
    Adjusts the epitope positions and sequences in the DataFrame to ensure they fit within a specified length limit.
    Ending Posistion doesn't include the last residue.

    Args:
        df (pd.DataFrame): The input DataFrame containing protein sequences and their start and end positions.
        length_limit (int, optional): The maximum allowed length for the protein sequences. Defaults to LENGTH_LIMIT.
        start_index (int, optional): The index to subtract from the starting and ending positions. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame with adjusted protein sequences and positions.
        dict: A dictionary mapping new peptide start/end to the originally provided.
    """
    large_proteins_index_map = dict()
    def process_row(row):
        seq = row['Protein Seq']
        start = row['Starting Position'] - start_index
        end = row['Ending Position'] - start_index

        if len(seq) > length_limit:
            epi_len = end - start
            context = length_limit - epi_len
            left = max(0, start - context // 2)
            right = min(len(seq), end + context // 2)
            new_start = start - left
            new_end = new_start + epi_len
            seq = seq[left:right]

            # write old indices to reconstruct them later
            epi_seq = seq[new_start:new_end]
            large_proteins_index_map[(row['Protein ID'], new_start, new_end, epi_seq)] = (start, end)
        else:
            new_start = start
            new_end = end

        row['Protein Seq'] = seq
        row['Starting Position'] = new_start
        row['Ending Position'] = new_end
        row['ID'] = f"{row['Protein ID']}|{row['Starting Position']}|{row['Ending Position']}"
        return row

    corrected_df = df.apply(process_row, axis=1)
    return corrected_df, large_proteins_index_map


def extract_epitope_seq(df: pd.DataFrame) -> pd.DataFrame:
    df['Epitope Seq'] = df.apply(lambda row: row['Protein Seq'][row['Starting Position']:row['Ending Position']], axis=1)
    return df


def rename_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.rename(columns={col: f'{prefix}_{col}' for col in df.columns})

class DataLoader:
    """
    Data loader for reading a FASTA file and creating batches based on a token limit.

    Args:
    - df (pd.DataFrame): Dataframe with sequences and IDs.
    - model (object): Model object with a `_tokenize` method for tokenizing sequences.
    - batch_token_limit (int, optional): Maximum number of tokens per batch.
    """
    def __init__(self, df, model, batch_token_limit=LENGTH_LIMIT + 2):
        self.df = df
        self.batch_token_limit = batch_token_limit
        self.model = model
        self.sequences = df['Protein Seq']
        self.id = df['ID']
        self.total_sequences = len(self.df)

    def __len__(self):
        # Approximate total number of batches
        total_tokens = sum(len(seq) + 2 for seq in self.sequences)  # +2 for BOS and EOS tokens
        return (total_tokens + self.batch_token_limit - 1) // self.batch_token_limit

    def __iter__(self):
        ids, lengths, seqs = [], [], []
        current_token_count = 0

        for i, seq in enumerate(self.sequences):
            seq_length = len(seq)
            token_count = seq_length + 2  # Include BOS and EOS tokens
            if current_token_count + token_count > self.batch_token_limit and ids:
                # Yield current batch if adding the new sequence exceeds the token limit
                tokens = self.model._tokenize(seqs)
                yield ids, lengths, tokens, seqs
                ids, lengths, seqs = [], [], []
                current_token_count = 0

            # Add the current sequence to the batch
            ids.append(self.id.iloc[i])
            lengths.append(seq_length)
            seqs.append(seq)
            current_token_count += token_count

        # Yield any remaining sequences
        if ids:
            tokens = self.model._tokenize(seqs)
            yield ids, lengths, tokens, seqs


class ILClassifier:
    """Wrapper for IL cytokine prediction pipeline"""
    def __init__(self,
                 device: torch.device,
                 models_dir: str,
                 il2_human_models: list[str] | str,
                 il4_human_models: list[str] | str,
                 il10_human_models: list[str] | str,
                 ifng_human_models: list[str] | str,
                 esmc_model_name: str,
                 il2_mouse_model_path: str | None = None,
                 ifng_mouse_model_path: str | None = None,
                 ):
        self.device = device
        self.esmc_model = ESMCfloat32.from_pretrained(esmc_model_name, device=self.device)
        self.models_dir = Path(models_dir)
        self.il_models = {
            "il2_human": self._load_model_ensemble(ensemble_paths=il2_human_models),
            "il4_human": self._load_model_ensemble(ensemble_paths=il4_human_models),
            "il10_human": self._load_model_ensemble(ensemble_paths=il10_human_models),
            "ifng_human": self._load_model_ensemble(ensemble_paths=ifng_human_models),
            "il2_mouse": self._load_model_ensemble(),
            "ifng_mouse": self._load_model_ensemble(),
        }

    def _load_model_ensemble(self, single_path=None, ensemble_paths=None):
        model_paths = []
        if ensemble_paths:
            model_paths.extend(ensemble_paths)
        elif single_path:
            model_paths.append(single_path)

        loaded_models = []
        for path in model_paths:
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = self.models_dir / path_obj
            loaded_models.append(joblib.load(path_obj))

        return loaded_models

    def extract_representations(self, df: pd.DataFrame, mean: bool = True):
        self.esmc_model.eval()
        representations = {}
        data_loader = DataLoader(df, model=self.esmc_model)

        with torch.no_grad():
            for batch_ids, batch_lengths, batch_tokens, batch_seqs in tqdm(data_loader, desc="Processing batches", leave=False):
                output = self.esmc_model(batch_tokens.to(self.device))
                embeddings = output.embeddings.detach().cpu()

                for i, full_id in enumerate(batch_ids):
                    embedding = embeddings[i, 1:batch_lengths[i] + 1, :]
                    protein_id, start, end = full_id.split('|')
                    start = int(start)
                    end = int(end)
                    seq = batch_seqs[i]
                    embedding = embedding[start:end]
                    if mean:
                        embedding = embedding.mean(dim=0)
                    representations[(protein_id, start, end, seq)] = embedding

        return representations

    @staticmethod
    def embeddings_to_df(embeddings_dict: dict) -> pd.DataFrame:
        data = []
        for (protein_id, start, end, seq), embedding in embeddings_dict.items():
            data.append({
                'Protein ID': protein_id,
                'Starting Position': start,
                'Ending Position': end,
                'Embedding': embedding,
                'Protein Seq': seq
            })
        return pd.DataFrame(data)

    @staticmethod
    def expand_embeddings(df: pd.DataFrame, embedding_col='Embedding') -> pd.DataFrame:
        embedding_matrix = np.stack(df[embedding_col].values)
        component_columns = [f'Component_{i}' for i in range(embedding_matrix.shape[1])]
        return pd.DataFrame(embedding_matrix, columns=component_columns)


    @staticmethod
    def ensemble_predict(il_name: str,
                         features_df: pd.DataFrame,
                         models: list[Any]) -> pd.DataFrame:
        model_scores = []
        for model in models:
            scores = model.predict(features_df).data.flatten()
            model_scores.append(scores)
        mean_scores = np.mean(model_scores, axis=0)
        predictions = (mean_scores > 0.5).astype(int)
        return pd.DataFrame(
            {
            f'Score_{il_name}': mean_scores,
            f'Prediction_{il_name}': predictions
            }
        )

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        per_il_results = []
        for il_name, il_model in self.il_models.items():
            per_il_results.append(self.ensemble_predict(il_name, features_df, il_model))
        return pd.concat(per_il_results, axis=1)

    @staticmethod
    def save_output_df(embeddings_df: pd.DataFrame, predictions_df: pd.DataFrame, output_path: str, large_proteins_index_map: dict) -> None:
        output_df = pd.concat([embeddings_df.drop(columns=['Embedding']), predictions_df], axis=1)

        # Convert epitope start/end coordinates back to original values for large proteins that were cut
        for idx, row in output_df.iterrows():
            peptide_id = (row["Protein ID"], row["Starting Position"], row["Ending Position"], row["Epitope Seq"])
            if peptide_id in large_proteins_index_map:
                original_epi_start, original_epi_end = large_proteins_index_map[peptide_id]
                output_df.loc[idx, "Starting Position"] = original_epi_start
                output_df.loc[idx, "Ending Position"] = original_epi_end
        output_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Extracting ESMC representations from a csv file")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input csv file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output csv file")
    parser.add_argument("-c", "--config", type=str, default="config.yml", help="Path to YAML config file")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU only (overrides CUDA)",
    )
    device_group.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device index to use (e.g. 0). If not set and CUDA is available, uses 0.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(args.input)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda}")

    df, large_proteins_index_map = correct_epitopes_df(df)
    classifier = ILClassifier(device=device,
                              models_dir=config['models_dir'],
                              il2_human_models=config['il2_human_models'],
                              il4_human_models=config['il4_human_models'],
                              il10_human_models=config['il10_human_models'],
                              ifng_human_models=config['ifng_human_models'],
                              esmc_model_name=config['esmc_model_name'])

    embeddings_dict = classifier.extract_representations(df)
    embeddings_df = classifier.embeddings_to_df(embeddings_dict)
    embeddings_df = extract_epitope_seq(embeddings_df)
    features_df = classifier.expand_embeddings(embeddings_df)

    aac_features_df = compute_features(embeddings_df, calculate_aac)
    dpc_features_df = compute_features(embeddings_df, compute_ngram, n=2)
    aap_features_df = compute_features(embeddings_df, calculate_aap, aap_scores=config['aap_scores'])

    aac_features_df = rename_columns(aac_features_df, 'aac')
    dpc_features_df = rename_columns(dpc_features_df, 'dpc')
    aap_features_df = rename_columns(aap_features_df, 'aap')

    aac_features_df = pd.concat([features_df, aac_features_df], axis=1).reset_index(drop=True)
    dpc_features_df = pd.concat([features_df, dpc_features_df], axis=1).reset_index(drop=True)
    aap_features_df = pd.concat([features_df, aap_features_df], axis=1).reset_index(drop=True)

    assert len(aac_features_df.columns) == 1152 + 20
    assert len(dpc_features_df.columns) == 1152 + 400
    assert len(aap_features_df.columns) == 1152 + 400

    il2_human_predictions = classifier.ensemble_predict('il2_human', aap_features_df, classifier.il_models['il2_human'])
    il4_human_predictions = classifier.ensemble_predict('il4_human', aac_features_df, classifier.il_models['il4_human'])
    il10_human_predictions = classifier.ensemble_predict('il10_human', features_df, classifier.il_models['il10_human'])
    ifng_human_predictions = classifier.ensemble_predict('ifng_human', aac_features_df, classifier.il_models['ifng_human'])

    predictions_df = pd.concat([il2_human_predictions, il4_human_predictions, il10_human_predictions, ifng_human_predictions], axis=1)
    classifier.save_output_df(embeddings_df, predictions_df, args.output, large_proteins_index_map)


if __name__ == "__main__":
    main()
