import itertools
import math
from collections import Counter

import pandas as pd


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DIPEPTIDES = [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]


def compute_features(df, fn, *args, **kwargs):
    data_dict = {}
    for epitope_id, sequence in zip(df["Epitope ID"], df["Epitope Seq"]):
        features_seq = fn(sequence, *args, **kwargs)
        data_dict[epitope_id] = features_seq
    data = pd.DataFrame.from_dict(data_dict, orient='index')
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Epitope ID'}, inplace=True)
    return data


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


def build_aap_scores(pos_sequences, neg_sequences, epsilon=1e-6):
    """
    Строит AAP-веса на основе положительных и отрицательных последовательностей.
    Возвращает словарь с лог-оценками.
    """
    def get_dipeptide_freqs(sequences):
        total_counts = Counter()
        total_pairs = 0
        for seq in sequences:
            pairs = [seq[i:i+2] for i in range(len(seq) - 1)]
            total_counts.update(pairs)
            total_pairs += len(pairs)
        freqs = {dp: total_counts[dp] / total_pairs if total_pairs > 0 else 1e-6
                 for dp in DIPEPTIDES}
        return freqs

    pos_freqs = get_dipeptide_freqs(pos_sequences)
    neg_freqs = get_dipeptide_freqs(neg_sequences)

    # log-отношение частот
    aap_scores = {
        dp: math.log((pos_freqs[dp] + epsilon) / (neg_freqs[dp] + epsilon))
        for dp in DIPEPTIDES
    }

    return aap_scores


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