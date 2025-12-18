import pickle
import re
from collections import defaultdict, Counter
import itertools
import math

import h5py
from Bio import SeqIO
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, f1_score
from imblearn.over_sampling import ADASYN


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def load_dict_from_hdf5(filename):
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


def parse_fasta_to_dict(file_path: str) -> dict[str, str]:
    fasta_dict = {}
    # Parse the FASTA file and print details
    for record in SeqIO.parse(file_path, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


class DataProcessor:
    def __init__(self, feature_importance_df=None, target_column='IL-4 release', variance_threshold=0.02,
                 corr_threshold=0.9, importance_threshold=0.0011):
        """
        Класс для предобработки данных:
        - Нормализация
        - Удаление скоррелированных признаков (оставляя более важный)
        - Удаление маловажных признаков по feature_importance

        :param feature_importance_df: DataFrame с важностью признаков (столбцы: 'Feature', 'Importance')
        :param target_column: Название таргета
        :param variance_threshold: Порог для удаления низковариативных признаков
        :param corr_threshold: Порог корреляции для удаления (оставляется более важный)
        :param importance_threshold: Минимальная важность признаков (фичи с меньшей удаляются)
        """
        self.target_column = target_column
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.selector = VarianceThreshold(threshold=variance_threshold)
        self.feature_importance_df = feature_importance_df
        self.feature_to_drop_ = None
        self.low_variance_features_ = None
        self.low_importance_features_ = None

    def fit_transform(self, df, do_reduce=False):
        """Обрабатывает тренировочные данные и запоминает удаленные признаки."""
        df_without_target = df.drop([self.target_column], axis=1)

        # Масштабируем данные
        df_scaled = df_without_target

        important_features = self.feature_importance_df[
            self.feature_importance_df['Importance'] >= self.importance_threshold
        ]['Feature'].tolist()

        df_scaled = df_scaled[important_features]
        self.low_importance_features_ = list(set(df_without_target.columns) - set(important_features))

        corr_matrix = df_scaled.corr(method="kendall")
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for column in upper.columns:
            if column in to_drop:
                continue
            correlated_features = [index for index in upper.index if upper.loc[index, column] > self.corr_threshold]

            # Если есть скоррелированные, выбираем самый важный
            if correlated_features:
                all_features = [column] + correlated_features
                sorted_features = sorted(all_features, key=lambda x: self._get_feature_importance(x), reverse=True)
                to_drop.update(sorted_features[1:])  # Удаляем все кроме самого важного

        self.feature_to_drop_ = list(to_drop)
        df_scaled = df_scaled.drop(columns=self.feature_to_drop_)

        if do_reduce:
            self.selector.fit(df_scaled)
            self.low_variance_features_ = df_scaled.columns[~self.selector.get_support()]
            df_scaled = df_scaled.drop(columns=self.low_variance_features_)
            df_scaled.columns = [f"feature_{i}" for i in range(len(df_scaled.columns))]

        df_scaled[self.target_column] = df[self.target_column].to_list()
        return df_scaled

    def transform(self, df, do_reduce=False):
        """Применяет сохраненные преобразования к тестовым данным."""
        df_without_target = df.drop([self.target_column], axis=1)

        # Масштабируем данные
        df_scaled = df_without_target

        # Удаляем признаки, которые удаляли в train
        df_scaled = df_scaled.drop(columns=self.low_importance_features_, errors='ignore')
        df_scaled = df_scaled.drop(columns=self.feature_to_drop_, errors='ignore')

        if do_reduce:
            df_scaled = df_scaled.drop(columns=self.low_variance_features_, errors='ignore')
            df_scaled.columns = [f"feature_{i}" for i in range(len(df_scaled.columns))]

        df_scaled[self.target_column] = df[self.target_column].to_list()
        return df_scaled

    def _get_feature_importance(self, feature):
        """Возвращает важность признака, если его нет в feature_importance_df, то 0."""
        importance_dict = dict(zip(self.feature_importance_df['Feature'], self.feature_importance_df['Importance']))
        return importance_dict.get(feature, 0)


def calculate_metrics(y_true, y_pred, y_scores):
    metrics = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_scores),
            "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics


def evaluate_predictions_with_external_labels(
    test_df,
    test_pos_indexes,
    test_neg_indexes,
    automl_model,
    target_column: str
) -> tuple[list[int], list[int], list[float]]:
    """
    Объединяет предсказания модели с внешне размеченными положительными и отрицательными выборками.

    Parameters:
    - test_df: pd.DataFrame — полный тестовый датафрейм
    - test_pos_indexes: list — индексы положительных образцов
    - test_neg_indexes: list — индексы отрицательных образцов
    - automl_model: модель LightAutoML с методом .predict()
    - target_column: str — имя целевой переменной

    Returns:
    - y_true: List[int] — истинные метки
    - predicted_classes: List[int] — предсказанные классы
    - y_scores: List[float] — вероятности положительного класса
    """

    # Проверка на отсутствие пересечений
    overlap = set(test_pos_indexes) & set(test_neg_indexes)
    assert not overlap, f"Overlap detected between positive and negative samples: {overlap}"

    # Убираем из выборки внешне размеченные образцы
    reduced_df = test_df[~test_df.index.isin(test_pos_indexes + test_neg_indexes)]

    # Предсказания модели
    y_true = reduced_df[target_column].tolist()
    predictions = automl_model.predict(reduced_df)
    pred_probs = predictions.data.flatten()
    y_scores = pred_probs.tolist()
    predicted_classes = (pred_probs > 0.5).astype(int).tolist()

    # Добавляем положительные метки с вероятностью 1.0
    pos_labels = test_df[test_df.index.isin(test_pos_indexes)][target_column].tolist()
    y_true.extend(pos_labels)
    y_scores.extend([1.0] * len(pos_labels))
    predicted_classes.extend([1] * len(pos_labels))

    # Добавляем отрицательные метки с вероятностью 0.0
    neg_labels = test_df[test_df.index.isin(test_neg_indexes)][target_column].tolist()
    y_true.extend(neg_labels)
    y_scores.extend([0.0] * len(neg_labels))
    predicted_classes.extend([0] * len(neg_labels))

    return y_true, predicted_classes, y_scores


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


def evaluate_model(automl, test_df, target_column):
    """
    Оценивает модель и визуализирует матрицу ошибок.

    Parameters:
    automl: Trained AutoML model
    test_df: DataFrame with features and target
    target_column: Name of the target column
    """
    y_true = test_df[target_column].copy()
    test_df_reduced = test_df.drop(columns=[target_column])
    test_predictions = automl.predict(test_df_reduced)
    test_predictions_flatten = test_predictions.data.flatten()

    predicted_classes = (test_predictions_flatten > 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, test_predictions_flatten)
    precision = precision_score(y_true, predicted_classes)
    recall = recall_score(y_true, predicted_classes)
    mcc = matthews_corrcoef(y_true, predicted_classes)

    print("ROC-AUC:", roc_auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Matthews Correlation Coefficient:", mcc)

    cm = confusion_matrix(y_true, predicted_classes)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=["Predicted 0", "Predicted 1"],
           yticklabels=["True 0", "True 1"],
           ylabel='Истинный класс',
           xlabel='Предсказанный класс',
           title='Матрица ошибок')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


def parse_motifs_from_file(filepath):
    motifs = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    motifs_started = False
    for line in lines:
        if motifs_started:
            if line.strip():  # пропускаем пустые строки
                motif = line.strip().replace(" ", "")
                motifs.append(motif)
        elif line.strip() == "Motifs:":
            motifs_started = True

    return motifs


def parse_merci_output(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    motif_stats = dict()
    current_motif = None
    current_seq = None

    for line in lines:
        # Новый мотив
        if line.startswith("MOTIF:"):
            current_motif = ''.join(line.strip().split()[1:])  # убираем пробелы между символами
            motif_stats[current_motif] = defaultdict(int)
        # ID последовательности
        elif line.startswith(">"):
            current_seq = line.strip()[1:]  # убираем >
        # Вхождения мотива
        elif line.strip().startswith("start position(s):") and current_motif and current_seq:
            positions = re.findall(r'\d+', line)
            motif_stats[current_motif][current_seq] += len(positions)

    return motif_stats


def motif_stats_to_dataframe(motif_stats):
    # Собираем уникальные sequence IDs и мотивы
    all_seqs = set()
    for motif_dict in motif_stats.values():
        all_seqs.update(motif_dict.keys())

    all_seqs = sorted(int(seq_id) for seq_id in all_seqs)  # превращаем в int и сортируем
    all_motifs = sorted(motif_stats.keys())

    # Заполняем таблицу
    data = []
    for seq_id in all_seqs:
        row = []
        for motif in all_motifs:
            count = motif_stats[motif].get(str(seq_id), 0)
            row.append(count)
        data.append([seq_id] + row)

    # Создаём DataFrame
    df = pd.DataFrame(data, columns=['Epitope ID'] + all_motifs)
    df.set_index('Epitope ID', inplace=True)

    return df


def features_dataset(df, fn, TARGET_NAME, **kwargs):
    data_dict = {}
    for sequence in df["Epitope Seq"]:
        features_seq = fn(sequence, **kwargs)
        data_dict[sequence] = features_seq
    data = pd.DataFrame.from_dict(data_dict, orient='index')
    data = data.merge(df[['Epitope Seq', TARGET_NAME]], left_index=True, right_on='Epitope Seq')
    assert len(data) == len(df)
    return data


def compute_features(df, fn, *args, **kwargs):
    data_dict = {}
    for epitope_id, sequence in zip(df["Epitope ID"], df["Epitope Seq"]):
        features_seq = fn(sequence, *args, **kwargs)
        data_dict[epitope_id] = features_seq
    data = pd.DataFrame.from_dict(data_dict, orient='index')
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Epitope ID'}, inplace=True)
    return data


def calculate_features(df, n=2, dpc_values=None, label_col=None):
    data_dict = {}
    for sequence in df["Epitope Seq"]:
        features_seq = compute_ngram(sequence, n=n)
        if dpc_values is not None:
            features_seq = {key: features_seq[key] for key in dpc_values}
        features_seq["length"] = len(sequence)
        data_dict[sequence] = features_seq
    data = pd.DataFrame.from_dict(data_dict, orient='index')
    data = data.merge(df[['Epitope Seq', label_col]], left_index=True, right_on='Epitope Seq')
    assert len(data) == len(df)
    return data


def custom_one_hot(df, column_name, prefix=None, train_categories=None):
    """
    One-hot кодирование с использованием pd.get_dummies.

    - В train сохраняется список категорий.
    - В test: категории из train добавляются, отсутствующие — заполняются нулями,
      новые — добавляются.
    - Значения в one-hot колонках — только int (0 или 1).

    :param df: DataFrame для кодирования
    :param column_name: имя категориальной переменной
    :param prefix: префикс для новых колонок (по умолчанию имя переменной)
    :param train_categories: список категорий из train (если None — значит train)
    :return: (df_encoded, categories)
    """
    if prefix is None:
        prefix = column_name

    # One-hot кодирование текущего df
    dummies = pd.get_dummies(df[column_name], prefix=prefix).astype(int)

    # Получаем список уникальных категорий в df
    current_categories = sorted(df[column_name].unique())

    if train_categories is None:
        # Тренировочный режим
        df_encoded = pd.concat([df.drop(columns=[column_name]), dummies], axis=1)
        return df_encoded, current_categories

    # Тестовый режим — собираем все категории
    all_categories = sorted(set(train_categories).union(current_categories))

    # Шаблон с нулями на все возможные категории
    dummies_full = pd.DataFrame(
        0, index=df.index,
        columns=[f"{prefix}_{cat}" for cat in all_categories]
    )

    # Копируем существующие данные из dummies
    for col in dummies.columns:
        dummies_full[col] = dummies[col]

    # Объединяем с df без исходной колонки
    df_encoded = pd.concat([df.drop(columns=[column_name]), dummies_full], axis=1)

    return df_encoded, all_categories


def balance_data(train_df, TARGET_NAME, class_balancer=ADASYN, random_state=42):
    """
    Балансирует обучающие данные с помощью ADASYN или SMOTE.
    """
    X_train, y_train = train_df.drop(columns=[TARGET_NAME]), train_df[TARGET_NAME]
    balancer = class_balancer(random_state=random_state)
    X_resampled, y_resampled = balancer.fit_resample(X_train, y_train)
    train_df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
    return train_df_balanced
