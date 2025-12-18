import os
from collections import defaultdict
import re
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, matthews_corrcoef
from sklearn.utils import resample
from sklearn.model_selection import StratifiedGroupKFold
import networkx as nx


CYTOKINE_TARGET_MAP = {
    'il2': 'IL-2 release',
    'il4': 'IL-4 release',
    'il10': 'IL-10 release',
    'ifng': 'IFNg release'
}


def find_best_threshold(y_true: list[float | int], y_scores: list[float]) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    return best_threshold


def calculate_metrics(y_true, y_pred, y_scores):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def partial_auc(y_true, y_scores, max_fpr=0.1):
    """
    Calculate the partial AUC up to a specified FPR threshold.

    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Target scores, can either be probability estimates or confidence values.
    max_fpr (float): Maximum False Positive Rate up to which the AUC is calculated.

    Returns:
    float: Partial AUC value.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Clip FPR and TPR at max_fpr
    if max_fpr < 1.0:
        stop = np.searchsorted(fpr, max_fpr, side="right")
        fpr = np.concatenate(([0], fpr[:stop], [max_fpr]))
        tpr = np.concatenate(([0], tpr[:stop], [tpr[stop - 1]]))

    partial_auc_value = auc(fpr, tpr) / max_fpr  # Normalize by max_fpr
    return partial_auc_value


def collect_predictions(model, dataloader, device):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            output = model(x)

            scores = torch.sigmoid(output.logits)
            all_scores.extend(scores.squeeze().cpu().numpy().tolist())

    all_scores = np.array(all_scores)
    return all_scores


class MHCProbabilityEstimator:
    def __init__(self, config, cytokine: str, target_name: str):
        self.config = config
        self.cytokine = cytokine
        self.target_column = target_name

        data_root = self.config.get("paths", "cytokine")
        self.full_dataset_path = f"{data_root}external/tcell_full_v3_processed.csv"
        self.il_splits_path = self.config.get("paths", "il_splits")

    def load_mhc_subset(self) -> pd.DataFrame:
        """Load and filter a subset with MHC II and human host."""
        full_df = pd.read_csv(self.full_dataset_path)
        filtered_df = full_df[
            (full_df['Response measured'] == self.target_column) &
            (full_df['MHC Class'] == 'II') &
            (full_df['Host'] == 'Homo sapiens (human)')
        ][['MHC Allele', 'Epitope ID']].drop_duplicates()
        return filtered_df

    def load_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Загружает обучающую и тестовую выборки."""
        train_path = f"{self.il_splits_path}{self.cytokine}_train.csv"
        test_path = f"{self.il_splits_path}{self.cytokine}_test.csv"

        train_df = pd.read_csv(train_path)[['Epitope Seq', 'Epitope ID', self.target_column]]
        test_df = pd.read_csv(test_path)[['Epitope Seq', 'Epitope ID', self.target_column]]
        return train_df, test_df

    def compute_probabilities(self) -> dict:
        """Основной метод для вычисления вероятностей на основе MHC-аллелей."""
        mhc_subset_df = self.load_mhc_subset()
        train_df, test_df = self.load_data_splits()

        full_df = pd.concat([train_df, test_df])
        merged_df = full_df.merge(mhc_subset_df, on='Epitope ID').dropna(subset=['MHC Allele'])

        assert merged_df['Epitope ID'].nunique() == full_df['Epitope ID'].nunique(), \
            "После объединения с MHC некоторые эпитопы были утеряны."

        train_with_mhc = merged_df[merged_df['Epitope ID'].isin(train_df['Epitope ID'])]
        test_with_mhc = merged_df[merged_df['Epitope ID'].isin(test_df['Epitope ID'])]

        allele_positive_ratios = train_with_mhc.groupby('MHC Allele')[self.target_column].mean().to_dict()
        epitope_to_alleles = test_with_mhc.groupby('Epitope ID')['MHC Allele'].unique().to_dict()

        epitope_probabilities = {}
        for epitope_id, alleles in epitope_to_alleles.items():
            ratios = [allele_positive_ratios[mhc] for mhc in alleles if mhc in allele_positive_ratios]

            if len(ratios) == 1 and ratios[0] in (0.0, 1.0):
                epitope_probabilities[epitope_id] = ratios[0]

        return epitope_probabilities


def load_dict_from_hdf5(filename):
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


def embeddings_dict_to_df(embeddings_dict):
    data = []
    for key, embedding in embeddings_dict.items():
        epitope_id_str, start_str, end_str = key.split('_')
        epitope_id = int(epitope_id_str)
        start = int(start_str) + 1
        end = int(end_str)
        data.append((epitope_id, start, end, embedding))
    return pd.DataFrame(data, columns=['Epitope ID', 'Starting Position', 'Ending Position', 'Embedding'])


def expand_embeddings(df: pd.DataFrame, epitope_col='Epitope ID', embedding_col='Embedding') -> pd.DataFrame:
    embedding_matrix = np.stack(df[embedding_col].values)
    component_columns = [f'Component_{i}' for i in range(embedding_matrix.shape[1])]
    embedding_df = pd.DataFrame(embedding_matrix, columns=component_columns)
    return pd.concat([df[[epitope_col]], embedding_df], axis=1)


def load_embeddings_none_human(embeddings_dir, target_name, model_name, cytokine):
    if model_name == 'ankh':
        prefix_name = 'parented_none_human_epitopes'  # parented_epitopes
        embeddings = load_dict_from_hdf5(embeddings_dir + f'{model_name}_{prefix_name}_{target_name}.hdf5')
    elif model_name == 'esmc':
        prefix_name = 'none_human'  # ''
        embeddings = torch.load(embeddings_dir + f'{model_name}_{prefix_name}{cytokine}.pt')
        embeddings = {key: embeddings[key].numpy() for key in embeddings}
    return embeddings


def load_none_human_data(config, target_name):
    cytokine_dir = config.get("paths", "cytokine")
    parent_epitopes_dir = config.get("paths", "parent_epitopes")

    tcell_full_v3_processed_extended = pd.read_csv(cytokine_dir + 'external/tcell_full_v3_processed_extended.csv')
    none_human_epitopes = pd.read_csv(parent_epitopes_dir + f'none_human_epitopes{target_name}.csv')

    target_df = tcell_full_v3_processed_extended[tcell_full_v3_processed_extended['Response measured'] == target_name]

    none_human_epitopes_output = none_human_epitopes.merge(target_df[['Epitope ID', 'Host', 'Label']],
                                                           on=['Epitope ID']).drop_duplicates(subset=['Epitope ID', 'Starting Position'])

    none_human_epitopes_output.rename(columns={'Label': target_name}, inplace=True)

    assert len(none_human_epitopes_output) == len(none_human_epitopes)
    return none_human_epitopes_output


def prepare_none_human_data(config, target_name, model_name, cytokine):
    none_human_df = load_none_human_data(config, target_name)
    embeddings_dir = config.get("paths", "embeddings")

    embeddings = load_embeddings_none_human(embeddings_dir, target_name, model_name, cytokine)
    epitopes_df = embeddings_dict_to_df(embeddings)
    embeddings_df_expanded = expand_embeddings(epitopes_df)

    train_embed_df = none_human_df[[target_name, 'Epitope ID']].merge(embeddings_df_expanded, on='Epitope ID')
    assert train_embed_df['Epitope ID'].nunique() == none_human_df['Epitope ID'].nunique()

    train_lama_df = train_embed_df.drop(columns='Epitope ID')

    if model_name == 'esmc':
        hid_dim = 1152
    elif model_name == 'ankh':
        hid_dim = 1536

    assert train_lama_df.shape[1] == hid_dim + 1

    return train_lama_df


def load_benchmarks_df(benchmarks: str, cytokine: str) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if cytokine == 'IL-2 release':
        train = pd.read_csv(os.path.join(benchmarks, "train_main_IL2.csv")).sample(frac=1, random_state=42)
        test = pd.read_csv(os.path.join(benchmarks, "test_main_IL2.csv"))
        return train, test
    elif cytokine == 'IL-4 release':
        return pd.read_csv(os.path.join(benchmarks, "filtered_IL4.csv"),
                           usecols=['Epitope ID', 'Epitope Seq', 'IL-4 release'])
    elif cytokine == 'IL-10 release':
        return pd.read_csv(os.path.join(benchmarks, "train_IL10.csv"),
                           usecols=['Epitope ID', 'Epitope Seq', 'IL-10 release'])
    else:
        raise ValueError(f"Unknown cytokine: {cytokine}")


def load_embeddings(embeddings_dir, model_name, prefix_path_embed, cytokine):
    if model_name == 'ankh':
        embeddings = load_dict_from_hdf5(embeddings_dir + f'{model_name}_{prefix_path_embed}_{cytokine}.hdf5')
    elif model_name == 'esmc' or model_name == 'esm3v0':
        embeddings = torch.load(embeddings_dir + f'{model_name}_{prefix_path_embed}_{cytokine}.pt', weights_only=True)  # esm3v0_bench_il10.pt
        embeddings = {key: embeddings[key].numpy() for key in embeddings}
    return embeddings


class EpitopeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_name: str):
        self.embeddings = torch.tensor(df.drop(columns=target_name).values, dtype=torch.float32)
        self.labels = torch.tensor(df[target_name].values, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class EpitopeDataset2D(Dataset):
    def __init__(self, df: pd.DataFrame, target_name: str):
        self.embeddings = df['Embedding']
        self.labels = df[target_name]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return emb, label


def custom_collate_fn(batch):
    # Extract the embeddings from the batch
    embeddings = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)

    # Pad the embeddings
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
    return padded_embeddings, labels


def log_dataset_statistics(logger, df: pd.DataFrame, target_name: str = 'Label', title: str = 'Train'):
    total = len(df)
    pos = df[target_name].sum()
    neg = total - pos

    logger.report_text(f"""📊 {title} Dataset Statistics:
- Total samples: {total}
- Positive: {pos}
- Negative: {neg}
""")


def check_counts(df: pd.DataFrame, df_embedded: pd.DataFrame, target_name: str = 'Label'):
    counts_1 = df[target_name].value_counts()
    counts_2 = df_embedded[target_name].value_counts()

    diff_ones = counts_2[1] - counts_1[1]
    diff_zeros = counts_2[0] - counts_1[0]

    if diff_ones != 0 or diff_zeros != 0:
        raise ValueError(f"The difference in counts for class 0 is {diff_zeros} and for class 1 is {diff_ones}.")


def balance_classes(df: pd.DataFrame, target_col: str, method: str = 'oversample') -> pd.DataFrame:
    """
    Балансирует классы в датафрейме.

    Параметры:
    - df: pd.DataFrame — входной датафрейм
    - target_col: str — имя колонки с целевой переменной
    - method: str — 'oversample' (по умолчанию) или 'undersample'

    Возвращает:
    - сбалансированный DataFrame
    """
    # Разделим по классам
    classes = df[target_col].unique()
    dfs = [df[df[target_col] == c] for c in classes]

    if method == 'oversample':
        max_len = max(len(subdf) for subdf in dfs)
        balanced_dfs = [
            resample(subdf, replace=True, n_samples=max_len, random_state=42)
            for subdf in dfs
        ]
    elif method == 'undersample':
        min_len = min(len(subdf) for subdf in dfs)
        balanced_dfs = [
            resample(subdf, replace=False, n_samples=min_len, random_state=42)
            for subdf in dfs
        ]
    else:
        raise ValueError("method должен быть 'oversample' или 'undersample'")

    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)


def create_embed_2d_df(df, config, cytokine=None, model_name=None,
                       prefix_path_embed=None, delete_duplicates=True,
                       seq_to_id_mapper=None):

    embeddings_dir = config.get('paths', 'embeddings')
    embeddings = load_embeddings(embeddings_dir, model_name, prefix_path_embed, cytokine)
    embeddings_df = embeddings_dict_to_df(embeddings)

    target_name = CYTOKINE_TARGET_MAP[cytokine]

    if seq_to_id_mapper:
        df['Epitope ID'] = df['Epitope Seq'].apply(lambda x: seq_to_id_mapper[x])

    embed_df_labeled = df[[target_name, 'Epitope ID']].merge(embeddings_df, on='Epitope ID')

    if delete_duplicates:
        embed_df_labeled = embed_df_labeled.drop_duplicates(subset='Epitope ID')

    assert embed_df_labeled['Epitope ID'].nunique() == df['Epitope ID'].nunique()

    embed_df_labeled = embed_df_labeled.drop(columns=['Epitope ID', 'Starting Position', 'Ending Position'])
    return embed_df_labeled


def create_groups(df):
    """
    Создает группы на основе связанных компонент графа.
    Строки связаны, если у них совпадает хотя бы один из идентификаторов.
    """
    # Создаем граф для поиска связанных компонент
    G = nx.Graph()

    # Добавляем все индексы как узлы
    G.add_nodes_from(df.index)

    # Словари для быстрого поиска индексов по значениям
    epitope_to_idx = defaultdict(list)
    ncbi_to_idx = defaultdict(list)
    uniprot_to_idx = defaultdict(list)

    for idx, row in df.iterrows():
        epitope_id = row['Epitope ID']
        ncbi_id = row['NCBI ID']
        uniprot_id = row['Uniprot ID']

        # Добавляем индексы в соответствующие словари
        epitope_to_idx[epitope_id].append(idx)

        if pd.notna(ncbi_id):
            ncbi_to_idx[ncbi_id].append(idx)
        elif pd.notna(uniprot_id):
            uniprot_to_idx[uniprot_id].append(idx)

    # Создаем ребра между строками с одинаковыми идентификаторами
    for indices in epitope_to_idx.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    G.add_edge(indices[i], indices[j])

    for indices in ncbi_to_idx.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    G.add_edge(indices[i], indices[j])

    for indices in uniprot_to_idx.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    G.add_edge(indices[i], indices[j])

    # Находим связанные компоненты - это и будут наши группы
    connected_components = list(nx.connected_components(G))

    # Создаем массив групп
    groups = np.zeros(len(df), dtype=int)
    for group_id, component in enumerate(connected_components):
        for idx in component:
            groups[idx] = group_id

    return groups


def stratified_group_k_fold_split(df, k=5, target_name='IL-2 release',
                                  group_col='Epitope ID', random_state=42):
    """
    Perform stratified group k-fold split on a DataFrame.
    df has columns: IL-X release, Epitope ID, Component_i

    Parameters:
        df (pd.DataFrame): The input dataframe.
        k (int): Number of folds.
        target_col (str): Name of the target column for stratification.
        group_col (str): Name of the group column to keep grouped.

    Returns:
        train_dfs (List[pd.DataFrame]): List of training DataFrames.
        test_dfs (List[pd.DataFrame]): List of testing DataFrames.
    """
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    train_dfs, test_dfs = [], []

    X = df.drop(columns=[target_name])
    y = df[target_name]
    if isinstance(group_col, (list, tuple)):
        groups = create_groups(df)

    elif isinstance(group_col, str):
        groups = df[group_col]

    test_indices = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        test_indices.append(test_idx)
        train_dfs.append(df.iloc[train_idx].reset_index(drop=True))
        test_dfs.append(df.iloc[test_idx].reset_index(drop=True))

    return train_dfs, test_dfs


def normalize_mhc(allele_str: str) -> str:
    """
    Преобразует запись аллелей MHC в требуемый формат:
    - HLA-DRB1*15:01  → DRB1_1501
    - HLA-DPA1*01:03/DPB1*04:01 → HLA-DPA10103-DPB10401
    """
    if pd.isna(allele_str):
        return allele_str

    # Убираем пробелы
    allele_str = allele_str.strip()

    # Если две аллели через '/'
    if '/' in allele_str:
        parts = allele_str.split('/')
        formatted_parts = []
        for part in parts:
            # Убираем HLA- только в первой, если вторая без префикса
            part = re.sub(r'^HLA-', '', part)
            part = re.sub(r'\*', '', part)
            part = part.replace(':', '')
            formatted_parts.append(part)
        return 'HLA-' + '-'.join(formatted_parts)

    # Если одна аллель
    match = re.match(r'^HLA-(\w+)\*(\d+):(\d+)$', allele_str)
    if match:
        gene, a1, a2 = match.groups()
        return f"{gene}_{a1}{a2}"
    else:
        # fallback на общий случай
        allele_str = allele_str.replace(':', '').replace('*', '')
        if allele_str.startswith('HLA-'):
            allele_str = allele_str[4:]
        return allele_str
