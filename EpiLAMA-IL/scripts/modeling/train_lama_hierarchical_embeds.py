import yaml
from pathlib import Path
from collections import defaultdict
import h5py

import pandas as pd

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task as LAMLTTask
from clearml import Task as ClearMLTask
from imblearn.over_sampling import ADASYN, SMOTE
from scripts.modeling.utils import calculate_metrics, expand_embeddings, log_dataset_statistics, check_counts, stratified_group_k_fold_split


def load_config(path: Path = 'scripts/modeling/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def filter_alleles(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['MHC Allele'].str.contains(r'\*\d{2}:\d{2}', na=False)
    return df[mask]


class EmbeddingsLoader:
    def __init__(self, path_embeddings: Path, multiple_embeddings: bool):
        self.path_embeddings = path_embeddings
        self.multiple_embeddings = multiple_embeddings

    @staticmethod
    def load_nested_embeddings_hdf5(path_embeddings: Path) -> dict:
        with h5py.File(path_embeddings, 'r') as h5f:
            embeddings = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            for source in h5f:
                for protein_id in h5f[source]:
                    for epitope_id in h5f[source][protein_id]:
                        for span in h5f[source][protein_id][epitope_id]:
                            start, end = map(int, span.split('_'))
                            embedding = h5f[source][protein_id][epitope_id][span][()]
                            embeddings[source][protein_id][epitope_id][(start, end)] = embedding
        return embeddings

    @staticmethod
    def embeddings_dict_to_df_hierarchical(embeddings_dict: dict) -> pd.DataFrame:
        """Converts hierarchical embeddings dict (with UNIPROT/NCBI) to flat dataframe"""
        data = []
        for source in embeddings_dict:  # 'UNIPROT', 'NCBI'
            for protein_id in embeddings_dict[source]:
                for epitope_id in embeddings_dict[source][protein_id]:
                    for (start, end), embedding in embeddings_dict[source][protein_id][epitope_id].items():
                        data.append({
                            'Source': source,
                            'Protein ID': protein_id,
                            'Epitope ID': int(epitope_id),
                            'Starting Position': start + 1,  # back to 1-based indexing
                            'Ending Position': end,
                            'Embedding': embedding
                        })
        return pd.DataFrame(data)

    @staticmethod
    def restructure_epitope_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Перестраивает DataFrame: перемещает 'Protein ID' в 'NCBI ID' или 'Uniprot ID'
        в зависимости от значения 'Source', а ненужное поле делает None.

        Parameters:
            df (pd.DataFrame): Исходный DataFrame с колонками:
                            ['Source', 'Protein ID', 'Epitope ID',
                                'Starting Position', 'Ending Position', 'Embedding']

        Returns:
            pd.DataFrame: Перестроенный DataFrame.
        """
        df = df.copy()
        df['Uniprot ID'] = None
        df['NCBI ID'] = None

        df.loc[df['Source'] == 'NCBI', 'NCBI ID'] = df.loc[df['Source'] == 'NCBI', 'Protein ID']
        df.loc[df['Source'] == 'Uniprot', 'Uniprot ID'] = df.loc[df['Source'] == 'Uniprot', 'Protein ID']

        df.drop(columns=['Protein ID', 'Source'], inplace=True)

        # Переставим столбцы для удобства
        column_order = ['Uniprot ID', 'NCBI ID', 'Epitope ID',
                        'Starting Position', 'Ending Position', 'Embedding']
        df = df[column_order]

        return df

    @staticmethod
    def deduplicate_df(df: pd.DataFrame) -> pd.DataFrame:
        mask = df['Uniprot ID'].notna()
        df_uniprot = df[mask].drop_duplicates(subset=['Epitope ID', 'Uniprot ID', 'Starting Position', 'Ending Position'])
        df_ncbi = df[~mask].drop_duplicates(subset=['Epitope ID', 'NCBI ID', 'Starting Position', 'Ending Position'])
        return pd.concat([df_uniprot, df_ncbi], ignore_index=True)

    def make_df(self) -> pd.DataFrame:
        if self.multiple_embeddings:
            directory = self.path_embeddings.parent
            stem = self.path_embeddings.stem
            files = [f for f in directory.glob(stem + "*") if f.is_file()]
            embeddings_dfs = []
            for file in files:
                embeddings_dict = self.load_nested_embeddings_hdf5(file)
                embeddings_df = self.embeddings_dict_to_df_hierarchical(embeddings_dict)
                embeddings_df = self.restructure_epitope_df(embeddings_df)
                embeddings_dfs.append(embeddings_df)
            embeddings_df = pd.concat(embeddings_dfs, ignore_index=True)
        else:
            embeddings_dict = self.load_nested_embeddings_hdf5(self.path_embeddings)
            embeddings_df = self.embeddings_dict_to_df_hierarchical(embeddings_dict)
            embeddings_df = self.restructure_epitope_df(embeddings_df)

        return self.deduplicate_df(embeddings_df)


def prepare_target_df(target_df, cytokine_name, host=None):  # если raw_data, то mhc_filter=True
    target_df = target_df[target_df['Response measured'] == cytokine_name]
    if host is not None:
        target_df = target_df[target_df['Host'] == host]
    return target_df


def check_ambigious_epitopes(df: pd.DataFrame, target_name: str = 'Label'):
    ambigious_cases = df.groupby('Epitope ID')[target_name].nunique() > 1
    assert ambigious_cases.sum() == 0, f'{ambigious_cases.sum()} ambigious cases'


class DataFormer:
    def __init__(self, parented_df: pd.DataFrame):
        self.uniprot_parented = parented_df[['Epitope ID', 'Uniprot ID', 'Starting Position', 'Ending Position']].dropna(subset=['Uniprot ID'])
        self.ncbi_parented = parented_df[['Epitope ID', 'NCBI ID', 'Starting Position', 'Ending Position']].dropna(subset=['NCBI ID'])

        assert self.uniprot_parented.duplicated().sum() == 0 and self.ncbi_parented.duplicated().sum() == 0

    def form(self,
             target_df: pd.DataFrame,
             clearml_logger=None,
             keys_uniprot: list[str] = ['Epitope ID', 'Uniprot ID', 'Starting Position', 'Ending Position'],
             keys_ncbi: list[str] = ['Epitope ID', 'NCBI ID', 'Starting Position', 'Ending Position']) -> pd.DataFrame:

        df_uniprot = target_df.drop(columns=['NCBI ID']).drop_duplicates()
        df_ncbi = target_df.drop(columns=['Uniprot ID']).drop_duplicates()

        df_uniprot = df_uniprot.drop_duplicates(subset=keys_uniprot)
        df_ncbi = df_ncbi.drop_duplicates(subset=keys_ncbi)

        df_uniprot = df_uniprot.merge(self.uniprot_parented, on=keys_uniprot, validate='one_to_one')
        df_ncbi = df_ncbi.merge(self.ncbi_parented, on=keys_ncbi, validate='one_to_one')

        merged_df = pd.concat([df_uniprot, df_ncbi], ignore_index=True)

        clearml_logger.report_text(f"Delta epitopes: {target_df["Epitope ID"].nunique() - merged_df["Epitope ID"].nunique()}")
        return merged_df

    @staticmethod
    def add_embeddings(df: pd.DataFrame,
                       embeddings_df: pd.DataFrame,
                       keys_uniprot: list[str] = ['Epitope ID', 'Uniprot ID', 'Starting Position', 'Ending Position'],
                       keys_ncbi: list[str] = ['Epitope ID', 'NCBI ID', 'Starting Position', 'Ending Position']) -> pd.DataFrame:

        df_uniprot = df.dropna(subset=['Uniprot ID']).drop_duplicates(subset=keys_uniprot).merge(
            embeddings_df.dropna(subset=['Uniprot ID']),
            on=keys_uniprot,
            validate='one_to_one'
        )

        df_ncbi = (df.dropna(subset=['NCBI ID']).drop_duplicates(subset=keys_ncbi).merge(
            embeddings_df.dropna(subset=['NCBI ID']),
            on=keys_ncbi,
            validate='one_to_one')
        )

        result_df = pd.concat([df_uniprot, df_ncbi], ignore_index=True)
        assert result_df['Epitope ID'].nunique() == df['Epitope ID'].nunique(), f'{result_df["Epitope ID"].nunique()} != {df["Epitope ID"].nunique()}'
        result_df = result_df[['Label', 'Embedding']]
        return result_df


def balance_data(train_df, target_name='Label', class_balancer=SMOTE, random_state=42):
    """
    Балансирует обучающие данные с помощью ADASYN или SMOTE.
    """
    X_train, y_train = train_df.drop(columns=[target_name]), train_df[target_name]
    balancer = class_balancer(random_state=random_state)
    X_resampled, y_resampled = balancer.fit_resample(X_train, y_train)
    train_df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
    return train_df_balanced


def train_model(train_df, target_name='Label', model_path=None, n_threads=8, n_folds=5, timeout=3600000, random_state=42):
    task = LAMLTTask('binary')
    roles = {'target': target_name}

    automl = TabularAutoML(
        task=task,
        timeout=timeout,
        cpu_limit=n_threads,
        reader_params={'n_jobs': n_threads, 'cv': n_folds, 'random_state': random_state},
    )

    if model_path:
        repo_root = Path(__file__).resolve().parents[2]
        models_dir = repo_root / 'data' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        path_to_save = str(models_dir / f'{model_path}.joblib')
    else:
        path_to_save = None
    automl.fit_predict(train_df, roles=roles, verbose=1, path_to_save=path_to_save)
    return automl


def evaluate_model(model, test_df, target_name='Label'):
    y_true = test_df[target_name]
    y_scores = model.predict(test_df).data.flatten()
    y_pred = (y_scores > 0.5).astype(int)
    return calculate_metrics(y_true, y_pred, y_scores)


def check_embeddings(df: pd.DataFrame, hid_dim: int = 1152):
    assert df.shape[1] == hid_dim + 1, f'The number of columns is {df.shape[1]}, but should be {hid_dim + 1}'


def cross_validate_model(target_df: pd.DataFrame,
                         embeddings_df: pd.DataFrame,
                         parented_df: pd.DataFrame,
                         k: int = 5,
                         target_name: str = 'Label',
                         group_col: str | list[str] = None,
                         subset_duplicates: list[str] | str = None,
                         clearml_logger=None) -> None:

    former = DataFormer(parented_df)
    target_df = former.form(target_df, clearml_logger=clearml_logger)

    # Разбиваем данные на фолды
    train_dfs, test_dfs = stratified_group_k_fold_split(target_df, k=k, target_name=target_name, group_col=group_col)

    print(f"Starting {k}-fold cross-validation...")
    fold_metrics = []
    for fold_idx in range(k):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold_idx + 1}/{k}")
        print(f"{'=' * 50}")

        train_df = train_dfs[fold_idx].copy()
        test_df = test_dfs[fold_idx].copy()

        train_df = train_df.drop_duplicates(subset=subset_duplicates)
        test_df = test_df.drop_duplicates(subset='Epitope ID')

        # Добавляем эмбеддинги
        train_df_embedded = former.add_embeddings(train_df, embeddings_df)
        test_df_embedded = former.add_embeddings(test_df, embeddings_df)

        # Разворачиваем эмбеддинги
        train_df_embedded = expand_embeddings(train_df_embedded, epitope_col='Label', embedding_col='Embedding')
        test_df_embedded = expand_embeddings(test_df_embedded, epitope_col='Label', embedding_col='Embedding')

        # Проверки
        check_embeddings(train_df_embedded)
        check_embeddings(test_df_embedded)

        check_counts(train_df, train_df_embedded)
        check_counts(test_df, test_df_embedded)

        log_dataset_statistics(clearml_logger, train_df_embedded, target_name='Label', title=f'Train_fold{fold_idx}')
        log_dataset_statistics(clearml_logger, test_df_embedded, target_name='Label', title=f'Test_fold{fold_idx}')

        if cfg['train_lama']['balance_method'] == 'smote':
            train_df_embedded = balance_data(train_df_embedded, target_name=target_name, class_balancer=SMOTE)
        elif cfg['train_lama']['balance_method'] == 'adasyn':
            train_df_embedded = balance_data(train_df_embedded, target_name=target_name, class_balancer=ADASYN)

        log_dataset_statistics(clearml_logger, train_df_embedded, target_name='Label', title=f'Train_fold{fold_idx} after {cfg["train_lama"]["balance_method"]}')

        model = train_model(train_df_embedded, model_path=task_name)
        metrics = evaluate_model(model, test_df_embedded)
        fold_metrics.append(metrics)

    metrics_df = pd.DataFrame(fold_metrics)
    summary_df = pd.DataFrame({
        'mean': metrics_df.mean(),
        'std': metrics_df.std(),
        'var': metrics_df.var()
    })

    clearml_logger.report_table(
        title='Fold metrics (per CV split)',
        series='metrics_df',
        iteration=0,
        table_plot=metrics_df,
    )

    clearml_logger.report_table(
        title='Aggregated metrics',
        series='summary_df',
        iteration=0,
        table_plot=summary_df,
    )


cfg = load_config()
task_name = f'{cfg["train_lama"]["response"]}_{cfg["train_lama"]["host"]}_' \
            f'{cfg["train_lama"]["balance_method"]}_{cfg["train_lama"]["variant"]}'  # if mouse .replace("/", "")

task_clearml = ClearMLTask.init(project_name=cfg['clearml']['project_name'],
                                task_name=task_name,
                                auto_connect_frameworks=False,
                                output_uri=True)

task_clearml.connect(cfg, name='experiment_config')

clearml_logger = task_clearml.get_logger()


target_df = pd.read_csv(cfg['data']['target_df_path'])
check_ambigious_epitopes(target_df)
parented_df = pd.read_csv(cfg['data']['parented_df_path'])
path_embeddings = Path(cfg['data']['embeddings_df_path'])
embeddings_df = EmbeddingsLoader(path_embeddings, cfg['data']['multiple_embeddings']).make_df()
if cfg['data']['prepare_target_df']:
    target_df = prepare_target_df(target_df,
                                  cytokine_name=cfg['train_lama']['response'],
                                  host=cfg['train_lama']['host'])

cross_validate_model(target_df,
                     embeddings_df,
                     parented_df,
                     group_col=cfg['train_lama']['group_col'],
                     subset_duplicates=cfg['train_lama']['subset_duplicates'],
                     clearml_logger=clearml_logger)
