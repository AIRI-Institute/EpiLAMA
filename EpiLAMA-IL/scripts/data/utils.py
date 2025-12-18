import configparser
import hashlib
import pandas as pd
import numpy as np
from sklearn.utils import resample
import h5py
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def balance_dataset(df, label_column='Label'):
    """
    Balances a binary classification dataset to a 1:1 ratio by undersampling the majority class.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the dataset.
    - label_column (str): The name of the column containing the binary class labels.

    Returns:
    - pd.DataFrame: A balanced dataframe with a 1:1 class ratio.
    """
    # Separate the majority and minority classes
    majority_class = df[df[label_column] == 0]  # Assuming class 0 is the majority class
    minority_class = df[df[label_column] == 1]  # Assuming class 1 is the minority class

    # Undersample the majority class to match the minority class size
    majority_class_undersampled = resample(majority_class,
                                           replace=False,    # Sample without replacement
                                           n_samples=len(minority_class),  # Match minority class size
                                           random_state=42)  # For reproducibility

    # Combine the undersampled majority class with the minority class
    df_balanced = pd.concat([majority_class_undersampled, minority_class])

    # Shuffle the dataset to ensure random distribution
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def create_id_mapper(df, column_name):
    """
    Create a mapping of unique strings in a specified column to unique IDs
    and replace the column values with their corresponding IDs.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: str, the name of the column to process.

    Returns:
    - df: Updated DataFrame with the column values replaced by IDs.
    - mapper: dict, mapping of unique strings to their corresponding IDs.
    """
    # Extract unique values and sort them
    unique_values = sorted(df[column_name].unique())

    # Create a dictionary mapping each unique value to an ID
    mapper = {value: idx for idx, value in enumerate(unique_values)}

    # Assign corresponding IDs
    df[column_name + " ID"] = df[column_name].map(mapper)

    return df, mapper


def read_embeddings_from_hdf5(filename):
    """
    Read embeddings from an HDF5 file and return them in a structured dictionary.

    Parameters:
    - filename: str, the path to the HDF5 file.

    Returns:
    - embeddings_dict: dict, a nested dictionary with keys as sequence types and
      unique IDs mapping to their corresponding embeddings.
    """
    embeddings_dict = {}

    with h5py.File(filename, "r") as f:
        for key in f.keys():  # Iterate over the top-level groups (e.g., "TCRa", "TCRb")
            embeddings_dict[key] = {}
            for uid in f[key].keys():  # Iterate over the datasets within each group
                embeddings_dict[key][uid] = f[key][uid][...]  # Read the dataset into a NumPy array

    return embeddings_dict


def load_dict_from_hdf5(filename):
    loaded_dict = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            loaded_dict[key] = f[key][:]
    return loaded_dict


def save_embeddings_one_level_keys_to_hdf5(embeddings_dict, filename, compression=True):
    with h5py.File(filename, "w") as f:
        for key, value in embeddings_dict.items():
            if compression:
                f.create_dataset(key, data=value, compression="gzip", compression_opts=9)
            else:
                f.create_dataset(key, data=value)


def save_embeddings_to_hdf5(embeddings_dict, filename):
    with h5py.File(filename, "w") as f:
        for key, value in embeddings_dict.items():
            for uid, embedding in value.items():
                # Create a dataset for each unique ID under the corresponding key
                dataset_name = f"{key}/{uid}"  # Create a hierarchical structure
                f.create_dataset(dataset_name, data=embedding)


def change_types(df, columns=["Epitope ID"]):
    for col in columns:
        df[col] = df[col].astype(str)
    return df


def save_df_to_fasta(df, fasta_filename, id_column="Epitope ID", sequence_column="Epitope Seq"):
    """
    Converts a DataFrame of epitopes to a list of SeqRecord objects and writes them to a FASTA file.

    Parameters:
    df (pd.DataFrame): DataFrame containing epitope sequences and IDs.
    fasta_filename (str): The name of the output FASTA file.
    """
    # Convert DataFrame to a list of SeqRecord objects
    df[id_column] = df[id_column].astype(str)
    records = [SeqRecord(Seq(row[sequence_column]), id=row[id_column], description="") for _, row in df.iterrows()]

    # Write to a FASTA file
    with open(fasta_filename, "w") as fasta_file:
        SeqIO.write(records, fasta_file, "fasta")


def filter_mhc_class(df, column_name='MHC Class', mhc_class='II'):
    return df[df[column_name] == mhc_class]


def filter_by_length(df, column_name='Epitope Seq', min_length=13, max_length=25):
    length = df[column_name].apply(len)
    return df[(length >= min_length) & (length <= max_length)]


def filter_host(df, host_column='host_organism_name', host='Homo sapiens (human)'):
    return df[df[host_column] == host]


def select_cytokine_class(df, column_name='IL-4 release'):
    df = df[df[column_name].notna()]
    return df


def filter_cytokine_class(df, column_name='Response measured', cytokine_class='IL-4 release'):
    df = df[df[column_name] == cytokine_class]
    return df


def filter_canonical_amino_acids(df: pd.DataFrame, column_name: str = "Epitope Seq") -> pd.DataFrame:
    def valid_sequence(sequence: str) -> bool:
        valid_amino_acids = "SNYLRQDPMFCEWGTKIVAH"
        return all(char in valid_amino_acids for char in sequence)

    df = df[df[column_name].apply(valid_sequence)]
    return df


def fasta_to_dataframe(fasta_path: str, id_column='Epitope ID', seq_column='Epitope Seq') -> pd.DataFrame:
    """
    Читает FASTA-файл и возвращает DataFrame с колонками: 'id', 'sequence'.

    Parameters:
        fasta_path (str): путь к FASTA-файлу

    Returns:
        pd.DataFrame: таблица с идентификаторами и последовательностями
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    data = [{id_column: record.id, seq_column: str(record.seq)} for record in records]
    return pd.DataFrame(data)


def parse_fasta_to_dict(file_path: str) -> dict[str, str]:
    fasta_dict = {}
    # Parse the FASTA file and print details
    for record in SeqIO.parse(file_path, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def deduplicate_df(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['Uniprot ID'].notna()
    uniprot_df = df[mask].drop_duplicates(subset=['Epitope ID', 'Uniprot ID', 'Starting Position', 'Ending Position'])
    ncbi_df = df[~mask].drop_duplicates(subset=['Epitope ID', 'NCBI ID', 'Starting Position', 'Ending Position'])
    return pd.concat([uniprot_df, ncbi_df], ignore_index=True)


def save_fasta_from_dict(seq_dict: dict, output_path: str):
    """
    Сохраняет последовательности в FASTA-файл из словаря {id: sequence}.

    Args:
        seq_dict (dict): Словарь, где ключ — ID (str), значение — аминокислотная или нуклеотидная последовательность (str).
        output_path (str): Путь к сохраняемому FASTA-файлу.
    """
    records = [
        SeqRecord(Seq(seq), id=str(seq_id), description="")
        for seq_id, seq in seq_dict.items()
    ]
    SeqIO.write(records, output_path, "fasta")


def init_benchmarks_data(target_name, prefix_path=None):
    config = configparser.ConfigParser()
    config.read("config.ini")
    parent_epitopes_dir = config.get("paths", "parent_epitopes")

    df = pd.read_csv(parent_epitopes_dir + f'{prefix_path}_{target_name}.csv')
    # Use repo-relative defaults for sequences
    uniprot_sequences = parse_fasta_to_dict('data/processed/uniprot_sequences.fasta')
    ncbi_sequences = parse_fasta_to_dict('data/processed/ncbi_sequences_v1.fasta')

    return df, uniprot_sequences, ncbi_sequences


def assign_protein_sequence(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict) -> pd.DataFrame:
    seq_from_uniprot = df['Uniprot ID'].map(uniprot_sequences)
    seq_from_ncbi = df['NCBI ID'].map(ncbi_sequences)
    df['Protein Seq'] = seq_from_uniprot.combine_first(seq_from_ncbi)
    return df


def preprocess_positions(df: pd.DataFrame) -> pd.DataFrame:
    df['Starting Position'] = df['Starting Position'] - 1  # корректируем в нулевую индексацию
    return df


def full_preprocessing_pipeline(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict) -> pd.DataFrame:
    df = assign_protein_sequence(df, uniprot_sequences, ncbi_sequences)
    df = preprocess_positions(df)
    return df


def shorten_protein_sequence(df: pd.DataFrame, length_limit: int = 1024) -> pd.DataFrame:
    updated_rows = []

    def replace_substring_with_updated_coords(sequence: str, start: int, end: int, replacement: str):
        new_sequence = sequence[:start] + replacement + sequence[end:]
        new_start = start
        new_end = start + len(replacement)
        return new_sequence, new_start, new_end

    for _, row in df.iterrows():
        sequence = row['Protein Seq']
        start = int(row['Starting Position'])
        end = int(row['Ending Position'])
        num_changes = row['Num changes']
        epitope_seq = row['Epitope Seq'].replace('-', '')

        # Проверяем, если длина последовательности больше лимита
        if len(sequence) > length_limit:
            length_epitope = end - start
            context = length_limit - length_epitope
            left = max(0, start - context // 2)
            right = min(len(sequence), end + context // 2)

            new_start = start - left
            new_end = new_start + length_epitope

            sequence = sequence[left:right]
        else:
            new_start = start
            new_end = end

        # Если есть изменения в эпитопе, вставляем новую последовательность
        if num_changes > 0:
            sequence, new_start, new_end = replace_substring_with_updated_coords(sequence, new_start, new_end, epitope_seq)

        # Обновляем строку
        updated_row = row.copy()
        assert sequence[new_start:new_end] == epitope_seq
        updated_row['Protein Seq'] = sequence
        updated_row['New Starting Position'] = new_start
        updated_row['New Ending Position'] = new_end

        updated_rows.append(updated_row)

    updated_df = pd.DataFrame(updated_rows)
    updated_df['ID'] = updated_df['Epitope ID'].astype(str) + '_' + updated_df['Starting Position'].astype(str) + '_' + updated_df['Ending Position'].astype(str)
    return updated_df


def load_data(file_path):
    return pd.read_csv(file_path)


def remove_duplicates(df, subset=None):
    if subset:
        return df.drop_duplicates(subset=subset)
    return df.drop_duplicates()


def transform_labels(df):
    # Get unique cytokines from the 'Response measured' column
    unique_responses = df['Response measured'].unique()

    # Create a column for each unique response
    for response in unique_responses:
        df[response] = np.where(df['Response measured'] == response, df['Label'], np.nan)

    # Drop the original columns
    df = df.drop(['Label'], axis=1)
    return df


def merge_cytokine_releases(df):
    """
    Merge cytokine release values for rows with the same Epitope ID by taking the max value for each column.

    Args:
    df (pd.DataFrame): Input dataframe containing cytokine release columns.

    Returns:
    pd.DataFrame: Processed dataframe with merged cytokine release values.
    """
    cytokine_cols = df['Response measured'].unique().tolist()
    # Use groupby to merge by Epitope ID, taking the maximum value per cytokine column
    merged_df = df.groupby(['Epitope ID'], as_index=False).agg({
        'Epitope Seq': 'first',
        **{col: 'max' for col in cytokine_cols}  # Take max for cytokine release columns
    })
    return merged_df


def remove_ambiguous_epitopes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove epitopes that have conflicting labels within the same host."""
    # Find all (Epitope ID, Response measured, Host) combinations with multiple unique Labels
    ambiguous_combinations = (
        df.groupby(['Epitope ID', 'Response measured'])['Label']
        .nunique()
        .gt(1)
    )

    # Convert index Series to set of tuples for fast lookup
    ambiguous_combinations = set(ambiguous_combinations[ambiguous_combinations].index)

    # Create mask: True if row belongs to an ambiguous combination
    ambiguous_mask = df.set_index(['Epitope ID', 'Response measured']).index.isin(ambiguous_combinations)

    # Remove all such rows
    cleaned_df = df.loc[~ambiguous_mask].copy()

    # Verification check
    label_counts = (
        cleaned_df.groupby(['Epitope ID', 'Response measured'])['Label']
        .nunique()
    )
    assert (label_counts == 1).all(), "Error in cleaning process"
    return cleaned_df


def get_mappers_epitopes(path_csv='data/external/tcell_full_v3_processed_extended.csv'):
    df = pd.read_csv(path_csv, usecols=['Epitope ID', 'Epitope Seq'])
    id2seq_mapper = pd.Series(df['Epitope Seq'].values, index=df['Epitope ID']).to_dict()
    seq2id_mapper = pd.Series(df['Epitope ID'].values, index=df['Epitope Seq']).to_dict()
    return id2seq_mapper, seq2id_mapper


def seq_hash_int64(seq: str) -> np.int64:
    # Используем BLAKE2b — быстро, безопасно, стабильный результат
    # digest_size=8 → 8 байт = 64 бита (оптимум под int64)
    h = hashlib.blake2b(seq.encode('utf-8'), digest_size=8)
    val = int.from_bytes(h.digest(), byteorder='big', signed=False)
    # Преобразуем в np.int64 (ограничим диапазон, чтобы не было переполнения)
    return np.int64(val & ((1 << 63) - 1))  # от 0 до 2^63-1
