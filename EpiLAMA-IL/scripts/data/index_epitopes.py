import re
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import parse_fasta_to_dict, filter_by_length, deduplicate_df


def find_all_substrings(subseq: str, sequence: str) -> list[int]:
    """Return all start indices where subseq occurs in sequence (overlapping)."""
    return [m.start() for m in re.finditer(f'(?={re.escape(subseq)})', sequence)]


def filter_matched_epitopes(df: pd.DataFrame, seq_data: dict[str, str], sequence_column: str = 'Epitope Seq',
                            database_id: str = 'Uniprot ID') -> pd.DataFrame:
    matched_rows = []
    not_found_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Matching against {database_id}"):
        protein_id = row[database_id]
        epitope_seq = row[sequence_column].replace('-', '')

        if pd.isna(protein_id):
            continue

        protein_seq = seq_data.get(protein_id)

        if not protein_seq:
            not_found_ids.append(protein_id)
            continue

        matches = find_all_substrings(epitope_seq, protein_seq)
        for start_idx in matches:
            row_data = {
                database_id: protein_id,
                'Epitope ID': row['Epitope ID'],
                sequence_column: epitope_seq,
                'Starting Position': start_idx + 1,
                'Ending Position': start_idx + len(epitope_seq)
            }
            matched_rows.append(row_data)

    with open(f'missing_v2{database_id.replace(" ", "_").lower()}.txt', 'a') as f:
        for pid in set(not_found_ids):  # set() — чтобы избежать повторов
            f.write(f"{pid}\n")

    return pd.DataFrame(matched_rows)


def deduplicate_by_antigen_sequence(df: pd.DataFrame, seq_dict: dict, database_id: str) -> pd.DataFrame:
    """
    Remove duplicates where different database IDs map to the same protein sequence.
    Rows with missing protein sequences are left untouched.

    Parameters:
        df: DataFrame with columns ['Epitope ID', 'Epitope Seq', database_id, 'Starting Position', 'Ending Position']
        seq_dict: Mapping database_id -> protein sequence

    Returns:
        DataFrame deduplicated by (Epitope ID, Epitope Seq, Starting Position, Ending Position, Protein Seq)
    """

    # Add protein sequence column
    df['Protein Seq'] = df[database_id].map(seq_dict)

    # Consider only rows with a known protein sequence
    mask = df['Protein Seq'].notna()

    # Deduplicate only among rows with known protein sequence
    dedup = df[mask].drop_duplicates(
        subset=['Epitope ID', 'Epitope Seq', 'Starting Position', 'Ending Position', 'Protein Seq']
    )

    # Keep rows with NA unchanged
    result = pd.concat([dedup, df[~mask]], ignore_index=True)

    # Drop helper column
    result = result.drop(columns='Protein Seq')

    return result


def сoncat_data(parented_matched, blast_output) -> pd.DataFrame:
    blast_output = blast_output[['Epitope ID', 'Epitope Seq', 'Num changes', 'NCBI ID', 'Starting Position', 'Ending Position']]

    # Add source flags
    parented_matched = parented_matched.copy()
    parented_matched['from_blast'] = False

    blast_output = blast_output.copy()
    blast_output['from_blast'] = True

    df = pd.concat([parented_matched, blast_output])

    df[['Num changes']] = df[['Num changes']].fillna(0)

    idx = df['Uniprot ID'].notna() & df['NCBI ID'].notna()
    assert idx.sum() == 0, "Rows with both Uniprot ID and NCBI ID found!"
    return df


def compare_sequences(row, uniprot_dict, ncbi_dict):
    try:
        uniprot_seq = uniprot_dict[row['Uniprot ID']]
        ncbi_seq = ncbi_dict[row['NCBI ID']]
    except KeyError:
        return None

    if uniprot_seq == ncbi_seq:
        row = row.copy()
        row['NCBI ID'] = None
        return row
    else:
        row_uniprot = row.copy()
        row_uniprot['NCBI ID'] = None

        row_ncbi = row.copy()
        row_ncbi['Uniprot ID'] = None

        return [row_uniprot, row_ncbi]


def expand_mismatches(df, uniprot_dict, ncbi_dict):
    new_rows = []
    for _, row in df.iterrows():
        result = compare_sequences(row, uniprot_dict, ncbi_dict)
        if result is None:
            continue
        elif isinstance(result, list):
            new_rows.extend(result)
        else:
            new_rows.append(result)
    return pd.DataFrame(new_rows)


def postprocess(parented_matched) -> pd.DataFrame:

    # Example post-processing hook; currently keeps all rows
    parented_matched = parented_matched.copy()

    parented_matched['from_blast'] = False

    parented_matched[['Num changes']] = 0

    idx = parented_matched['Uniprot ID'].notna() & parented_matched['NCBI ID'].notna()
    assert idx.sum() == 0, "Rows with both Uniprot ID and NCBI ID found"

    return parented_matched


def split_by_merge(parented_matched):
    both_merged = parented_matched[parented_matched['_merge'] == 'both']
    unmerged = parented_matched[parented_matched['_merge'] != 'both']

    unmerged = unmerged.drop(columns=['_merge'])
    both_merged = both_merged.drop(columns=['_merge'])
    return both_merged, unmerged


def index_epitopes(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict):
    base_df_uniprot = df[['Epitope ID', 'Epitope Seq', 'Uniprot ID', 'Starting Position', 'Ending Position']].drop_duplicates().copy()
    base_df_uniprot = deduplicate_by_antigen_sequence(base_df_uniprot, uniprot_sequences, database_id='Uniprot ID')
    uniprot_parented_df = filter_matched_epitopes(base_df_uniprot, uniprot_sequences, database_id='Uniprot ID')

    base_df_ncbi = df[['Epitope ID', 'Epitope Seq', 'NCBI ID', 'Starting Position', 'Ending Position']].drop_duplicates().copy()
    base_df_ncbi = deduplicate_by_antigen_sequence(base_df_ncbi, ncbi_sequences, database_id='NCBI ID')
    ncbi_parented_df = filter_matched_epitopes(base_df_ncbi, ncbi_sequences, database_id='NCBI ID')

    parented_matched = pd.merge(uniprot_parented_df,
                                ncbi_parented_df,
                                on=['Epitope ID', 'Starting Position', 'Ending Position', 'Epitope Seq'],
                                how='outer', indicator=True)  # rows unmatched on one of the sources

    both_merged, unmerged = split_by_merge(parented_matched)
    both_merged = expand_mismatches(both_merged, uniprot_sequences, ncbi_sequences)
    output_df = pd.concat([both_merged, unmerged])
    output_df = output_df.drop_duplicates()
    output_df = postprocess(output_df)

    return output_df


def index_unlabeled_epitopes(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict, output_path: Path):
    assert df['Epitope ID'].duplicated().sum() == 0, "Epitope IDs must be unique"
    df = filter_by_length(df, column_name='Epitope Seq', min_length=13, max_length=25)

    base_df_uniprot = df[['Epitope ID', 'Epitope Seq', 'Uniprot ID', 'Starting Position', 'Ending Position']].copy()
    base_df_uniprot = deduplicate_by_antigen_sequence(base_df_uniprot, uniprot_sequences, database_id='Uniprot ID')
    uniprot_parented_df = filter_matched_epitopes(base_df_uniprot, uniprot_sequences, database_id='Uniprot ID')
    uniprot_parented_df = uniprot_parented_df.merge(base_df_uniprot, on=['Epitope ID', 'Epitope Seq', 'Starting Position', 'Ending Position', 'Uniprot ID'])

    base_df_ncbi = df[['Epitope ID', 'Epitope Seq', 'NCBI ID', 'Starting Position', 'Ending Position']].copy()
    base_df_ncbi = deduplicate_by_antigen_sequence(base_df_ncbi, ncbi_sequences, database_id='NCBI ID')
    ncbi_parented_df = filter_matched_epitopes(base_df_ncbi, ncbi_sequences, database_id='NCBI ID')
    ncbi_parented_df = ncbi_parented_df.merge(base_df_ncbi, on=['Epitope ID', 'Epitope Seq', 'Starting Position', 'Ending Position', 'NCBI ID'])

    output_df = pd.concat([uniprot_parented_df, ncbi_parented_df], ignore_index=True)
    output_df = output_df.drop_duplicates(subset='Epitope ID')
    output_df = postprocess(output_df)
    output_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Index epitopes by matching them to parent protein sequences")
    parser.add_argument("--input", "-i", required=False, default="data/processed/base_tcell_data_ifng.csv",
                        help="Path to processed epitopes CSV")
    parser.add_argument("--output", "-o", required=False, default="data/parent_epitopes/base_tcell_data_ifng.csv",
                        help="Path to output parented epitopes CSV")
    parser.add_argument("--uniprot_fasta", required=False, default="data/processed/uniprot_sequences.fasta",
                        help="Path to UniProt FASTA with protein sequences")
    parser.add_argument("--ncbi_fasta", required=False, default="data/processed/ncbi_sequences_v1.fasta",
                        help="Path to NCBI FASTA with protein sequences")
    parser.add_argument("--add_unparented", action="store_true", help="Append unmatched epitopes using fallback rules")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_path)
    uniprot_sequences = parse_fasta_to_dict(Path(args.uniprot_fasta))
    ncbi_sequences = parse_fasta_to_dict(Path(args.ncbi_fasta))

    output_df = index_epitopes(input_df, uniprot_sequences, ncbi_sequences)

    if args.add_unparented:
        output_path = output_path.with_name(output_path.stem + "_with_unparented.csv")
        left_df = input_df[~input_df['Epitope ID'].isin(output_df['Epitope ID'])]  # epitopes without matched parent protein
        blast_output = pd.read_csv("data/blast/blast_output.csv")
        blast_output = blast_output[blast_output['Epitope ID'].isin(left_df['Epitope ID'])]
        blast_output['from_blast'] = True

        left_2_df = left_df[~left_df['Epitope ID'].isin(blast_output['Epitope ID'])]  # epitopes not found by BLAST
        left_2_df = left_2_df[['Epitope ID', 'Epitope Seq']].drop_duplicates()
        left_2_df['Num changes'] = 0
        left_2_df['NCBI ID'] = left_2_df['Epitope ID']
        left_2_df['from_blast'] = False
        left_2_df['Starting Position'] = 1
        left_2_df['Ending Position'] = left_2_df['Epitope Seq'].str.len()

        assert blast_output['Epitope ID'].nunique() + output_df['Epitope ID'].nunique() + left_2_df['Epitope ID'].nunique() == input_df['Epitope ID'].nunique()

        # blast_output_matched columns: NCBI ID, Epitope ID, Hit Seq, Epitope Seq, Starting Position, Ending Position
        # blast_output columns: Epitope ID, Epitope Seq, Hit Seq, Num changes, NCBI ID, from_blast
        # desired blast_output_matched columns after processing: NCBI ID, Epitope ID, Epitope Seq, Starting Position, Ending Position, Num changes, from_blast
        blast_output_matched = filter_matched_epitopes(blast_output, ncbi_sequences, database_id='NCBI ID', sequence_column='Hit Seq')
        blast_output_matched = blast_output_matched.drop(columns='Hit Seq')
        blast_output_matched = blast_output_matched.merge(blast_output[['Epitope ID', 'NCBI ID', 'Epitope Seq', 'Num changes', 'from_blast']], on=['Epitope ID', 'NCBI ID'])
        blast_output_matched = deduplicate_by_antigen_sequence(blast_output_matched, ncbi_sequences, database_id='NCBI ID')
        output_2_df = pd.concat([blast_output_matched, left_2_df])
        output_2_df['Uniprot ID'] = None

        output_df = pd.concat([output_df, output_2_df], ignore_index=True)
        assert output_df['Epitope ID'].nunique() == input_df['Epitope ID'].nunique(), "Epitope IDs must be keep"

    output_df = deduplicate_df(output_df)
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
