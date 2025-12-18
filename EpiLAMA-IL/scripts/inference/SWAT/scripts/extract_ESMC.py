import argparse
from pathlib import Path
from collections import defaultdict
import torch
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import h5py
from SWAT.src.models.esmc import ESMCfloat32


LENGTH_LIMIT = 2048


def parse_fasta_to_dict(file_path: str) -> dict[str, str]:
    fasta_dict = {}
    for record in SeqIO.parse(file_path, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def assign_protein_sequence(df: pd.DataFrame, database_id: str, protein_sequences: dict) -> pd.DataFrame:
    df['Protein Seq'] = df[database_id].map(protein_sequences)
    return df


def split_by_database_id(df: pd.DataFrame) -> pd.DataFrame:
    df_uniprot = df[df['Uniprot ID'].notna()].copy()
    df_ncbi = df[df['NCBI ID'].notna()].copy()
    return df_uniprot, df_ncbi


def preprocess_positions(df: pd.DataFrame) -> pd.DataFrame:
    df['Starting Position'] = df['Starting Position'] - 1  # корректируем в нулевую индексацию
    return df


def full_preprocessing_pipeline(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict) -> pd.DataFrame:
    df = preprocess_positions(df)
    df_uniprot, df_ncbi = split_by_database_id(df)
    df_uniprot = assign_protein_sequence(df_uniprot, 'Uniprot ID', uniprot_sequences)
    df_ncbi = assign_protein_sequence(df_ncbi, 'NCBI ID', ncbi_sequences)
    return df_uniprot, df_ncbi


def correct_epitopes_df(df: pd.DataFrame, database_id: str, length_limit: int = LENGTH_LIMIT) -> pd.DataFrame:
    def process_row(row):
        seq = row['Protein Seq']
        start = row['Starting Position']
        end = row['Ending Position']
        epitope_seq = row['Epitope Seq'].replace('-', '')
        num_changes = row['Num changes']

        if len(seq) > length_limit:
            epi_len = end - start
            context = length_limit - epi_len
            left = max(0, start - context // 2)
            right = min(len(seq), end + context // 2)
            new_start = start - left
            new_end = new_start + epi_len
            seq = seq[left:right]
        else:
            new_start = start
            new_end = end

        if num_changes > 0:
            seq = seq[:new_start] + epitope_seq + seq[new_end:]
            new_end = new_start + len(epitope_seq)

        assert seq[new_start:new_end] == epitope_seq, f"{row['Epitope ID']} {row[database_id]}"

        row['Protein Seq'] = seq
        row['New Starting Position'] = new_start
        row['New Ending Position'] = new_end
        row['ID'] = f"{row[database_id]}|{row['Epitope ID']}|{row['Starting Position']}|{row['Ending Position']}|{new_start}|{new_end}"
        return row

    return df.apply(process_row, axis=1)


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
                yield ids, lengths, tokens
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
            yield ids, lengths, tokens


def extract_representations(model, df, database_id: str, mean: bool = True):
    model.eval()
    representations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    data_loader = DataLoader(df, model=model)

    with torch.no_grad():
        for batch_ids, batch_lengths, batch_tokens in tqdm(data_loader, desc="Processing batches", leave=False):
            output = model(batch_tokens)
            embeddings = output.embeddings.detach().cpu()  # (batch_size, seq_len, hidden_size)

            for i, full_id in enumerate(batch_ids):
                embedding = embeddings[i, 1:batch_lengths[i]+1, :]  # seq_len, hidden_size

                db_id, epi_id, start, end, new_start, new_end = full_id.split('|')
                embedding = embedding[int(new_start):int(new_end)]
                if mean:
                    embedding = embedding.mean(dim=0)
                representations[database_id][db_id][epi_id][(start, end)] = embedding

    return representations


def save_nested_embeddings_hdf5(embeddings_dict: dict, output_path: Path, num_chunk: int | None = None):
    """
    Save nested embeddings dictionary (Source → ProteinID → EpitopeID → (start, end) → np.array) to HDF5.
    """
    if num_chunk:
        output_path = output_path.with_suffix(f'.{num_chunk}.hdf5')

    with h5py.File(output_path, 'w') as h5f:
        for source, proteins in embeddings_dict.items():
            for protein_id, epitopes in proteins.items():
                for epitope_id, spans in epitopes.items():
                    for (start, end), embedding in spans.items():
                        group_path = f"{source}/{protein_id}/{epitope_id}/{start}_{end}"
                        h5f.create_dataset(group_path, data=embedding.numpy())


def save_one_key_pt(embeddings_dict: dict, output_path: Path, num_chunk: int | None = None):
    """
    Save embeddings dictionary (Source → ProteinID → EpitopeID → (start, end) → np.array) to HDF5.
    """
    if num_chunk:
        output_path = output_path.with_suffix(f'.{num_chunk}.hdf5')

    embeddings = {}
    for source, proteins in embeddings_dict.items():
        for protein_id, epitopes in proteins.items():
            for epitope_id, spans in epitopes.items():
                for (start, end), embedding in spans.items():
                    embeddings[epitope_id + f'_{start}_{end}'] = embedding

    torch.save(embeddings, output_path)


def single_run(df_path: Path, model: ESMCfloat32, uniprot_sequences: dict,
               ncbi_sequences: dict, embed_path: Path,
               mean: bool = True, save_func=save_one_key_pt):

    df = pd.read_csv(df_path)
    embeddings_uniprot = {}
    embeddings_ncbi = {}

    df_uniprot, df_ncbi = full_preprocessing_pipeline(df, uniprot_sequences, ncbi_sequences)
    if not df_uniprot.empty:
        df_uniprot = correct_epitopes_df(df_uniprot, 'Uniprot ID')
        embeddings_uniprot = extract_representations(model, df_uniprot, 'Uniprot ID', mean=mean)
    if not df_ncbi.empty:
        df_ncbi = correct_epitopes_df(df_ncbi, 'NCBI ID')
        embeddings_ncbi = extract_representations(model, df_ncbi, 'NCBI ID', mean=mean)

    embeddings = {
        'Uniprot': embeddings_uniprot.get('Uniprot ID', {}),
        'NCBI': embeddings_ncbi.get('NCBI ID', {})
    }

    save_func(embeddings, embed_path)


def multiple_runs(df_path: Path, model: ESMCfloat32, uniprot_sequences: dict, ncbi_sequences: dict, embed_path: Path, chunksize: int = 25_000):
    chunks = pd.read_csv(df_path, chunksize=chunksize)
    embeddings_uniprot = {}
    embeddings_ncbi = {}
    for num_chunk, chunk in enumerate(chunks):
        df_uniprot, df_ncbi = full_preprocessing_pipeline(chunk, uniprot_sequences, ncbi_sequences)
        if not df_uniprot.empty:
            df_uniprot = correct_epitopes_df(df_uniprot, 'Uniprot ID')
            embeddings_uniprot = extract_representations(model, df_uniprot, 'Uniprot ID')
        if not df_ncbi.empty:
            df_ncbi = correct_epitopes_df(df_ncbi, 'NCBI ID')
            embeddings_ncbi = extract_representations(model, df_ncbi, 'NCBI ID')

        embeddings = {
            'Uniprot': embeddings_uniprot.get('Uniprot ID', {}),
            'NCBI': embeddings_ncbi.get('NCBI ID', {})
        }

        save_nested_embeddings_hdf5(embeddings, embed_path, num_chunk)
        print(f"Processed chunk {num_chunk + 1}")


def main():
    parser = argparse.ArgumentParser(description="Extracting ESMC representations from a FASTA file")
    parser.add_argument("-i", "--data_name", type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument("-m", "--model_checkpoint", type=str, required=True, help="Model checkpoint identifier")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use (default: 0)")
    parser.add_argument("--multiple_runs", type=str, choices=["true", "false"], default="false", help="Iterative mode (true/false)")
    parser.add_argument("--mean", type=str, choices=["true", "false"], default="true", help="Use mean representation (true/false)")

    args = parser.parse_args()

    args.multiple_runs = args.multiple_runs == "true"
    args.mean = args.mean == "true"

    # Define the device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Load the model based on the checkpoint identifier
    if args.model_checkpoint == 'esmc-300m':
        model = ESMCfloat32.from_pretrained("esmc_300m").to(device)  # "cuda" or "cpu"
        print("Model transferred to device:", model.device)

    elif args.model_checkpoint == 'esmc-600m':
        model = ESMCfloat32.from_pretrained("esmc_600m").to(device)
        print("Model transferred to device:", model.device)
    else:
        print("Model not found!")
        print("Choose a valid model checkpoint: 'esmc-300m' or 'esmc-600m'")
        exit(1)

    project_dir = Path('/mnt/nfs_protein/gavrilenko/vaccine-design/')
    uniprot_sequences = parse_fasta_to_dict(project_dir / 'sequences.fasta')
    ncbi_sequences = parse_fasta_to_dict(project_dir / 'ncbi_sequences.fasta')

    df_path = project_dir / 'cytokine' / 'parent_epitopes' / args.data_name
    embed_dir = project_dir / 'cytokine' / 'embeddings'

    if args.multiple_runs:
        embed_path = embed_dir / 'unlabaled' / args.output
        multiple_runs(df_path, model, uniprot_sequences, ncbi_sequences, embed_path, chunksize=25_000)
    else:
        embed_path = embed_dir / args.output
        single_run(df_path, model, uniprot_sequences, ncbi_sequences, embed_path, mean=args.mean)


if __name__ == "__main__":
    main()
