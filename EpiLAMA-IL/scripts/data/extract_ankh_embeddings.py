import argparse
from pathlib import Path
from collections import defaultdict
import torch
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import h5py
import ankh


LENGTH_LIMIT = 1024  # Ankh typically has a lower limit than ESMC


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
    # Convert to zero-based indexing
    df['Starting Position'] = df['Starting Position'] - 1
    return df


def full_preprocessing_pipeline(df: pd.DataFrame, uniprot_sequences: dict, ncbi_sequences: dict) -> pd.DataFrame:
    df = preprocess_positions(df)
    df_uniprot, df_ncbi = split_by_database_id(df)
    df_uniprot = assign_protein_sequence(df_uniprot, 'Uniprot ID', uniprot_sequences)
    df_ncbi = assign_protein_sequence(df_ncbi, 'NCBI ID', ncbi_sequences)
    return df_uniprot, df_ncbi


def correct_epitopes_df(df: pd.DataFrame, database_id: str, truncate: bool = False, length_limit: int = LENGTH_LIMIT) -> pd.DataFrame:
    def process_row(row, truncate=truncate):
        seq = row['Protein Seq']
        start = row['Starting Position']
        end = row['Ending Position']
        epitope_seq = row['Epitope Seq'].replace('-', '')
        num_changes = row['Num changes']

        if truncate:
            seq = epitope_seq
            new_start = 0
            new_end = len(epitope_seq)
            row['Protein Seq'] = epitope_seq
            row['New Starting Position'] = new_start
            row['New Ending Position'] = new_end
            row['ID'] = f"{row[database_id]}|{row['Epitope ID']}|{start}|{end}|{new_start}|{new_end}"
            assert seq[new_start:new_end] == epitope_seq, f"{row['Epitope ID']} {row[database_id]}"
            assert len(seq) <= length_limit, f"{row['Epitope ID']} {row[database_id]}"
            return row

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

        if len(seq) > length_limit:
            epi_len = new_end - new_start
            context = length_limit - epi_len
            left = max(0, new_start - context // 2)
            right = min(len(seq), new_end + context // 2)
            new_start -= left
            new_end = new_start + epi_len
            seq = seq[left:right]

        assert seq[new_start:new_end] == epitope_seq, f"{row['Epitope ID']} {row[database_id]}"
        assert len(seq) <= length_limit, f"{row['Epitope ID']} {row[database_id]}"

        row['Protein Seq'] = seq
        row['New Starting Position'] = new_start
        row['New Ending Position'] = new_end
        row['ID'] = f"{row[database_id]}|{row['Epitope ID']}|{start}|{end}|{new_start}|{new_end}"
        return row

    return df.apply(process_row, axis=1)


def calculate_protein_embedding(model, tokenizer, protein_sequence: str, device: torch.device):
    """Calculate embedding for a single protein sequence using Ankh."""
    # Convert the protein sequence to a list of characters
    protein_sequence = list(protein_sequence)

    outputs = tokenizer.batch_encode_plus(
        [protein_sequence],  # Wrap in a list for batch processing
        add_special_tokens=False,
        padding=False,
        is_split_into_words=True,
        return_tensors="pt"
    )

    input_ids = outputs['input_ids'].to(device)
    attention_mask = outputs['attention_mask'].to(device)

    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    return embeddings.hidden_states[-1].squeeze(0).cpu()  # [seq_len, hidden_dim]


def extract_representations(model, tokenizer, df, database_id: str, device: torch.device, mean: bool = True):
    """Extract representations for all sequences in the dataframe."""
    model.eval()
    representations = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences", leave=False):
            sequence = row['Protein Seq']
            full_id = row['ID']

            # Calculate embedding for the entire sequence
            embedding = calculate_protein_embedding(model, tokenizer, sequence, device)

            # Parse the ID and extract epitope region
            protein_id, epi_id, start, end, new_start, new_end = full_id.split('|')
            epitope_embedding = embedding[int(new_start):int(new_end)]

            if mean:
                epitope_embedding = epitope_embedding.mean(dim=0)

            representations[database_id][protein_id][epi_id][(start, end)] = epitope_embedding

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


def single_run(df_path: Path, model, tokenizer, device: torch.device,
               uniprot_sequences: dict, ncbi_sequences: dict, embed_path: Path,
               mean: bool = True, truncate: bool = False, save_func=None):
    """Process entire dataframe in a single run."""
    df = pd.read_csv(df_path)
    embeddings_uniprot = {}
    embeddings_ncbi = {}

    df_uniprot, df_ncbi = full_preprocessing_pipeline(df, uniprot_sequences, ncbi_sequences)

    if not df_uniprot.empty:
        df_uniprot = correct_epitopes_df(df_uniprot, 'Uniprot ID', truncate=truncate)
        embeddings_uniprot = extract_representations(model, tokenizer, df_uniprot, 'Uniprot ID', device, mean=mean)

    if not df_ncbi.empty:
        df_ncbi = correct_epitopes_df(df_ncbi, 'NCBI ID', truncate=truncate)
        embeddings_ncbi = extract_representations(model, tokenizer, df_ncbi, 'NCBI ID', device, mean=mean)

    embeddings = {
        'Uniprot': embeddings_uniprot.get('Uniprot ID', {}),
        'NCBI': embeddings_ncbi.get('NCBI ID', {})
    }

    save_func(embeddings, embed_path)


def multiple_runs(df_path: Path, model, tokenizer, device: torch.device,
                  uniprot_sequences: dict, ncbi_sequences: dict, embed_path: Path,
                  chunksize: int = 10_000, mean: bool = True, truncate: bool = False, save_func=None):
    """Process dataframe in chunks for memory efficiency."""
    chunks = pd.read_csv(df_path, chunksize=chunksize)

    for num_chunk, chunk in enumerate(chunks):
        embeddings_uniprot = {}
        embeddings_ncbi = {}

        df_uniprot, df_ncbi = full_preprocessing_pipeline(chunk, uniprot_sequences, ncbi_sequences)

        if not df_uniprot.empty:
            df_uniprot = correct_epitopes_df(df_uniprot, 'Uniprot ID', truncate=truncate)
            embeddings_uniprot = extract_representations(model, tokenizer, df_uniprot, 'Uniprot ID', device, mean=mean)

        if not df_ncbi.empty:
            df_ncbi = correct_epitopes_df(df_ncbi, 'NCBI ID', truncate=truncate)
            embeddings_ncbi = extract_representations(model, tokenizer, df_ncbi, 'NCBI ID', device, mean=mean)

        embeddings = {
            'Uniprot': embeddings_uniprot.get('Uniprot ID', {}),
            'NCBI': embeddings_ncbi.get('NCBI ID', {})
        }

        save_func(embeddings, embed_path, num_chunk)
        print(f"Processed chunk {num_chunk + 1}")


def main():
    parser = argparse.ArgumentParser(description="Extract Ankh embeddings for epitopes with context")
    parser.add_argument("-i", "--data_name", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("-m", "--model_size", type=str, choices=['base', 'large'], default='large',
                        help="Ankh model size (base/large)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output HDF5 file")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number to use (default: 0)")
    parser.add_argument("--multiple_runs", type=str, choices=["true", "false"], default="false",
                        help="Iterative mode (true/false)")
    parser.add_argument("--mean", type=str, choices=["true", "false"], default="true",
                        help="Use mean representation (true/false)")
    parser.add_argument("--truncate", type=str, choices=["true", "false"], default="false",
                        help="Truncate representation (true/false)")
    parser.add_argument("--chunksize", type=int, default=10000,
                        help="Chunk size for multiple runs mode (default: 10000)")
    parser.add_argument("--uniprot_fasta", type=str, default="data/processed/uniprot_sequences.fasta",
                        help="Path to UniProt FASTA")
    parser.add_argument("--ncbi_fasta", type=str, default="data/processed/ncbi_sequences_v1.fasta",
                        help="Path to NCBI FASTA")

    args = parser.parse_args()

    args.multiple_runs = args.multiple_runs == "true"
    args.mean = args.mean == "true"
    args.truncate = args.truncate == "true"

    # Define the device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # Load the Ankh model based on size
    print(f"Loading Ankh {args.model_size} model...")
    if args.model_size == 'large':
        model, tokenizer = ankh.load_large_model()
    else:
        model, tokenizer = ankh.load_base_model()

    model.eval()
    model.to(device)
    print(f"Model transferred to device: {device}")

    # Load sequences
    uniprot_sequences = parse_fasta_to_dict(Path(args.uniprot_fasta))
    ncbi_sequences = parse_fasta_to_dict(Path(args.ncbi_fasta))

    # Setup paths
    df_path = Path(args.data_name)
    embed_path = Path(args.output)
    embed_path.parent.mkdir(parents=True, exist_ok=True)

    # Run processing
    if args.multiple_runs:
        multiple_runs(df_path, model, tokenizer, device, uniprot_sequences, ncbi_sequences,
                      embed_path, chunksize=args.chunksize, mean=args.mean, truncate=args.truncate,
                      save_func=save_nested_embeddings_hdf5)
    else:
        single_run(df_path, model, tokenizer, device, uniprot_sequences, ncbi_sequences,
                   embed_path, mean=args.mean, truncate=args.truncate,
                   save_func=save_nested_embeddings_hdf5)


if __name__ == "__main__":
    main()
