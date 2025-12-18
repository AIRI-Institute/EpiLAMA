import time
import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import requests
from tqdm import tqdm
from Bio import Entrez, SeqIO
from typing import List
from requests.exceptions import RequestException, HTTPError

# ---------------------------------------------------------------------
#  Settings
# ---------------------------------------------------------------------
MAX_UNIPROT_WORKERS = 2  # ~1–2 RPS per thread is safe
MAX_NCBI_WORKERS = 3     # NCBI rules: ≤ 3 RPS total
RETRIES = 5
BACKOFF_FACTOR = 1.0
FASTA_LINE_LEN = 60

FAILED_429_PATH = Path("not_found_proteins_v2.txt")  # where to log HTTP 429 IDs

Entrez.email = "gavrilenkoalex154@gmail.com"

NCBI_ID_RE = re.compile(r"\[NCBI\s+(\S+?)\]")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(FAILED_429_PATH, mode="a")
    ]
)


def _log_429(pid: str, source: str) -> None:
    """Log `[Source] 429` line for the given ID."""
    logging.warning(f"[{source}] HTTP 429 for {pid}")


def _wrap_fasta(pid: str, seq: str) -> str:
    lines = [f">{pid}"]
    lines += [seq[i:i + FASTA_LINE_LEN] for i in range(0, len(seq), FASTA_LINE_LEN)]
    return "\n".join(lines) + "\n"


def parse_proteins_wrong_file(filepath):
    """
    Read proteins from a text file, map each [NCBI ...] token to its prefix, and
    collect IDs from lines with HTTP 429.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    mapper = {}
    proteins429 = []
    ncbi_pattern = re.compile(r'\[NCBI (.*?)\]')
    for line in lines:
        match = ncbi_pattern.search(line)
        term = match.group(1)
        if '.' in term:
            prefix = term.split('.')[0]
        elif '-' in term:
            prefix = term.split('-')[0]
        else:
            prefix = term

        if 'HTTP Error 429: Too Many Requests' in line:
            proteins429.append(term)
        else:
            mapper[term] = prefix

    return mapper, proteins429


def get_uniprot_sequence(uniprot_id: str, *, retries=RETRIES,
                         backoff=BACKOFF_FACTOR, session: requests.Session | None = None) -> str | None:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    sess = session or requests
    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=10)
            if resp.ok:
                return "".join(resp.text.splitlines()[1:])
            print(f"[UniProt {uniprot_id}] HTTP {resp.status_code}, attempt {attempt + 1}/{retries}")
        except RequestException as e:
            print(f"[UniProt {uniprot_id}] {e}, attempt {attempt + 1}/{retries}")
        time.sleep(backoff * 2 ** attempt)
    return None


def get_ncbi_sequence(ncbi_id: str) -> str | None:
    try:
        with Entrez.efetch(db="protein", id=ncbi_id,
                           rettype="fasta", retmode="text") as handle:
            return str(SeqIO.read(handle, "fasta").seq)
    except HTTPError as e:
        if e.code == 429:
            _log_429(ncbi_id, "NCBI")
        else:
            logging.info(f"[NCBI {ncbi_id}] HTTP {e.code}")
    except Exception as e:
        logging.info(f"[NCBI {ncbi_id}] {e}")
    return None


def parallel_fetch(ids: list[str],
                   fetch_fn,
                   max_workers: int,
                   desc: str) -> list[str]:
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # lazily create requests Session for UniProt
        session = requests.Session() if fetch_fn is get_uniprot_sequence else None
        futures = {
            pool.submit(fetch_fn, pid, session=session) if session
            else pool.submit(fetch_fn, pid): pid
            for pid in ids
        }
        for fut in tqdm(as_completed(futures),
                        total=len(ids),
                        desc=desc,
                        dynamic_ncols=True):
            pid = futures[fut]
            seq = fut.result()
            if seq:
                out.append(_wrap_fasta(pid, seq))
    return out


def append_many(fasta_records: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.writelines(fasta_records)


def parse_fasta_to_dict(file_path: str) -> dict[str, str]:
    fasta_dict = {}
    for record in SeqIO.parse(file_path, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def parse_unmatched_txt(filepath: str) -> list[str]:
    """Read a .txt file of IDs (one per line) and return as a list."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def extract_ncbi_id(line: str) -> str | None:
    """Extract an NCBI ID from a log line, or None if not found."""
    m = NCBI_ID_RE.search(line)
    return m.group(1) if m else None


def parse_log(fp: Path | str) -> List[str]:
    """Parse file lines and collect all found NCBI IDs into a list."""
    ids: List[str] = []
    with Path(fp).open(encoding="utf-8") as f:
        for line in f:
            if (ncbi_id := extract_ncbi_id(line)) is not None:
                ids.append(ncbi_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Download protein sequences by UniProt/NCBI IDs")
    parser.add_argument("--ncbi_ids", default="missing_v2ncbi_id.txt", help="Path to file with NCBI IDs (one per line)")
    parser.add_argument("--uniprot_ids", default="missing_v2uniprot_id.txt", help="Path to file with UniProt IDs (one per line)")
    parser.add_argument("--out_ncbi", default="data/processed/ncbi_sequences_v1.fasta", help="Output FASTA for NCBI sequences")
    parser.add_argument("--out_uniprot", default="data/processed/uniprot_sequences.fasta", help="Output FASTA for UniProt sequences")
    args = parser.parse_args()

    # ---------- NCBI ----------
    proteins = parse_unmatched_txt(args.ncbi_ids)
    proteins = list(set([protein for protein in proteins if 'ONTIE' not in protein]))
    ncbi_fasta = parallel_fetch(
        proteins,
        fetch_fn=get_ncbi_sequence,
        max_workers=MAX_NCBI_WORKERS,
        desc="NCBI download",
    )
    append_many(ncbi_fasta, args.out_ncbi)

    # ---------- UniProt ----------
    proteins = parse_unmatched_txt(args.uniprot_ids)
    proteins = list(set([protein for protein in proteins]))
    uniprot_fasta = parallel_fetch(
        proteins,
        fetch_fn=get_uniprot_sequence,
        max_workers=MAX_UNIPROT_WORKERS,
        desc="UniProt download",
    )
    append_many(uniprot_fasta, args.out_uniprot)


if __name__ == "__main__":
    main()
