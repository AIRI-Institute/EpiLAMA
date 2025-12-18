import zipfile
import re
import json
from typing import List, Dict, Any
from glob import glob
import configparser
import pandas as pd
from utils import get_mappers_epitopes, seq_hash_int64


class BlastZipParser:
    def __init__(self, zip_paths: List[str]):
        """
        Принимает список путей к ZIP-файлам.
        """
        self.zip_paths = zip_paths
        self.json_objects: List[Dict[str, Any]] = []

    def load_jsons(self) -> None:
        """Загружает все .json файлы из всех zip-архивов"""
        for zip_path in self.zip_paths:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
                    for filename in json_files:
                        with zip_ref.open(filename) as file:
                            try:
                                json_obj = json.load(file)
                                json_obj['__source_zip__'] = zip_path  # сохраняем путь до архива
                                self.json_objects.append(json_obj)
                            except json.JSONDecodeError as e:
                                print(f"[!] Ошибка чтения {filename} в {zip_path}: {e}")
            except zipfile.BadZipFile:
                print(f"[!] Невозможно открыть архив: {zip_path}")

    def parse_blast_results(self) -> pd.DataFrame:
        """Парсит JSON BLAST-результаты и возвращает их как DataFrame."""
        data = []

        for obj in self.json_objects:
            if 'BlastOutput2' not in obj:
                continue

            result = obj['BlastOutput2']['report']['results']
            # query_title либо int, либо строка ─ оставляем как есть
            try:
                epitope_id = int(result['search']['query_title'])
            except ValueError:
                epitope_id = result['search']['query_title']

            hits = result['search'].get('hits', [])
            if not hits:
                print(f"[!] Нет hits для эпитопа {epitope_id}")
                continue

            # найдём первый «не-unknown» hit – у него самый высокий identity
            top_hit = next(
                (h for h in hits
                 if not all(d.get('accession', 'unknown') == 'unknown'
                            for d in h.get('description', []))),
                None
            )
            if top_hit is None:
                print(f"[!] Только unknown hits для эпитопа {epitope_id}")
                continue

            top_hsp = top_hit['hsps'][0]
            top_ratio = top_hsp['identity'] / top_hsp['align_len'] if top_hsp['align_len'] else 0

            # соберём все hit’ы с идентичностью == топовой (с небольшой погрешностью)
            for hit in hits:
                hsp = hit['hsps'][0]
                ratio = hsp['identity'] / hsp['align_len'] if hsp['align_len'] else 0
                if abs(ratio - top_ratio) > 1e-9:
                    break                        # остальные хуже – выходим (хиты уже отсортированы)
                if all(d.get('accession', 'unknown') == 'unknown' for d in hit.get('description', [])):
                    continue                    # пропускаем «unknown»‐хиты

                data.append({
                    'Epitope ID': epitope_id,          # временный, позже перезапишем
                    'identity_ratio': ratio,
                    'Num changes': hsp['align_len'] - hsp['identity'],
                    'Epitope Seq': hsp['qseq'],
                    'Hit Seq': hsp['hseq'],
                    'NCBI ID': ';'.join(
                        d.get('accession', 'unknown') for d in hit.get('description', [])
                    ),
                    'source_zip': obj.get('__source_zip__', 'unknown')
                })

        return pd.DataFrame(data)

    def extract(self) -> pd.DataFrame:
        """Главный метод: загружает и возвращает спарсенные данные"""
        self.load_jsons()
        return self.parse_blast_results()


def process_outputs_local_blast():
    blast_files = glob('/mnt/nfs_protein/shashkova/mRNA_vaccine_data/for_sasha/refseq_select_prot.*.csv')
    column_names = [
        "qseqid", "sseqid", "pident", "qlen", "slen", "length",
        "mismatch", "gapopen", "qstart", "qend", "sstart", "send",
        "evalue", "bitscore", "qseq", "sseq"
    ]
    dataframes = []
    for file in blast_files:
        df = pd.read_csv(file, sep='\t', header=None, names=column_names)
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[['qseqid', 'sseqid', 'qseq', 'sseq', 'mismatch', 'pident']]

    df = df.rename(columns={'qseqid': 'Epitope ID', 'sseqid': 'NCBI ID', 'qseq': 'Epitope Seq',
                            'sseq': 'Hit Seq', 'mismatch': 'Num changes', 'pident': 'identity_ratio'})

    df['identity_ratio'] = df['identity_ratio'] / 100

    def extract_accession(s: str) -> str:
        """
        Извлекает идентификатор из строки вида 'ref|WP_377042078.1|'
        """
        match = re.search(r'\|([A-Z_]+\d+\.\d+)\|', s)
        return match.group(1)

    df['NCBI ID'] = df['NCBI ID'].apply(extract_accession)
    return df


def normalize_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляем «-» из Epitope Seq, но НЕ убираем дубликаты."""
    df = df.copy()
    df['Epitope Seq'] = df['Epitope Seq'].str.replace('-', '', regex=False)
    return df


def assign_id(df: pd.DataFrame, seq2id_mapper: dict) -> pd.DataFrame:
    df = df.copy()

    # Сопоставляем через map (для известных последовательностей)
    df['Epitope ID'] = df['Epitope Seq'].map(seq2id_mapper)

    # Находим неизвестные
    mask_unknown = df['Epitope ID'].isna()
    df.loc[mask_unknown, 'Epitope ID'] = df.loc[mask_unknown, 'Epitope Seq'].apply(seq_hash_int64)

    # Приводим к int
    df['Epitope ID'] = df['Epitope ID'].astype(int)

    return df


def filter_blast_output(df: pd.DataFrame) -> pd.DataFrame:
    idx = (df['Num changes'] == 0) & (df['Epitope Seq'] != df['Hit Seq'])  # less than 1 % cases, I don't know why
    df.loc[idx, 'Num changes'] = 1
    return df


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")
    cytokine_dir = config.get("paths", "cytokine")
    zip_files = glob(cytokine_dir + 'blast/*zip')

    _, seq2id_mapper = get_mappers_epitopes()
    parser = BlastZipParser(zip_files)
    web_blast_df = parser.extract()
    local_blast_df = process_outputs_local_blast()

    df = pd.concat([web_blast_df, local_blast_df], ignore_index=True)
    df = normalize_sequences(df)
    df = assign_id(df, seq2id_mapper)
    df = df.loc[df.groupby('Epitope ID')['identity_ratio'].transform('max') == df['identity_ratio']]

    df['NCBI ID'] = df['NCBI ID'].astype(str).str.split(';')
    df = df.explode('NCBI ID').reset_index(drop=True)
    df['NCBI ID'] = df['NCBI ID'].str.strip()
    df = df[['Epitope ID', 'Epitope Seq', 'Hit Seq', 'Num changes', 'NCBI ID']]
    df = df.drop_duplicates()
    df = filter_blast_output(df)

    df.to_csv(cytokine_dir + 'blast/blast_output.csv', index=False)


if __name__ == "__main__":
    main()
