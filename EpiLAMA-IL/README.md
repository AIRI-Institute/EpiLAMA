# EpiLAMA-IL: Cytokine-inducing epitope modeling and inference

This repository provides a complete, reproducible workflow to preprocess epitope datasets, index epitopes against parent proteins, compute protein embeddings (Ankh and ESM C), train gradient boosting models (EpiLAMA), and run inference with pre-trained cytokine-inducing epitope classifiers (IL2, IL4, IL10, IFNG).

## Environments

- Main training environment (EpiLAMA):
  - Create with conda: `conda env create -f environment.yml`
  - Activate: `conda activate EpiLAMA`

- Ankh embeddings environment:
  - Create: `conda env create -f env_ankh.yml`
  - Activate: `conda activate ankh`

- ESM C embeddings environment (inside `scripts/inference`):
  - Use an existing environment with PyTorch CUDA 12.x support, then:
    - `pip install -r scripts/inference/requirements.txt`

## Data layout

Repository-relative paths are used throughout. Key folders:

- `data/external/` — raw epitope tables (e.g., `tcell_full_v3_processed_extended.csv`)
- `data/processed/` — processed tables and fetched FASTA sequences
- `data/parent_epitopes/` — epitopes indexed with parent proteins (Uniprot/NCBI and positions)
- `data/embeddings/` — embeddings in HDF5/pt formats
- `data/models/` — pretrained cytokine classifier `.joblib` files (IL2, IL4, IL10, IFNG)
- `data/predictions/` — optional directory for saving predictions

## End-to-end workflow

Below is the primary workflow. You can override any input/output path via CLI flags; defaults are repo-relative.

### 1) Process raw epitopes

```bash
python -m scripts.data.process_epitopes \
  -i data/external/tcell_full_v3_processed_extended.csv \
  -o data/processed/base_tcell_data_ifng.csv
```

Output: `data/processed/base_tcell_data_ifng.csv`

### 2) Index epitopes by parent proteins

This step maps epitopes to their positions inside parent protein sequences. It expects UniProt/NCBI FASTA files. If some IDs are missing, the script writes `missing_v2uniprot_id.txt` and `missing_v2ncbi_id.txt` so you can fetch those sequences in step 3.

```bash
python -m scripts.data.index_epitopes \
  -i data/processed/base_tcell_data_ifng.csv \
  -o data/parent_epitopes/base_tcell_data_ifng.csv \
  --uniprot_fasta data/processed/uniprot_sequences.fasta \
  --ncbi_fasta data/processed/ncbi_sequences_v1.fasta
```

Output: `data/parent_epitopes/base_tcell_data_ifng.csv`

### 3) Fetch missing protein sequences (UniProt/NCBI)

After step 2, collect any missing IDs and fetch sequences into FASTA files:

```bash
python scripts/data/wget_sequence.py \
  --ncbi_ids missing_v2ncbi_id.txt \
  --uniprot_ids missing_v2uniprot_id.txt \
  --out_ncbi data/processed/ncbi_sequences_v1.fasta \
  --out_uniprot data/processed/uniprot_sequences.fasta
```

Re-run step 2 after fetching sequences if necessary.

### 4) Compute Ankh embeddings

```bash
bash scripts/data/launch_ankh.sh \
  data/parent_epitopes/base_tcell_data_ifng.csv \
  data/embeddings/base_tcell_data_ifng_ankh.hdf5
```

This calls `scripts/data/extract_ankh_embeddings.py` and supports additional flags like `--cuda`, `--mean`, `--multiple_runs`, and `--truncate`.

### 5) Compute ESM C embeddings

From the repo root:

```bash
bash scripts/inference/embedder.sh \
  data/parent_epitopes/ifnepitope2_human_dataset.csv \
  data/embeddings/ifnepitope2_human_dataset.hdf5 \
  esmc-600m
```

### 6) Train EpiLAMA with precomputed embeddings

Paths and options are configured in `scripts/modeling/config.yaml` (all repo-relative). Run:

```bash
python -m scripts.modeling.train_lama_hierarchical_embeds
```

Models are saved to `data/models/`.

### 7) Cytokine classifier inference (IL2, IL4, IL10, IFNG)

This step predicts cytokine-inducing epitope responses using pre-trained classifier models.

#### Environment setup

The inference pipeline requires ESM C embeddings. If you haven't already:

```bash
# Install requirements in your environment (with PyTorch CUDA support)
pip install -r scripts/inference/requirements.txt
```

#### Input CSV format

Your input CSV must contain the following columns (tab or comma-separated):

- `Protein ID` — Protein identifier
- `Protein Seq` — Full protein sequence
- `Starting Position` — Epitope start position in the protein (1-indexed)
- `Ending Position` — Epitope end position in the protein (inclusive)

Example (`proteins.csv`):

```csv
Protein ID,Protein Seq,Starting Position,Ending Position
P20569,MGHIITYCQVHTNISILIRKAHHIIFFVIDCDCISLQFSNYVHHGNRFRTVLISKTSIACFSDIKRILPCTFKIYSINDCP,13,21
```

See `scripts/inference/proteins.csv` for a complete example.

#### Running inference

Navigate to the inference directory and run the prediction script:

```bash
cd scripts/inference

# Using GPU (recommended)
python -m SWAT.scripts.predict \
  -i proteins.csv \
  -o predictions.csv \
  -c config.yml \
  --cuda 0

# Using CPU only
python -m SWAT.scripts.predict \
  -i proteins.csv \
  -o predictions.csv \
  -c config.yml \
  --cpu
```

**Important notes:**
- Before running inference, ensure `config.yml` has the correct `models_dir` path pointing to your trained `.joblib` model files
- The script must be run from the `scripts/inference` directory for the module imports to work correctly
- The prediction process extracts ESM C embeddings on-the-fly, so it may take time depending on the dataset size

#### Output format

The output CSV (`predictions.csv`) contains the original epitope information plus prediction scores and binary labels for each cytokine:

- `Score_il2`, `Prediction_il2` — IL-2 response predictions
- `Score_il4`, `Prediction_il4` — IL-4 response predictions
- `Score_il10`, `Prediction_il10` — IL-10 response predictions
- `Score_ifng`, `Prediction_ifng` — IFN-γ response predictions

Prediction scores are continuous values; predictions are binary (0 or 1) based on the classifier thresholds.

## Configuration

- Global paths (repo-relative) are in `config.ini`.
- Modeling-specific paths are in `scripts/modeling/config.yaml`.
- Inference-specific config resides in `scripts/inference/config.yml`.

All configs now use only paths within this repository under `data/`.

## Notes & tips

- If GPUs are available, use `--cuda <index>` flags in embedding scripts to utilize them.
- The `wget_sequence.py` script respects NCBI/UniProt rate limits; for large batches, it will take time.
- Ensure `data/models/` contains the four classifier files:
  - `human_IL2_predictor.joblib`, `human_IL4_predictor.joblib`, `human_IL10_predictor.joblib`, `human_Ifng_predictor.joblib`

## License

TBD
