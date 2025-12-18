#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Extract ESMC embeddings
# Usage: bash scripts/inference/embedder.sh [INPUT_CSV] [OUTPUT_HDF5] [MODEL_CHECKPOINT]
INPUT_CSV="${1:-data/parent_epitopes/ifnepitope2_human_dataset.csv}"
EMBEDDINGS_OUTPUT="${2:-data/embeddings/ifnepitope2_human_dataset.hdf5}"
MODEL_NAME="${3:-esmc-600m}"

echo "Extracting ESMC ($MODEL_NAME) embeddings from $INPUT_CSV -> $EMBEDDINGS_OUTPUT"
python -m SWAT.scripts.extract_ESMC \
  -i "$INPUT_CSV" \
  -m "$MODEL_NAME" \
  -o "$EMBEDDINGS_OUTPUT" \
  --cuda 1 \
  --mean true \
  --multiple_runs false \
  --truncate false
