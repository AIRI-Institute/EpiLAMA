#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Extract Ankh embeddings
# Usage: bash scripts/data/launch_ankh.sh [INPUT_CSV] [OUTPUT_HDF5]
INPUT_CSV="${1:-data/parent_epitopes/base_tcell_data_ifng.csv}"
EMBEDDINGS_OUTPUT="${2:-data/embeddings/base_tcell_data_ifng_ankh.hdf5}"

echo "Extracting Ankh embeddings from $INPUT_CSV -> $EMBEDDINGS_OUTPUT"
python scripts/data/extract_ankh_embeddings.py \
    -i "$INPUT_CSV" \
    -o "$EMBEDDINGS_OUTPUT" \
    --cuda 1 \
    --mean true \
    --multiple_runs false \
    --truncate false
