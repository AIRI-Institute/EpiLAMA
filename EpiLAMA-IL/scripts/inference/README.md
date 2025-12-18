# Setup Instructions

## Environment Setup

1. Create a new conda environment with Python 3.12.9:
```bash
conda create -n IL-prediction python=3.12.9
```

2. Activate the environment:
```bash
conda activate IL-prediction
```

3. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

The inference pipeline consists of two main steps:

1. Extracting ESMC embeddings from input sequences
2. Predicting cytokine responses using the extracted embeddings

### Running the Pipeline

To run the complete pipeline, use the `predict.sh` script:

```bash
bash predict.sh
```
