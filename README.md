# LCF-ATEPC

Aspect Term Extraction and Polarity Classification using BERT with Local Context Focus mechanism.

## Overview

Extracts product aspects from reviews and classifies their sentiment polarity.

```
Input:  "This phone has stylish design, but the camera is disappointing."
Output: design → Positive, camera → Negative
```

## Project Structure

```
code/
├── config.py       # Hyperparameters
├── model.py        # BERT + LCF + CRF model
├── utils.py        # Dataset and utilities
├── process.py      # Data preprocessing
├── train.py        # Training script
├── test.py         # Evaluation script
└── predict.py      # Inference script
```

## Setup

### 1. Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install torch transformers pandas torchcrf
```

### 2. BERT Model (Required)

Download `bert-base-chinese` and place it in the parent directory:

```bash
# Using Hugging Face CLI
huggingface-cli download bert-base-chinese --local-dir ../huggingface/bert-base-chinese

# Or manually download from: https://huggingface.co/bert-base-chinese
```

Expected structure:

```
parent_folder/
├── code/                        # This repository
│   ├── config.py
│   └── ...
└── huggingface/
    └── bert-base-chinese/
        ├── config.json
        ├── vocab.txt
        └── model.safetensors    # or pytorch_model.bin
```

> **Note**: Modify `BERT_MODEL_NAME` in `config.py` if using a different path.

### 3. Run

```bash
# Preprocess data
python process.py

# Train model
python train.py

# Evaluate
python test.py

# Inference
python predict.py
```

## Model Architecture

| Component | Description |
|-----------|-------------|
| Encoder | BERT-base-chinese (768-dim) |
| Entity Extraction | Linear + CRF → BIO tags (O, B-ASP, I-ASP) |
| Sentiment Classification | LCF weighting + Self-Attention → Positive/Negative |

**LCF (Local Context Focus)**: Weights tokens by distance to aspects, focusing on relevant local context.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SRD` | 3 | Local context window size |
| `BATCH_SIZE` | 50 | Batch size |
| `EPOCH` | 100 | Training epochs |
| `LR` | 1e-4 | Learning rate |
| `LCF` | cdw | Strategy: cdw / cdm / fusion |

## Data Format

**Input** (one token per line):

```
screen  B-ASP  1
is      O      -1
good    O      -1
```

**Labels**: BIO tags (O/B-ASP/I-ASP), Polarity (0=Negative, 1=Positive, -1=None)

**Domains**: camera, car, laptop, notebook, phone, restaurant, twitter, mixed

## Evaluation

Joint accuracy: both entity span AND sentiment must match ground truth.

```
Metrics: Precision, Recall, F1-Score
```

## References

- BERT: Devlin et al.
- LCF-ATEPC: Zeng et al.
- CRF: Lafferty et al.
