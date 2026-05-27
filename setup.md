# Setup Guide

Step-by-step instructions to reproduce the experiments.

## 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/BIGREASONS/BugSeverityPrediction-CodeBERT-KICL.git
cd BugSeverityPrediction-CodeBERT-KICL

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Data Preparation

Place the preprocessed JSONL dataset files in the `data/` directory:

```
data/
├── train.jsonl   (2,414 samples)
├── valid.jsonl   (426 samples)
└── test.jsonl    (502 samples)
```

See [`data/README.md`](data/README.md) for dataset format and download instructions.

## 3. Training

### Baseline CodeBERT

```bash
python scripts/train.py --epochs 9 --lr 1e-5 --loss_type weighted_ce
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 15 | Number of training epochs |
| `--lr` | 1e-5 | Learning rate |
| `--batch_size` | 16 | Batch size |
| `--loss_type` | weighted_ce | Loss function: `ce`, `weighted_ce`, `focal` |
| `--max_length` | 256 | Max token sequence length |
| `--patience` | 5 | Early stopping patience |

### KICL Pipeline

```bash
# Stage 1: Knowledge-Intensified MLM (50% masking)
python scripts/kicl_pretrain.py --stage mlm --epochs 5

# Stage 2: Supervised Contrastive Learning
python scripts/kicl_pretrain.py --stage contrastive --epochs 5 \
    --checkpoint models/kicl_mlm_best.pt

# Stage 3: Fine-tuning
python scripts/kicl_pretrain.py --stage finetune --epochs 5 \
    --checkpoint models/kicl_contrastive_best.pt
```

## 4. Evaluation

```bash
python scripts/evaluate.py \
    --model_path models/best_model.pt \
    --test_file data/test.jsonl
```

This generates:
- `results/baseline_results.json` — all metrics
- `results/confusion_matrix.png` — confusion matrix visualization

## 5. Generate Visualizations

```bash
# Training curves
python scripts/plot_training_curves.py

# Model comparison (if KICL results exist)
python scripts/compare_results.py
```

## Hardware Notes

- Training was performed on **CPU** (Intel Core i7)
- Each epoch takes approximately **25 minutes** on CPU
- GPU training is recommended for faster iteration
- Minimum RAM: 8GB (16GB recommended)
