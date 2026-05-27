# Models

This directory stores trained model checkpoints. Model files (`.pt`) are excluded from version control due to their size (~500MB).

## Expected Files

| File | Description | Size |
|------|-------------|------|
| `best_model.pt` | Best baseline CodeBERT checkpoint | ~500MB |
| `training_history.json` | Training metrics per epoch | ~3KB |
| `kicl_mlm_best.pt` | KICL Stage 1 (KI-MLM) checkpoint | ~500MB |
| `kicl_contrastive_best.pt` | KICL Stage 2 (SupCon) checkpoint | ~500MB |
| `kicl_finetune_best.pt` | KICL Stage 3 (Fine-tune) checkpoint | ~500MB |

## Checkpoint Format

Each `.pt` file is a PyTorch checkpoint dict:

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'val_loss': float,
    'val_acc': float,
    'args': dict,  # training hyperparameters
}
```

## How to Reproduce

Train the baseline model:

```bash
python scripts/train.py --epochs 9 --lr 1e-5 --loss_type weighted_ce
```

Train the KICL pipeline:

```bash
python scripts/kicl_pretrain.py --stage mlm --epochs 5
python scripts/kicl_pretrain.py --stage contrastive --epochs 5 --checkpoint models/kicl_mlm_best.pt
python scripts/kicl_pretrain.py --stage finetune --epochs 5 --checkpoint models/kicl_contrastive_best.pt
```
