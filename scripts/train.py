"""
Training script for CodeBERT bug severity classifier.

Fixes for majority-class collapse:
  1. Inverse-frequency class weights in loss function
  2. Prediction distribution monitoring per epoch
  3. Lower learning rate (1e-5 default)
  4. Focal Loss option (--loss_type focal)

Usage:
    python scripts/train.py --epochs 15 --lr 1e-5 --loss_type weighted_ce
    python scripts/train.py --epochs 15 --lr 1e-5 --loss_type focal
    python scripts/train.py --epochs 1 --max_train_samples 50  # Quick smoke test
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BugSeverityDataset
from model import CodeBERTClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Train CodeBERT bug severity classifier')
    parser.add_argument('--train_file', type=str, default='data/train.jsonl')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max token length (256 for CPU speed, 512 for full quality)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Effective batch = batch_size * gradient_accumulation_steps')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (1e-5 recommended for imbalanced data)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--loss_type', type=str, default='weighted_ce',
                        choices=['ce', 'weighted_ce', 'focal'],
                        help='Loss function: ce (plain), weighted_ce, or focal')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit training samples (for debugging)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                        help='Limit validation samples (for debugging)')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights_sklearn(dataset, num_labels, device):
    """
    Compute balanced class weights using sklearn.
    Uses: weight_i = n_samples / (n_classes * count_i)
    This is the standard approach for imbalanced classification.
    """
    train_labels = np.array([s['label'] for s in dataset.samples])
    classes = np.arange(num_labels)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels,
    )

    label_counts = Counter(train_labels.tolist())
    total = len(dataset)
    print('\n  Class distribution and weights (sklearn balanced):')
    for i in range(num_labels):
        count = label_counts.get(i, 0)
        print(f'    Class {i}: {count:>5} samples ({count/total*100:5.1f}%) -> weight={weights[i]:.4f}')

    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return weights, train_labels


def create_weighted_sampler(dataset, train_labels):
    """
    Create WeightedRandomSampler to fix batch-level class imbalance.
    Each sample is weighted by the inverse frequency of its class,
    so minority classes are oversampled.
    """
    class_counts = Counter(train_labels.tolist())
    class_sample_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.tensor(
        [class_sample_weights[int(label)] for label in train_labels],
        dtype=torch.float,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    print(f'  WeightedRandomSampler created ({len(sample_weights)} samples)')
    return sampler


def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum_steps):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pred_counts = Counter()

    optimizer.zero_grad()
    progress = tqdm(dataloader, desc='Training', leave=False)
    for step, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss'] / grad_accum_steps

        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs['loss'].item() * input_ids.size(0)
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Track prediction distribution
        for p in preds.cpu().tolist():
            pred_counts[p] += 1

        progress.set_postfix({
            'loss': f'{outputs["loss"].item():.4f}',
            'acc': f'{correct/total:.3f}'
        })

    return total_loss / total, correct / total, pred_counts


def evaluate(model, dataloader, device):
    """Evaluate model, return average loss, accuracy, and prediction distribution."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    pred_counts = Counter()
    label_counts = Counter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs['loss'].item() * input_ids.size(0)
            preds = outputs['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p in preds.cpu().tolist():
                pred_counts[p] += 1
            for l in labels.cpu().tolist():
                label_counts[l] += 1

    return total_loss / total, correct / total, pred_counts, label_counts


def format_pred_distribution(pred_counts, total, label_names=None):
    """Format prediction distribution as a string."""
    if label_names is None:
        label_names = {0: 'Crit', 1: 'Major', 2: 'Med', 3: 'Minor'}
    parts = []
    for i in sorted(label_names.keys()):
        count = pred_counts.get(i, 0)
        pct = count / total * 100 if total > 0 else 0
        parts.append(f'{label_names[i]}:{count}({pct:.0f}%)')
    return ' | '.join(parts)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f'Loading tokenizer: {args.model_name}')
    tokenizer_path = 'codet5p_tokenizer' if 'codet5p' in args.model_name.lower() and os.path.exists('codet5p_tokenizer') else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load datasets
    print('Loading datasets...')
    train_dataset = BugSeverityDataset(
        args.train_file, tokenizer, args.max_length, max_samples=args.max_train_samples
    )
    valid_dataset = BugSeverityDataset(
        args.valid_file, tokenizer, args.max_length, max_samples=args.max_valid_samples
    )
    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Valid: {len(valid_dataset)} samples')

    # Compute class weights from training data
    class_weights = None
    train_labels = None
    if args.loss_type in ('weighted_ce', 'focal'):
        class_weights, train_labels = compute_class_weights_sklearn(
            train_dataset, args.num_labels, device
        )

    # WeightedRandomSampler to fix batch-level imbalance
    sampler = None
    if train_labels is not None:
        sampler = create_weighted_sampler(train_dataset, train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler,  # replaces shuffle=True; sampler handles rebalancing
        shuffle=(sampler is None),  # only shuffle if no sampler
        num_workers=0, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model with class weights
    print(f'Creating model with {args.num_labels} labels, loss={args.loss_type}...')
    model = CodeBERTClassifier(
        model_name=args.model_name,
        num_labels=args.num_labels,
        class_weights=class_weights,
        loss_type=args.loss_type,
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {param_count:,} total, {trainable:,} trainable')

    # Optimizer and scheduler
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    print(f'\n{"="*60}')
    print(f'Training Configuration:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size} (effective: {effective_batch})')
    print(f'  Max sequence length: {args.max_length}')
    print(f'  Learning rate: {args.lr}')
    print(f'  Loss type: {args.loss_type}')
    print(f'  Total optimizer steps: {total_steps}')
    print(f'  Warmup steps: {warmup_steps}')
    print(f'  Early stopping patience: {args.patience}')
    print(f'{"="*60}\n')

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_preds = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.gradient_accumulation_steps
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(model, valid_loader, device)

        epoch_time = time.time() - epoch_start

        # Check for class collapse
        val_total = sum(val_preds.values())
        dominant_class_pct = max(val_preds.values()) / val_total * 100 if val_total > 0 else 0
        classes_predicted = len([c for c in val_preds.values() if c > 0])

        record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 4),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4),
            'epoch_time_s': round(epoch_time, 1),
            'train_pred_dist': dict(sorted(train_preds.items())),
            'val_pred_dist': dict(sorted(val_preds.items())),
            'classes_predicted': classes_predicted,
        }
        history.append(record)

        print(f'\nEpoch {epoch}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
              f'Time: {epoch_time:.0f}s')
        print(f'  Train preds: {format_pred_distribution(train_preds, sum(train_preds.values()))}')
        print(f'  Val preds:   {format_pred_distribution(val_preds, val_total)}')
        print(f'  Val labels:  {format_pred_distribution(val_labels, sum(val_labels.values()))}')

        if classes_predicted <= 1:
            print(f'  [!] WARNING: Model predicting only {classes_predicted} class(es)!')
        elif dominant_class_pct > 80:
            print(f'  [!] WARNING: Dominant class = {dominant_class_pct:.0f}% of predictions')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args),
            }, model_path)
            print(f'  -> Saved best model (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  -> Early stopping after {args.patience} epochs without improvement')
                break

    total_time = time.time() - start_time
    print(f'\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)')
    print(f'Best validation loss: {best_val_loss:.4f}')

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training history saved to {history_path}')


if __name__ == '__main__':
    main()
