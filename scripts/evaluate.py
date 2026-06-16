"""
Evaluation script for bug severity prediction.

Computes all required metrics:
  - Precision (Weighted)
  - Recall (Weighted)
  - F1 (Weighted)
  - F1 per class [0, 1, 2, 3]
  - AUC (Weighted, OVR)
  - MCC (Matthews Correlation Coefficient)

Also generates confusion matrix visualization.

Usage:
    python scripts/evaluate.py --model_path models/best_model.pt --test_file data/test.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BugSeverityDataset, EXPERIMENT_PRESETS
from model import CodeBERTClassifier
from kicl_model import KICLModel


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate bug severity classifier')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl')
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--arch', type=str, default='codebert',
                        choices=['codebert', 'kicl'],
                        help="Model architecture. 'kicl' is auto-selected when the "
                             "checkpoint is a KICL model or --experiment is given.")
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['A', 'B', 'C'],
                        help='BugsRepo experiment preset (sets text fields + fusion).')
    parser.add_argument('--fusion_type', type=str, default='none',
                        choices=['none', 'concat10', 'metric_encoder64'])
    parser.add_argument('--num_metrics', type=int, default=10)
    return parser.parse_args()


def get_predictions(model, dataloader, device, use_metrics=False):
    """Run inference and collect all predictions, labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']

            if use_metrics:
                metrics = batch['metrics'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                metrics=metrics)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
            probs = outputs['probs'].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob, label_names=None):
    """Compute all required evaluation metrics."""
    if label_names is None:
        label_names = ['Critical (0)', 'Major (1)', 'Medium (2)', 'Minor (3)']

    metrics = {}

    # Overall accuracy
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))

    # Weighted metrics
    metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = {f'class_{i}': float(f1_per_class[i]) for i in range(len(f1_per_class))}

    # AUC (weighted, one-vs-rest)
    try:
        metrics['auc_weighted'] = float(roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr'))
    except ValueError as e:
        print(f'Warning: Could not compute AUC: {e}')
        metrics['auc_weighted'] = None

    # MCC (Matthews Correlation Coefficient)
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0, output_dict=True
    )

    return metrics


def plot_confusion_matrix(cm, label_names, save_path):
    """Save confusion matrix as a PNG image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=label_names,
            yticklabels=label_names,
            ylabel='True Label',
            xlabel='Predicted Label',
            title='Bug Severity Prediction — Confusion Matrix',
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=14)

        fig.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Confusion matrix saved to {save_path}')
    except ImportError:
        print('Warning: matplotlib not available, skipping confusion matrix plot')


def print_results(metrics):
    """Print evaluation results in a formatted table."""
    print('\n' + '=' * 60)
    print('  BUG SEVERITY PREDICTION — EVALUATION RESULTS')
    print('=' * 60)

    print(f'\n  Accuracy:              {metrics["accuracy"]:.4f}')
    print(f'  Precision (Weighted):  {metrics["precision_weighted"]:.4f}')
    print(f'  Recall (Weighted):     {metrics["recall_weighted"]:.4f}')
    print(f'  F1 (Weighted):         {metrics["f1_weighted"]:.4f}')

    print(f'\n  F1 per Class:')
    for cls, f1 in metrics['f1_per_class'].items():
        print(f'    {cls}: {f1:.4f}')

    auc = metrics.get('auc_weighted')
    print(f'\n  AUC (Weighted):        {auc:.4f}' if auc is not None else '\n  AUC (Weighted):        N/A')
    print(f'  MCC:                   {metrics["mcc"]:.4f}')

    print('\n  Confusion Matrix:')
    cm = np.array(metrics['confusion_matrix'])
    labels = ['Crit', 'Major', 'Med', 'Minor']
    print(f'          {"  ".join(f"{l:>6}" for l in labels)}')
    for i, row in enumerate(cm):
        print(f'  {labels[i]:>5}  {"  ".join(f"{v:>6}" for v in row)}')

    print('\n' + '=' * 60)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load checkpoint first so it can describe its own architecture/config.
    print(f'Loading model from {args.model_path}...')
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Resolve inference config. Precedence: --experiment > checkpoint metadata > CLI flags.
    if args.experiment is not None:
        preset = EXPERIMENT_PRESETS[args.experiment]
        text_fields = preset['text_fields']
        metrics_keys = preset['metric_keys']
        fusion_type = preset['fusion_type']
        num_metrics = len(metrics_keys) if metrics_keys else args.num_metrics
    else:
        text_fields = checkpoint.get('text_fields')
        metrics_keys = checkpoint.get('metrics_keys')
        fusion_type = checkpoint.get('fusion_type', args.fusion_type)
        num_metrics = checkpoint.get('num_metrics', args.num_metrics)

    # A KICL checkpoint carries a 'classifier.*' head; CodeBERT carries 'out_layer.*'.
    is_kicl = (args.arch == 'kicl' or args.experiment is not None
               or fusion_type != 'none'
               or any(k.startswith('classifier.') for k in state_dict))
    use_metrics = is_kicl and fusion_type != 'none'

    # Load test dataset with the matching text fields / metric keys.
    print(f'Loading test data from {args.test_file}...')
    test_dataset = BugSeverityDataset(
        args.test_file, tokenizer, args.max_length,
        text_fields=text_fields, metrics_keys=metrics_keys,
    )

    # Re-apply the training StandardScaler (persisted in the checkpoint) to metrics.
    if use_metrics and checkpoint.get('scaler_mean') is not None:
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_mean'], dtype=float)
        scaler.scale_ = np.array(checkpoint['scaler_scale'], dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        test_dataset.apply_scaler(scaler, fit=False)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'  Test samples: {len(test_dataset)}')

    # Build the matching model architecture.
    if is_kicl:
        print(f'  Architecture: KICL (fusion_type={fusion_type}, num_metrics={num_metrics})')
        model = KICLModel(model_name=args.model_name, num_labels=args.num_labels,
                          fusion_type=fusion_type, num_metrics=num_metrics)
    else:
        print('  Architecture: CodeBERTClassifier')
        model = CodeBERTClassifier(model_name=args.model_name, num_labels=args.num_labels)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    val_metric = checkpoint.get('val_loss', checkpoint.get('metric'))
    val_str = f'{val_metric:.4f}' if isinstance(val_metric, (int, float)) else str(val_metric)
    print(f'  Loaded from epoch {checkpoint.get("epoch", "?")} (val_metric={val_str})')

    # Get predictions
    y_true, y_pred, y_prob = get_predictions(model, test_loader, device, use_metrics=use_metrics)

    # Compute metrics
    label_names = ['Critical (0)', 'Major (1)', 'Medium (2)', 'Minor (3)']
    metrics = compute_metrics(y_true, y_pred, y_prob, label_names)

    # Print results
    print_results(metrics)

    # Save results
    results_path = os.path.join(args.output_dir, 'baseline_results.json')
    serializable_metrics = {k: v for k, v in metrics.items() if k != 'classification_report'}
    serializable_metrics['classification_report_text'] = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    with open(results_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f'\nResults saved to {results_path}')

    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_names, cm_path)


if __name__ == '__main__':
    from sklearn.metrics import classification_report
    main()
