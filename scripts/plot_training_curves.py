"""
Plot training curves from training history JSON.

Generates a multi-panel plot showing:
  - Train/Validation loss over epochs
  - Validation accuracy over epochs
  - Number of predicted classes over epochs

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --history models/training_history.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def parse_args():
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--history', type=str, default='models/training_history.json',
                        help='Path to training history JSON')
    parser.add_argument('--output', type=str, default='results/training_curves.png',
                        help='Output image path')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.history, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    classes_predicted = [h.get('classes_predicted', 0) for h in history]

    # Style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Bug Severity Prediction — Training Curves', fontsize=16, fontweight='bold', y=1.02)

    colors = {
        'train': '#2196F3',
        'val': '#FF5722',
        'classes': '#4CAF50',
    }

    # Panel 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'o-', color=colors['train'], linewidth=2, markersize=6, label='Train Loss')
    ax1.plot(epochs, val_loss, 's-', color=colors['val'], linewidth=2, markersize=6, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Panel 2: Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, train_acc, 'o-', color=colors['train'], linewidth=2, markersize=6, label='Train Accuracy')
    ax2.plot(epochs, val_acc, 's-', color=colors['val'], linewidth=2, markersize=6, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Panel 3: Classes predicted
    ax3 = axes[2]
    bar_colors = ['#FF9800' if c < 4 else '#4CAF50' for c in classes_predicted]
    ax3.bar(epochs, classes_predicted, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax3.axhline(y=4, color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (4 classes)')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Classes Predicted', fontsize=12)
    ax3.set_title('Prediction Diversity', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 5)
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax3.legend(fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Training curves saved to {args.output}')


if __name__ == '__main__':
    main()
