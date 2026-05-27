"""
Compare results between baseline CodeBERT and KICL-enhanced models.
Generates comparison tables and charts.

Usage:
    python scripts/compare_results.py
"""

import json
import os
import sys

import numpy as np


def load_results(path):
    """Load results from JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def format_table(baseline, kicl=None):
    """Format comparison table."""
    header = f'{"Metric":<25} {"Baseline":>12}'
    if kicl:
        header += f' {"KICL":>12} {"Δ":>10}'
    separator = '-' * len(header)

    rows = [separator, header, separator]

    metrics = [
        ('Precision (Weighted)', 'precision_weighted'),
        ('Recall (Weighted)', 'recall_weighted'),
        ('F1 (Weighted)', 'f1_weighted'),
        ('AUC (Weighted)', 'auc_weighted'),
        ('MCC', 'mcc'),
    ]

    for name, key in metrics:
        b_val = baseline.get(key, 0) or 0
        row = f'{name:<25} {b_val:>12.4f}'
        if kicl:
            k_val = kicl.get(key, 0) or 0
            delta = k_val - b_val
            sign = '+' if delta >= 0 else ''
            row += f' {k_val:>12.4f} {sign}{delta:>9.4f}'
        rows.append(row)

    # Per-class F1
    rows.append(separator)
    rows.append('F1 per Class:')
    for cls_idx in range(4):
        key = f'class_{cls_idx}'
        cls_names = {0: 'Critical', 1: 'Major', 2: 'Medium', 3: 'Minor'}
        name = f'  F1 {cls_names[cls_idx]} ({cls_idx})'
        b_val = baseline.get('f1_per_class', {}).get(key, 0)
        row = f'{name:<25} {b_val:>12.4f}'
        if kicl:
            k_val = kicl.get('f1_per_class', {}).get(key, 0)
            delta = k_val - b_val
            sign = '+' if delta >= 0 else ''
            row += f' {k_val:>12.4f} {sign}{delta:>9.4f}'
        rows.append(row)

    rows.append(separator)
    return '\n'.join(rows)


def plot_comparison(baseline, kicl, save_path):
    """Generate comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metrics = ['Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1\n(Weighted)', 'AUC\n(Weighted)', 'MCC']
        keys = ['precision_weighted', 'recall_weighted', 'f1_weighted', 'auc_weighted', 'mcc']

        baseline_vals = [baseline.get(k, 0) or 0 for k in keys]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline CodeBERT',
                       color='#4C72B0', alpha=0.85)

        if kicl:
            kicl_vals = [kicl.get(k, 0) or 0 for k in keys]
            bars2 = ax.bar(x + width/2, kicl_vals, width, label='KICL-Enhanced',
                          color='#DD8452', alpha=0.85)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Bug Severity Prediction — Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                       fontsize=9)
        if kicl:
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                           fontsize=9)

        fig.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Comparison chart saved to {save_path}')
    except ImportError:
        print('Warning: matplotlib not available, skipping chart generation')


def main():
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    baseline = load_results(os.path.join(results_dir, 'baseline_results.json'))
    kicl = load_results(os.path.join(results_dir, 'kicl_results.json'))

    if baseline is None:
        print('ERROR: No baseline results found. Run evaluate.py first.')
        sys.exit(1)

    print('\n' + '=' * 60)
    print('  BUG SEVERITY PREDICTION — RESULTS COMPARISON')
    print('=' * 60 + '\n')

    table = format_table(baseline, kicl)
    print(table)

    # Save comparison report
    report_path = os.path.join(results_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write('BUG SEVERITY PREDICTION — RESULTS COMPARISON\n')
        f.write('=' * 60 + '\n\n')
        f.write(table + '\n')

        if baseline.get('classification_report_text'):
            f.write('\n\nBaseline Classification Report:\n')
            f.write(baseline['classification_report_text'])

        if kicl and kicl.get('classification_report_text'):
            f.write('\n\nKICL Classification Report:\n')
            f.write(kicl['classification_report_text'])

    print(f'\nReport saved to {report_path}')

    # Generate chart
    chart_path = os.path.join(results_dir, 'comparison_chart.png')
    plot_comparison(baseline, kicl, chart_path)


if __name__ == '__main__':
    main()
