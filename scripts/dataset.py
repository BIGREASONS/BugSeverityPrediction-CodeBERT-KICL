"""
PyTorch Dataset for bug severity prediction.
Loads JSONL data and tokenizes text using the CodeBERT tokenizer.

Backward compatible with the original ISSRE-2023 method-source dataset
(field 'code' + 10 code-complexity metrics) and extended to support
BugsRepo bug-report fields for Experiments A / B / C.
"""

import json
import torch
from torch.utils.data import Dataset


# --- Legacy (ISSRE 2023 method-source) configuration -------------------------
LEGACY_TEXT_FIELDS = ['code']
LEGACY_METRIC_KEYS = ['lc', 'pi', 'ma', 'nbd', 'ml', 'd', 'mi', 'fo', 'r', 'e']

# --- BugsRepo configuration --------------------------------------------------
BUGSREPO_TEXT_FIELDS = ['Summary', 'StepsToReproduce', 'ExpectedBehavior', 'ActualBehavior']
BUGSREPO_METRIC_KEYS = ['num_comments', 'bugs_filed', 'assigned_and_fixed',
                        'patches_submitted', 'patches_reviewed']

# Experiment presets: text fields concatenated with [SEP], metric keys, fusion.
EXPERIMENT_PRESETS = {
    'A': {'text_fields': ['Summary'],
          'metric_keys': [],
          'fusion_type': 'none'},
    'B': {'text_fields': BUGSREPO_TEXT_FIELDS,
          'metric_keys': [],
          'fusion_type': 'none'},
    'C': {'text_fields': BUGSREPO_TEXT_FIELDS,
          'metric_keys': BUGSREPO_METRIC_KEYS,
          'fusion_type': 'metric_encoder64'},
    'legacy': {'text_fields': LEGACY_TEXT_FIELDS,
               'metric_keys': LEGACY_METRIC_KEYS,
               'fusion_type': 'none'},
}


class BugSeverityDataset(Dataset):
    """Dataset for bug severity prediction (method source code or bug reports)."""

    def __init__(self, filepath, tokenizer, max_length=256, max_samples=None,
                 text_fields=None, metrics_keys=None):
        """
        Args:
            filepath: Path to JSONL file with text field(s), 'label', optional metrics.
            tokenizer: HuggingFace tokenizer (e.g., CodeBERT).
            max_length: Maximum token sequence length.
            max_samples: If set, only load this many samples (for debugging).
            text_fields: Ordered list of fields concatenated with [SEP] to form the
                         input sequence. Defaults to ['code'] (legacy behavior).
            metrics_keys: Ordered list of numeric metadata keys. Defaults to the 10
                          legacy code-complexity metrics. Pass [] for text-only runs.
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_fields = list(text_fields) if text_fields is not None else list(LEGACY_TEXT_FIELDS)
        self.metrics_keys = list(metrics_keys) if metrics_keys is not None else list(LEGACY_METRIC_KEYS)

        sep = f' {tokenizer.sep_token} ' if getattr(tokenizer, 'sep_token', None) else ' '

        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Build the text sequence from the configured fields, skipping
                # any that are missing or empty (handles BugsRepo gaps safely).
                parts = []
                for field in self.text_fields:
                    val = data.get(field)
                    if val is None:
                        continue
                    val = str(val).strip()
                    if val:
                        parts.append(val)
                text = sep.join(parts)

                # Backward compatibility: fall back to legacy 'code' if the
                # configured fields produced nothing but a 'code' field exists.
                if not text and data.get('code') is not None:
                    text = str(data['code'])

                # Extract metrics (default to 0.0 if missing/empty/None).
                metrics = [float(data.get(k) or 0.0) for k in self.metrics_keys]

                self.samples.append({
                    'text': text,
                    'label': int(data['label']),
                    'metrics': metrics,
                })

    def apply_scaler(self, scaler, fit=False):
        """
        Applies a StandardScaler to the metrics.
        If fit is True, fits the scaler on this dataset's metrics.
        No-op when there are no metric columns (text-only experiments).
        """
        import numpy as np
        all_metrics = np.array([s['metrics'] for s in self.samples], dtype=float)
        if all_metrics.size == 0 or all_metrics.shape[1] == 0:
            return scaler
        if fit:
            scaler.fit(all_metrics)
        scaled_metrics = scaler.transform(all_metrics)
        for i, s in enumerate(self.samples):
            s['metrics'] = scaled_metrics[i].tolist()
        return scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'metrics': torch.tensor(sample['metrics'], dtype=torch.float),
        }
