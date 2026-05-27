"""
PyTorch Dataset for bug severity prediction.
Loads JSONL data and tokenizes method source code using CodeBERT tokenizer.
"""

import json
import torch
from torch.utils.data import Dataset


class BugSeverityDataset(Dataset):
    """Dataset for method-level bug severity prediction."""

    def __init__(self, filepath, tokenizer, max_length=256, max_samples=None):
        """
        Args:
            filepath: Path to JSONL file with 'code' and 'label' fields.
            tokenizer: HuggingFace tokenizer (e.g., CodeBERT).
            max_length: Maximum token sequence length.
            max_samples: If set, only load this many samples (for debugging).
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                data = json.loads(line.strip())
                self.samples.append({
                    'code': data['code'],
                    'label': int(data['label']),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }
