"""
CodeBERT-based model for bug severity classification.
Architecture follows the ISSRE 2023 paper:
  CodeBERT encoder -> [CLS] token -> dropout -> dense -> tanh -> dropout -> 4-class output

Supports:
  - Weighted CrossEntropyLoss (inverse-frequency class weights)
  - Focal Loss (for hard-example mining in imbalanced datasets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for imbalanced classification.
    Down-weights easy examples, focusing training on hard misclassified samples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight   # class weights (tensor)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CodeBERTClassifier(nn.Module):
    """CodeBERT sequence classifier for bug severity prediction (4 classes)."""

    def __init__(self, model_name='microsoft/codebert-base', num_labels=4,
                 dropout_rate=0.1, class_weights=None, loss_type='weighted_ce'):
        """
        Args:
            model_name: HuggingFace model name.
            num_labels: Number of severity classes.
            dropout_rate: Dropout probability.
            class_weights: Tensor of shape (num_labels,) with class weights.
                           If None, uses uniform weights.
            loss_type: 'ce' (plain), 'weighted_ce', or 'focal'.
        """
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size  # 768 for codebert-base

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, num_labels)

        # Loss function selection
        self.loss_type = loss_type
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        if loss_type == 'focal':
            self.loss_fct = FocalLoss(weight=self.class_weights, gamma=2.0)
        elif loss_type == 'weighted_ce':
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) optional, if provided returns loss

        Returns:
            dict with 'loss' (if labels provided), 'logits', 'probs'
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Take [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Classification head
        x = self.dropout(cls_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_layer(x)

        probs = torch.softmax(logits, dim=-1)

        result = {'logits': logits, 'probs': probs}

        if labels is not None:
            result['loss'] = self.loss_fct(logits, labels)

        return result
