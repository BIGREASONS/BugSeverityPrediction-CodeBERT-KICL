"""
KICL (Knowledge-Intensified Contrastive Learning) model for bug severity prediction.

Extends CodeBERT with:
  1. MLM head for Knowledge-Intensified pre-training (50% masking)
  2. Projection head for contrastive learning
  3. Classification head for severity prediction

Reference: Wei et al., "Improving Bug Severity Prediction With
Domain-Specific Representation Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class KICLModel(nn.Module):
    """KICL model combining CodeBERT with domain-specific pre-training heads."""

    def __init__(self, model_name='microsoft/codebert-base', num_labels=4,
                 dropout_rate=0.1, projection_dim=128, temperature=0.07,
                 class_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.temperature = temperature
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Store class weights for weighted CE in finetune
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None


        hidden_size = self.config.hidden_size  # 768

        # MLM head for Knowledge-Intensified pre-training
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.config.vocab_size),
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

        # Classification head (same architecture as baseline)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def encode(self, input_ids, attention_mask):
        """Get encoder outputs."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    def get_cls_embedding(self, input_ids, attention_mask):
        """Get [CLS] token embedding."""
        outputs = self.encode(input_ids, attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def forward_mlm(self, input_ids, attention_mask, mlm_labels=None):
        """Forward pass for masked language model pre-training."""
        outputs = self.encode(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.mlm_head(sequence_output)

        result = {'mlm_logits': prediction_scores}

        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            result['mlm_loss'] = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                mlm_labels.view(-1)
            )

        return result

    def forward_contrastive(self, input_ids, attention_mask, labels):
        """
        Forward pass for supervised contrastive learning.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) severity labels for creating pos/neg pairs

        Returns:
            dict with 'contrastive_loss' and 'projections'
        """
        cls_embedding = self.get_cls_embedding(input_ids, attention_mask)
        projections = self.projection_head(cls_embedding)
        projections = F.normalize(projections, dim=-1)

        # Supervised contrastive loss (SupCon)
        contrastive_loss = self.supervised_contrastive_loss(projections, labels)

        return {
            'contrastive_loss': contrastive_loss,
            'projections': projections,
        }

    def supervised_contrastive_loss(self, features, labels):
        """
        Supervised Contrastive Loss (SupCon).
        Positive pairs: samples with the same severity label.
        Negative pairs: samples with different severity labels.

        Args:
            features: (batch_size, projection_dim) L2-normalized projections
            labels: (batch_size,) severity labels
        """
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask for positive pairs (same label, not self)
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()
        mask_self = torch.eye(batch_size, device=features.device)
        mask_pos = mask_pos - mask_self

        # If no positive pairs exist, return 0 loss
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Log-softmax over all negatives + positives (excluding self)
        logits_mask = 1.0 - mask_self
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-probability over positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        return loss

    def forward_classify(self, input_ids, attention_mask, labels=None):
        """Forward pass for classification (fine-tuning stage)."""
        cls_embedding = self.get_cls_embedding(input_ids, attention_mask)

        x = self.dropout(cls_embedding)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)

        result = {'logits': logits, 'probs': probs}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            result['loss'] = loss_fct(logits, labels)

        return result

    def forward(self, input_ids, attention_mask, labels=None):
        """Default forward pass = classification."""
        return self.forward_classify(input_ids, attention_mask, labels)
