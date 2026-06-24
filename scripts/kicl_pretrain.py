"""
KICL pre-training script: Knowledge-Intensified Contrastive Learning.

Two-stage domain-specific pre-training before fine-tuning:
  Stage 1: Knowledge-Intensified MLM (KI-MLM) — mask 50% of tokens
  Stage 2: Supervised Contrastive Learning (SupCon)

Usage:
    python scripts/kicl_pretrain.py --stage mlm --epochs 5
    python scripts/kicl_pretrain.py --stage contrastive --epochs 5
    python scripts/kicl_pretrain.py --stage finetune --epochs 5
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight as sklearn_compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BugSeverityDataset, EXPERIMENT_PRESETS
from kicl_model import KICLModel


def parse_args():
    parser = argparse.ArgumentParser(description='KICL pre-training')
    parser.add_argument('--stage', type=str, required=True,
                        choices=['mlm', 'contrastive', 'finetune'],
                        help='Pre-training stage')
    parser.add_argument('--train_file', type=str, default='data/train.jsonl')
    parser.add_argument('--valid_file', type=str, default='data/valid.jsonl')
    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--mlm_probability', type=float, default=0.5,
                        help='Masking probability for KI-MLM (paper uses 50 percent)')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--fusion_type', type=str, default='none',
                        choices=['none', 'concat10', 'metric_encoder64'],
                        help='Type of structural metric fusion')
    parser.add_argument('--num_metrics', type=int, default=10,
                        help='Number of metric features fused into the classifier '
                             '(10 = legacy code-complexity metrics; 5 = BugsRepo metadata)')
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['A', 'B', 'C'],
                        help='BugsRepo experiment preset. Overrides --fusion_type, '
                             'text fields, metric keys and --num_metrics.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Suffix for checkpoint/history filenames so runs do not '
                             'overwrite each other. Defaults to the --experiment letter.')
    return parser.parse_args()


def create_mlm_inputs(input_ids, tokenizer, mlm_probability=0.5):
    """
    Create masked inputs for KI-MLM.
    Masks 50% of tokens (vs standard 15%) to force learning project-specific patterns.
    """
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # Special tokens mask
    special_tokens = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    probability_matrix = torch.full(input_ids.shape, mlm_probability)

    for i in range(input_ids.shape[0]):
        for j in range(input_ids.shape[1]):
            if input_ids[i, j].item() in special_tokens:
                probability_matrix[i, j] = 0.0

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    masked_input[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, replace with random token
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), input_ids.shape, dtype=torch.long)
    masked_input[indices_random] = random_words[indices_random]

    # 10% of the time, keep original token
    return masked_input, labels


def train_mlm(model, dataloader, optimizer, scheduler, device, tokenizer, mlm_prob):
    """Train one epoch of KI-MLM."""
    model.train()
    total_loss = 0
    total = 0

    progress = tqdm(dataloader, desc='KI-MLM Training', leave=False)
    for batch in progress:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'].to(device)

        masked_input, mlm_labels = create_mlm_inputs(input_ids, tokenizer, mlm_prob)
        masked_input = masked_input.to(device)
        mlm_labels = mlm_labels.to(device)

        optimizer.zero_grad()
        outputs = model.forward_mlm(masked_input, attention_mask, mlm_labels)
        loss = outputs['mlm_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)
        progress.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total


def train_contrastive(model, dataloader, optimizer, scheduler, device):
    """Train one epoch of supervised contrastive learning."""
    model.train()
    total_loss = 0
    total = 0

    progress = tqdm(dataloader, desc='Contrastive Training', leave=False)
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model.forward_contrastive(input_ids, attention_mask, labels)
        loss = outputs['contrastive_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)
        progress.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / total


def train_finetune(model, dataloader, optimizer, scheduler, device):
    """Fine-tune classification head."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc='Fine-tuning', leave=False)
    for batch in progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        metrics = batch.get('metrics')
        if metrics is not None:
            metrics = metrics.to(device)

        optimizer.zero_grad()
        outputs = model.forward_classify(input_ids, attention_mask, labels, metrics=metrics)
        loss = outputs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        progress.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

    return total_loss / total, correct / total


def evaluate_finetune(model, dataloader, device):
    """Evaluate fine-tuned model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            metrics = batch.get('metrics')
            if metrics is not None:
                metrics = metrics.to(device)

            outputs = model.forward_classify(input_ids, attention_mask, labels, metrics=metrics)
            total_loss += outputs['loss'].item() * input_ids.size(0)
            preds = outputs['logits'].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    avg_loss = total_loss / total
    
    acc = accuracy_score(all_labels, all_preds)
    p_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    p_weight = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    r_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    r_weight = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weight = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'p_macro': p_macro,
        'p_weight': p_weight,
        'r_macro': r_macro,
        'r_weight': r_weight,
        'f1_macro': f1_macro,
        'f1_weight': f1_weight,
        'mcc': mcc
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Resolve a BugsRepo experiment preset (else fall back to legacy 'code' + 10 metrics).
    if args.experiment is not None:
        preset = EXPERIMENT_PRESETS[args.experiment]
        text_fields = preset['text_fields']
        metrics_keys = preset['metric_keys']
        args.fusion_type = preset['fusion_type']
        if metrics_keys:
            args.num_metrics = len(metrics_keys)
        if args.run_name is None:
            args.run_name = args.experiment
        print(f"Experiment {args.experiment}: text_fields={text_fields} | "
              f"fusion_type={args.fusion_type} | num_metrics={args.num_metrics}")
    else:
        text_fields = None      # legacy: ['code']
        metrics_keys = None     # legacy: 10 code-complexity metrics
    run_suffix = f'_{args.run_name}' if args.run_name else ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Stage: {args.stage}')

    os.makedirs(args.output_dir, exist_ok=True)

    import transformers.tokenization_utils_tokenizers
    original_add_tokens = transformers.tokenization_utils_tokenizers.PreTrainedTokenizerFast._add_tokens
    def patched_add_tokens(self, new_tokens, *args, **kwargs):
        # Prevent HuggingFace from crashing on malformed added_tokens.json
        # and prevent it from incorrectly inflating the 32100 vocab size.
        return 0
    transformers.tokenization_utils_tokenizers.PreTrainedTokenizerFast._add_tokens = patched_add_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = BugSeverityDataset(
        args.train_file, tokenizer, args.max_length, max_samples=args.max_train_samples,
        text_fields=text_fields, metrics_keys=metrics_keys,
    )
    valid_dataset = BugSeverityDataset(
        args.valid_file, tokenizer, args.max_length,
        text_fields=text_fields, metrics_keys=metrics_keys,
    )



    print(f'Train: {len(train_dataset)}, Valid: {len(valid_dataset)}')

    metric_scaler = None
    if args.fusion_type != 'none':
        print('Applying StandardScaler to metrics...')
        metric_scaler = StandardScaler()
        train_dataset.apply_scaler(metric_scaler, fit=True)
        valid_dataset.apply_scaler(metric_scaler, fit=False)

    # Create or load model
    model = KICLModel(
        model_name=args.model_name,
        num_labels=4,
        temperature=args.temperature,
        fusion_type=args.fusion_type,
        num_metrics=args.num_metrics,
    )

    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    print('Moving model to device...'); model.to(device); print('Model moved to device!')

    # For finetune stage: compute class weights and create weighted sampler
    class_weights_tensor = None
    sampler = None
    if args.stage == 'finetune':
        train_labels_np = np.array([s['label'] for s in train_dataset.samples])
        unique_classes = np.unique(train_labels_np)
        cw_partial = sklearn_compute_class_weight(
            class_weight='balanced', classes=unique_classes, y=train_labels_np
        )
        cw = np.zeros(4, dtype=np.float32)
        for c, w in zip(unique_classes, cw_partial):
            cw[c] = w
        class_weights_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
        model.class_weights = class_weights_tensor

        # Per-sample weights for sampler
        from collections import Counter as Ctr
        cc = Ctr(train_labels_np.tolist())
        csw = {cls: 1.0 / count for cls, count in cc.items()}
        sw = torch.tensor([csw[int(l)] for l in train_labels_np], dtype=torch.float)
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
        print(f'  Finetune: class_weights={[round(w, 3) for w in cw]}')
        print(f'  WeightedRandomSampler created ({len(sw)} samples)')

    print('Creating DataLoaders...'); nw = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=True
    )
    print('DataLoaders created!'); total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    print('Creating optimizer...'); optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f'\nTraining for {args.epochs} epochs ({total_steps} steps)...\n')

    history = []
    best_metric = float('-inf') if args.stage == 'finetune' else float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if args.stage == 'mlm':
            train_loss = train_mlm(model, train_loader, optimizer, scheduler,
                                    device, tokenizer, args.mlm_probability)
            print(f'Epoch {epoch}/{args.epochs} | MLM Loss: {train_loss:.4f} | '
                  f'Time: {time.time()-epoch_start:.0f}s')
            history.append({'epoch': epoch, 'mlm_loss': round(train_loss, 4)})
            metric = train_loss

        elif args.stage == 'contrastive':
            train_loss = train_contrastive(model, train_loader, optimizer, scheduler, device)
            print(f'Epoch {epoch}/{args.epochs} | Contrastive Loss: {train_loss:.4f} | '
                  f'Time: {time.time()-epoch_start:.0f}s')
            history.append({'epoch': epoch, 'contrastive_loss': round(train_loss, 4)})
            metric = train_loss

        elif args.stage == 'finetune':
            train_loss, train_acc = train_finetune(model, train_loader, optimizer, scheduler, device)
            eval_metrics = evaluate_finetune(model, valid_loader, device)
            val_loss = eval_metrics['loss']
            val_acc = eval_metrics['acc']
            print(f'Epoch {epoch}/{args.epochs} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1-Macro: {eval_metrics["f1_macro"]:.4f} MCC: {eval_metrics["mcc"]:.4f} | '
                  f'Time: {time.time()-epoch_start:.0f}s')
            history.append({
                'epoch': epoch,
                'train_loss': round(train_loss, 4),
                'train_acc': round(train_acc, 4),
                **{k: round(v, 4) for k, v in eval_metrics.items()}
            })
            metric = eval_metrics['f1_macro']

        # Save best checkpoint
        better = (metric > best_metric) if args.stage == 'finetune' else (metric < best_metric)
        if better:
            best_metric = metric
            save_path = os.path.join(args.output_dir, f'kicl_{args.stage}{run_suffix}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metric': metric,
                'args': vars(args),
                'model_name': args.model_name,
                # Self-describing config so evaluate.py can reproduce inference.
                'fusion_type': args.fusion_type,
                'num_metrics': args.num_metrics,
                'metrics_keys': train_dataset.metrics_keys,
                'text_fields': train_dataset.text_fields,
                'scaler_mean': metric_scaler.mean_.tolist() if metric_scaler is not None else None,
                'scaler_scale': metric_scaler.scale_.tolist() if metric_scaler is not None else None,
            }, save_path)
            print(f'  -> Saved best checkpoint to {save_path}')

    total_time = time.time() - start_time
    print(f'\n{args.stage} complete in {total_time:.0f}s ({total_time/60:.1f} min)')

    # Save history
    hist_path = os.path.join(args.output_dir, f'kicl_{args.stage}{run_suffix}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'History saved to {hist_path}')


if __name__ == '__main__':
    main()
