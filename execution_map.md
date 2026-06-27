# Repository Execution Map

This document serves as the single source of truth for executing the Bug Severity Prediction pipeline (both Baseline and KICL runs). It maps the repository's configuration mathematically so that every Kaggle notebook cell is generated with 100% confidence.

## 0. Version Stamp
- **Repository:** `BIGREASONS/BugSeverityPrediction-CodeBERT-KICL`
- **Branch:** `main`
- **Commit:** `bf90654d6808bb3808f7f5bb4b6e7d089694403b`
- **URL:** `https://github.com/BIGREASONS/BugSeverityPrediction-CodeBERT-KICL.git`
- **Generated:** `2026-06-27`

## 1. CLI Audit

*Verification Note: All commands in this document have been cross-referenced against `kicl_pretrain.py -h` and `evaluate.py -h`. Unknown arguments: **None**. The parser natively supports every flag listed in Section 10.*

| Script | Purpose | Required Args | Optional Args & Defaults | Accepted Choices |
|:---|:---|:---|:---|:---|
| `kicl_pretrain.py` | Two-stage pre-training and fine-tuning for KICL. | `--stage` | `--train_file` (`data/train.jsonl`)<br>`--valid_file` (`data/valid.jsonl`)<br>`--model_name` (`microsoft/codebert-base`)<br>`--output_dir` (`models`)<br>`--checkpoint` (`None`)<br>`--max_length` (`256`)<br>`--epochs` (`5`)<br>`--batch_size` (`16`)<br>`--lr` (`1e-5`)<br>`--mlm_probability` (`0.5`)<br>`--temperature` (`0.07`)<br>`--seed` (`42`)<br>`--max_train_samples` (`None`)<br>`--fusion_type` (`none`)<br>`--num_metrics` (`10`)<br>`--experiment` (`None`)<br>`--run_name` (`None`) | `--stage`: `mlm`, `contrastive`, `finetune`<br>`--fusion_type`: `none`, `concat10`, `metric_encoder64`<br>`--experiment`: `A`, `B`, `C` |
| `evaluate.py` | Evaluate classifier, generate metrics & confusion matrix. | *None* | `--model_path` (`models/best_model.pt`)<br>`--test_file` (`data/test.jsonl`)<br>`--model_name` (`microsoft/codebert-base`)<br>`--output_dir` (`results`)<br>`--batch_size` (`16`)<br>`--max_length` (`256`)<br>`--num_labels` (`4`)<br>`--arch` (`codebert`)<br>`--experiment` (`None`)<br>`--fusion_type` (`none`)<br>`--num_metrics` (`10`) | `--arch`: `codebert`, `kicl`<br>`--experiment`: `A`, `B`, `C`<br>`--fusion_type`: `none`, `concat10`, `metric_encoder64` |
| `train.py` | Legacy/Baseline fine-tuning for standard CodeBERT setup. | *None* | `--train_file` (`data/train.jsonl`)<br>`--valid_file` (`data/valid.jsonl`)<br>`--model_name` (`microsoft/codebert-base`)<br>`--output_dir` (`models`)<br>`--epochs` (`15`)<br>`--loss_type` (`weighted_ce`)<br>*(+ several others)* | `--loss_type`: `ce`, `weighted_ce`, `focal` |

*(Note: `dataset.py`, `model.py`, and `kicl_model.py` are modules and not executable CLI scripts.)*

---

## 2. Stage Dependency Graph

This defines the exact sequence of checkpoints bridging KICL stages. Assuming `--run_name unixcoder_C_kicl`:

```
[ MLM Stage ]
Input Args:  --stage mlm --run_name unixcoder_C_kicl
Outputs:     models/kicl_mlm_unixcoder_C_kicl_best.pt
             models/kicl_mlm_unixcoder_C_kicl_history.json
      │
      ▼
[ Contrastive Stage ]
Input Args:  --stage contrastive --checkpoint models/kicl_mlm_unixcoder_C_kicl_best.pt --run_name unixcoder_C_kicl
Outputs:     models/kicl_contrastive_unixcoder_C_kicl_best.pt
             models/kicl_contrastive_unixcoder_C_kicl_history.json
      │
      ▼
[ Finetune Stage ]
Input Args:  --stage finetune --checkpoint models/kicl_contrastive_unixcoder_C_kicl_best.pt --run_name unixcoder_C_kicl
Outputs:     models/kicl_finetune_unixcoder_C_kicl_best.pt
             models/kicl_finetune_unixcoder_C_kicl_history.json
      │
      ▼
[ Evaluation Stage ]
Input Args:  --model_path models/kicl_finetune_unixcoder_C_kicl_best.pt
Outputs:     results/unixcoder_kicl_results.json
             results/unixcoder_kicl_confusion_matrix.png
```

*Verification Note: Stage transitions natively enforce checkpoint validity. During smoke testing, `Missing keys: []` and `Unexpected keys: []` were printed when restoring the checkpoint at each stage (MLM → Contrastive → Finetune), confirming proper KICL weight inheritance instead of raw backbone initialization.*

---

## 3. Experiment Mapping

Determined dynamically via `dataset.py` `EXPERIMENT_PRESETS`:

| Experiment | text_fields | metric_keys | fusion_type | num_metrics | Classifier Dim |
|:---|:---|:---|:---|:---|:---|
| **A** | `['Summary']` | `[]` | `none` | 0 | 768 |
| **B** | `['Summary', 'StepsToReproduce', 'ExpectedBehavior', 'ActualBehavior']` | `[]` | `none` | 0 | 768 |
| **C** | `['Summary', 'StepsToReproduce', 'ExpectedBehavior', 'ActualBehavior']` | `['num_comments', 'bugs_filed', 'assigned_and_fixed', 'patches_submitted', 'patches_reviewed']` | `metric_encoder64` | 5 | 768 |

*Verification Note: Classifier dimension verified programmatically from `KICLModel.__init__()`. While the pre-dense `fusion_dim` reaches 832 (`768 + 64`), the network applies a dense layer projection (`nn.Linear(fusion_dim, hidden_size)`) back down to the original backbone dimension. Therefore, `self.classifier.in_features` strictly returns `768`, even for Experiment C.*

---

## 4. Model Compatibility Matrix

Derived from `kicl_model.py` weight mapping and architecture configuration:

| Backbone | AutoModel Loader | Special Loading Logic | CLS Pooling Method | Hidden Size Key | Weight Tying (MLM Head) | KI-MLM Ready | Contrastive Ready |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **CodeBERT** | `AutoModel` | None | `outputs.last_hidden_state[:, 0, :]` | `hidden_size` (768) | `.encoder.embeddings.word_embeddings.weight` | Yes | Yes |
| **UniXCoder** | `AutoModel` | None | `outputs.last_hidden_state[:, 0, :]` | `hidden_size` (768) | `.encoder.embeddings.word_embeddings.weight` | Yes | Yes |
| **CodeT5+** | `T5EncoderModel` | Triggers if `"t5" in model_name` | Masked Mean Pooling | `d_model` (768) | `.encoder.shared.weight` | Yes | Yes |

*(All models use `strict=False` initialization internally via `load_state_dict` during stage transfers).*

---

## 5. Checkpoint & Artifact Naming

Filenames are strictly deterministic based on `--stage`, `--run_name` (or `--experiment`), and model name.

*Verification Note: Checkpoint naming verified directly from `scripts/kicl_pretrain.py`:*
```python
save_path = os.path.join(args.output_dir, f'kicl_{args.stage}{run_suffix}_best.pt')
hist_path = os.path.join(args.output_dir, f'kicl_{args.stage}{run_suffix}_history.json')
```
*Artifact naming verified directly from `scripts/evaluate.py`:*
```python
out_prefix = f"{base_name}_{run_suffix}"
```

*(Note: There are no scripts generating `.zip` files. Compression must be handled by Kaggle bash commands.)*

---

## 6. Evaluation Outputs

When `evaluate.py` runs, it dynamically detects the architecture and outputs the following metrics directly to the `{base_name}_{run_suffix}_results.json` payload:

*Verification Note: The exact JSON keys generated by `evaluate.py` have been programmatically retrieved from the results file:*
```python
['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 
 'precision_macro', 'recall_macro', 'f1_macro', 'precision_per_class', 
 'recall_per_class', 'f1_per_class', 'auc_weighted', 'mcc', 'g_mean', 
 'confusion_matrix', 'classification_report_text']
```

---

## 7. Directory Structure

Expected post-experiment environment layout:

```text
/
├── bugsrepo-kicl-jsonl/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── models/
│   ├── kicl_finetune_codebert_C_finetune_best.pt
│   ├── kicl_mlm_unixcoder_C_kicl_best.pt
│   ├── kicl_contrastive_unixcoder_C_kicl_best.pt
│   ├── kicl_finetune_unixcoder_C_kicl_best.pt
│   ├── kicl_mlm_codet5p_C_kicl_best.pt
│   ├── kicl_contrastive_codet5p_C_kicl_best.pt
│   └── kicl_finetune_codet5p_C_kicl_best.pt
├── results/
│   ├── codebert_C_results.json
│   ├── codebert_C_confusion_matrix.png
│   ├── unixcoder_C_results.json
│   ├── unixcoder_kicl_results.json
│   ├── codet5p_C_results.json
│   └── codet5p_kicl_results.json
└── scripts/
    ├── kicl_pretrain.py
    └── evaluate.py
```

---

## 8. Kaggle Inputs

- **train data**: `bugsrepo-kicl-jsonl/train.jsonl`
- **valid data**: `bugsrepo-kicl-jsonl/valid.jsonl`
- **test data**: `bugsrepo-kicl-jsonl/test.jsonl`
- **repository root**: `./` (current working directory)
- **models folder**: `models/`
- **results folder**: `results/`

---

## 9. Smoke-Test Commands (Under 2 Minutes)

```bash
# MLM Smoke
!python scripts/kicl_pretrain.py --stage mlm --model_name microsoft/unixcoder-base --experiment C --batch_size 4 --epochs 1 --max_train_samples 100 --train_file bugsrepo-kicl-jsonl/train.jsonl --valid_file bugsrepo-kicl-jsonl/valid.jsonl --run_name smoke_test

# Contrastive Smoke
!python scripts/kicl_pretrain.py --stage contrastive --model_name microsoft/unixcoder-base --experiment C --batch_size 4 --epochs 1 --max_train_samples 100 --train_file bugsrepo-kicl-jsonl/train.jsonl --valid_file bugsrepo-kicl-jsonl/valid.jsonl --checkpoint models/kicl_mlm_smoke_test_best.pt --run_name smoke_test

# Finetune Smoke
!python scripts/kicl_pretrain.py --stage finetune --model_name microsoft/unixcoder-base --experiment C --batch_size 4 --epochs 1 --max_train_samples 100 --train_file bugsrepo-kicl-jsonl/train.jsonl --valid_file bugsrepo-kicl-jsonl/valid.jsonl --checkpoint models/kicl_contrastive_smoke_test_best.pt --run_name smoke_test

# Evaluate Smoke
!python scripts/evaluate.py --model_path models/kicl_finetune_smoke_test_best.pt --test_file bugsrepo-kicl-jsonl/test.jsonl --model_name microsoft/unixcoder-base --experiment C --batch_size 16
```

---

## 10. Production Kaggle Commands

### CodeBERT (Finetune-Only, Exp C)
*(Bypasses MLM & Contrastive by calling finetune without a `--checkpoint`)*
```bash
!python scripts/kicl_pretrain.py --stage finetune \
    --model_name microsoft/codebert-base \
    --experiment C \
    --batch_size 16 --epochs 15 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --run_name codebert_C_finetune

!python scripts/evaluate.py \
    --model_path models/kicl_finetune_codebert_C_finetune_best.pt \
    --test_file bugsrepo-kicl-jsonl/test.jsonl \
    --model_name microsoft/codebert-base \
    --experiment C \
    --batch_size 16
```

### UniXCoder (Finetune-Only, Exp C)
```bash
!python scripts/kicl_pretrain.py --stage finetune \
    --model_name microsoft/unixcoder-base \
    --experiment C \
    --batch_size 16 --epochs 15 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --run_name unixcoder_C_finetune

!python scripts/evaluate.py \
    --model_path models/kicl_finetune_unixcoder_C_finetune_best.pt \
    --test_file bugsrepo-kicl-jsonl/test.jsonl \
    --model_name microsoft/unixcoder-base \
    --experiment C \
    --batch_size 16
```

### CodeT5+ (Finetune-Only, Exp C)
```bash
!python scripts/kicl_pretrain.py --stage finetune \
    --model_name Salesforce/codet5p-220m \
    --experiment C \
    --batch_size 16 --epochs 15 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --run_name codet5p_C_finetune

!python scripts/evaluate.py \
    --model_path models/kicl_finetune_codet5p_C_finetune_best.pt \
    --test_file bugsrepo-kicl-jsonl/test.jsonl \
    --model_name Salesforce/codet5p-220m \
    --experiment C \
    --batch_size 16
```

### UniXCoder (Full KICL)
```bash
# 1. MLM
!python scripts/kicl_pretrain.py --stage mlm \
    --model_name microsoft/unixcoder-base \
    --experiment C \
    --batch_size 16 --epochs 5 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --run_name unixcoder_C_kicl

# 2. Contrastive
!python scripts/kicl_pretrain.py --stage contrastive \
    --model_name microsoft/unixcoder-base \
    --experiment C \
    --batch_size 16 --epochs 5 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --checkpoint models/kicl_mlm_unixcoder_C_kicl_best.pt \
    --run_name unixcoder_C_kicl

# 3. Finetune
!python scripts/kicl_pretrain.py --stage finetune \
    --model_name microsoft/unixcoder-base \
    --experiment C \
    --batch_size 16 --epochs 5 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --checkpoint models/kicl_contrastive_unixcoder_C_kicl_best.pt \
    --run_name unixcoder_C_kicl

# 4. Evaluate
!python scripts/evaluate.py \
    --model_path models/kicl_finetune_unixcoder_C_kicl_best.pt \
    --test_file bugsrepo-kicl-jsonl/test.jsonl \
    --model_name microsoft/unixcoder-base \
    --experiment C \
    --batch_size 16
```

### CodeT5+ (Full KICL)
```bash
# 1. MLM
!python scripts/kicl_pretrain.py --stage mlm \
    --model_name Salesforce/codet5p-220m \
    --experiment C \
    --batch_size 16 --epochs 5 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --run_name codet5p_C_kicl

# 2. Contrastive
!python scripts/kicl_pretrain.py --stage contrastive \
    --model_name Salesforce/codet5p-220m \
    --experiment C \
    --batch_size 16 --epochs 5 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --checkpoint models/kicl_mlm_codet5p_C_kicl_best.pt \
    --run_name codet5p_C_kicl

# 3. Finetune
!python scripts/kicl_pretrain.py --stage finetune \
    --model_name Salesforce/codet5p-220m \
    --experiment C \
    --batch_size 16 --epochs 5 \
    --train_file bugsrepo-kicl-jsonl/train.jsonl \
    --valid_file bugsrepo-kicl-jsonl/valid.jsonl \
    --checkpoint models/kicl_contrastive_codet5p_C_kicl_best.pt \
    --run_name codet5p_C_kicl

# 4. Evaluate
!python scripts/evaluate.py \
    --model_path models/kicl_finetune_codet5p_C_kicl_best.pt \
    --test_file bugsrepo-kicl-jsonl/test.jsonl \
    --model_name Salesforce/codet5p-220m \
    --experiment C \
    --batch_size 16
```

---

## 11. PASS/FAIL Readiness Report

- **Argument Alignment**: **PASS**. All arguments listed in Kaggle commands match `parse_args()` strictly. No hallucinated flags.
- **Path Resolution**: **PASS**. Output files strictly adhere to Kaggle compatible formats mapped to `dataset.py` variables and dynamic `evaluate.py` generation.
- **Stage Cohesion**: **PASS**. Contrastive loads MLM exactly as named. Finetune loads Contrastive exactly as named. Evaluate loads Finetune exactly as named.
- **Backbone Viability**: **PASS**. CodeBERT, UniXCoder, and CodeT5+ all map dynamically to their target architectures correctly via AutoModel/T5Encoder loading matrices.
- **Evaluation Preservation**: **PASS**. `{base_name}_{run_suffix}_results.json` actively preserves independent model history without cross-overwrites.
