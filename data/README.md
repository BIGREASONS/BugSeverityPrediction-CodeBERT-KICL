# Data

This directory should contain the preprocessed JSONL dataset files.

## Required Files

| File | Description | Samples |
|------|-------------|---------|
| `train.jsonl` | Training set | 2,414 |
| `valid.jsonl` | Validation set | 426 |
| `test.jsonl` | Test set | 502 |

## Dataset Source

The dataset is derived from the **ISSRE 2023 Bug Severity Prediction** replication package, using bug reports from:

- **Defects4J** — A database of real bugs from Java projects
- **Bugs.jar** — A large-scale dataset of reproducible Java bugs

## Severity Labels

| Label | Severity | Train | Valid | Test |
|-------|----------|-------|-------|------|
| 0 | Critical | 198 | 30 | 47 |
| 1 | Major | 1,524 | 264 | 294 |
| 2 | Medium | 203 | 40 | 48 |
| 3 | Minor | 489 | 92 | 113 |

> **Note:** The dataset exhibits severe class imbalance — "Major" (label 1) accounts for ~63% of all samples.

## JSONL Format

Each line is a JSON object with the following fields:

```json
{
  "project_name": "Closure",
  "project_version": 144,
  "label": 2,
  "code": "/* full method source code */",
  "code_comment": "/* extracted comments */",
  "code_no_comment": "/* code without comments */",
  "lc": 0.545,
  "pi": -0.124,
  "ma": 0.8,
  "nbd": 0.5,
  "ml": 0.75,
  "d": 0.289,
  "mi": -0.431,
  "fo": 1.0,
  "r": -0.026,
  "e": 0.411
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `code` | Full method source code (with comments) |
| `code_comment` | Extracted code comments |
| `code_no_comment` | Method source code without comments |
| `label` | Severity label (0–3) |
| `lc`, `pi`, `ma`, ... | Normalized code complexity metrics |

## How to Obtain

1. Clone the ISSRE 2023 replication package
2. Follow their preprocessing pipeline
3. Place the resulting JSONL files in this directory
