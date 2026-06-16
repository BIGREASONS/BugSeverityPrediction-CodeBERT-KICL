"""
Convert raw BugsRepo (Mozilla/Bugzilla) bug reports into the train/valid/test
JSONL format consumed by dataset.py / kicl_pretrain.py / evaluate.py.

Produces records with the fields used by all three experiments:

    Experiment A : Summary
    Experiment B : Summary [SEP] StepsToReproduce [SEP] ExpectedBehavior [SEP] ActualBehavior
    Experiment C : (B) + metadata fusion over
                   num_comments, bugs_filed, assigned_and_fixed,
                   patches_submitted, patches_reviewed

Every output line always contains all available text fields, all five metric
fields, and the integer `label`, so one set of files serves A, B and C — the
experiment is selected at train time via `kicl_pretrain.py --experiment`.

Input may be a single CSV or JSON/JSONL file (the BugsRepo structured-report
subset already joined with the per-reporter contributor metrics). Column names
are matched case-insensitively against the alias table below; edit the tables
to fit your exact export.

Usage:
    python scripts/bugsrepo_to_jsonl.py --input raw_bugsrepo.csv --outdir data
    python scripts/bugsrepo_to_jsonl.py --input raw_bugsrepo.json --outdir data \
        --train_frac 0.7 --valid_frac 0.15 --seed 42
"""

import argparse
import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Severity -> integer label.  EDIT THIS to match your label scheme.
# Keeps the repo's existing 4-class ordinal convention (0=most severe ..
# 3=least severe), covering both legacy Bugzilla names and the S1-S4 values
# Mozilla uses for 2018+ reports. `enhancement` is intentionally absent so
# feature requests (not real bugs) are dropped, matching the BugsRepo paper.
# ---------------------------------------------------------------------------
SEVERITY_MAP = {
    'blocker': 0, 'critical': 0, 's1': 0,
    'major': 1, 's2': 1,
    'normal': 2, 's3': 2,
    'minor': 3, 'trivial': 3, 's4': 3,
}

# Canonical field -> accepted source-column aliases (matched after normalization:
# lowercased, non-alphanumeric stripped). Extend as needed for your export.
TEXT_ALIASES = {
    'Summary':            ['summary', 'shortdesc', 'short_desc', 'title', 'bugtitle'],
    'StepsToReproduce':   ['stepstoreproduce', 's2r', 'steps', 'steps_to_reproduce', 'stepstorepro', 'text'],
    'ExpectedBehavior':   ['expectedbehavior', 'expectedbehaviour', 'eb', 'er', 'expected'],
    'ActualBehavior':     ['actualbehavior', 'actualbehaviour', 'ab', 'ar', 'actual'],
}
SEVERITY_ALIASES = ['severity', 'bugseverity', 'severitylevel', 'bugseveritylevel']
METRIC_ALIASES = {
    'num_comments':        ['numcomments', 'num_comments', 'commentsmade', 'comments_made',
                            'commentcount', 'ncomments', 'comments'],
    'bugs_filed':          ['bugsfiled', 'bugs_filed'],
    'assigned_and_fixed':  ['assignedandfixed', 'assigned_and_fixed',
                            'assignedtoandfixed', 'assigned_to_and_fixed'],
    'patches_submitted':   ['patchessubmitted', 'patches_submitted'],
    'patches_reviewed':    ['patchesreviewed', 'patches_reviewed'],
}

TEXT_FIELDS = list(TEXT_ALIASES.keys())
METRIC_FIELDS = list(METRIC_ALIASES.keys())


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def build_column_index(columns):
    """Map each canonical field to the actual source column name (or None)."""
    norm_to_actual = {_norm(c): c for c in columns}
    resolved = {}
    for canonical, aliases in TEXT_ALIASES.items():
        resolved[canonical] = next((norm_to_actual[a] for a in map(_norm, aliases)
                                    if a in norm_to_actual), None)
    resolved['__severity__'] = next((norm_to_actual[a] for a in map(_norm, SEVERITY_ALIASES)
                                     if a in norm_to_actual), None)
    for canonical, aliases in METRIC_ALIASES.items():
        resolved[canonical] = next((norm_to_actual[a] for a in map(_norm, aliases)
                                    if a in norm_to_actual), None)
    return resolved


def severity_to_label(raw):
    """Map a raw severity string to an int label, or None to drop the row."""
    if raw is None:
        return None
    norm = _norm(raw)
    if norm in SEVERITY_MAP:
        return SEVERITY_MAP[norm]
    # Fall back to the first whitespace token, e.g. "S2 (Serious)" -> "S2".
    first = _norm(str(raw).split()[0]) if str(raw).split() else ''
    return SEVERITY_MAP.get(first)


def to_float(val):
    try:
        if val is None or (isinstance(val, str) and not val.strip()):
            return 0.0
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def clean_text(val):
    if val is None:
        return ''
    s = str(val).strip()
    return '' if s.lower() in ('nan', 'none', 'null') else s


def load_rows(path):
    """Load raw rows as a list of dicts from CSV, JSON (array) or JSONL."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.csv', '.tsv'):
        import pandas as pd
        sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
        return df.to_dict(orient='records')
    if ext == '.jsonl':
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get('data', [])
    raise ValueError(f'Unsupported input extension: {ext} (use .csv/.tsv/.json/.jsonl)')


def convert_rows(rows, experiment='all'):
    """Convert raw rows -> output records. Returns (records, stats)."""
    if not rows:
        raise ValueError('Input contains no rows.')
    cols = build_column_index(rows[0].keys())

    if cols['__severity__'] is None:
        raise ValueError(f'No severity column found. Looked for: {SEVERITY_ALIASES}. '
                         f'Available columns: {list(rows[0].keys())}')

    # Which fields to emit for the requested experiment ('all' = everything).
    if experiment == 'A':
        keep_text, keep_metrics = ['Summary'], []
    elif experiment == 'B':
        keep_text, keep_metrics = TEXT_FIELDS, []
    elif experiment == 'C':
        keep_text, keep_metrics = TEXT_FIELDS, METRIC_FIELDS
    else:  # 'all'
        keep_text, keep_metrics = TEXT_FIELDS, METRIC_FIELDS

    records, dropped = [], 0
    for row in rows:
        label = severity_to_label(row.get(cols['__severity__']))
        if label is None:
            dropped += 1
            continue
        rec = {'label': label}
        for field in keep_text:
            src = cols.get(field)
            rec[field] = clean_text(row.get(src)) if src else ''
        for field in keep_metrics:
            src = cols.get(field)
            rec[field] = to_float(row.get(src)) if src else 0.0
        records.append(rec)

    stats = {
        'resolved_columns': {k: v for k, v in cols.items() if v is not None},
        'missing_columns': [k for k, v in cols.items() if v is None],
        'kept': len(records),
        'dropped_unknown_severity': dropped,
    }
    return records, stats


def stratified_split(records, train_frac, valid_frac, seed):
    """Label-stratified split into train/valid/test."""
    from sklearn.model_selection import train_test_split
    labels = [r['label'] for r in records]
    # Stratify only when every class has >= 2 members; else fall back to plain split.
    from collections import Counter
    stratify = labels if min(Counter(labels).values()) >= 2 else None

    test_frac = 1.0 - train_frac - valid_frac
    if test_frac <= 0:
        raise ValueError('train_frac + valid_frac must be < 1.0')

    train, temp = train_test_split(
        records, train_size=train_frac, random_state=seed, stratify=stratify
    )
    rel_valid = valid_frac / (valid_frac + test_frac)
    temp_labels = [r['label'] for r in temp]
    temp_stratify = temp_labels if (stratify is not None
                                    and min(Counter(temp_labels).values()) >= 2) else None
    valid, test = train_test_split(
        temp, train_size=rel_valid, random_state=seed, stratify=temp_stratify
    )
    return train, valid, test


def write_jsonl(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def label_dist(records):
    from collections import Counter
    return dict(sorted(Counter(r['label'] for r in records).items()))


def parse_args():
    p = argparse.ArgumentParser(description='Convert raw BugsRepo to train/valid/test JSONL')
    p.add_argument('--input', required=True, help='Raw BugsRepo CSV/TSV/JSON/JSONL file')
    p.add_argument('--outdir', default='data', help='Output directory for the JSONL splits')
    p.add_argument('--experiment', default='all', choices=['all', 'A', 'B', 'C'],
                   help="Restrict emitted fields to one experiment ('all' serves A/B/C)")
    p.add_argument('--train_frac', type=float, default=0.70)
    p.add_argument('--valid_frac', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = load_rows(args.input)
    print(f'Loaded {len(rows)} raw rows from {args.input}')

    records, stats = convert_rows(rows, experiment=args.experiment)
    print(f'Resolved columns : {stats["resolved_columns"]}')
    if stats['missing_columns']:
        print(f'WARNING missing columns (filled with empty/0.0): {stats["missing_columns"]}')
    print(f'Kept {stats["kept"]} rows; dropped {stats["dropped_unknown_severity"]} '
          f'(enhancement/unknown severity)')
    if not records:
        sys.exit('No usable records after severity mapping — check SEVERITY_MAP / columns.')

    train, valid, test = stratified_split(records, args.train_frac, args.valid_frac, args.seed)
    for name, split in [('train', train), ('valid', valid), ('test', test)]:
        path = os.path.join(args.outdir, f'{name}.jsonl')
        write_jsonl(split, path)
        print(f'  {name}: {len(split):>6} -> {path}  | label dist {label_dist(split)}')

    print('Done.')


if __name__ == '__main__':
    main()
