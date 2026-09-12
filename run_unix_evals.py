import os
import subprocess
import json

checkpoints = [
    ("UniXCoder", "A", r"results_archive\unixcoder\A\kicl_finetune_unixcoder_A_best.pt", "microsoft/unixcoder-base"),
    ("UniXCoder", "B", r"results_archive\unixcoder\B\kicl_finetune_unixcoder_B_best.pt", "microsoft/unixcoder-base"),
    ("UniXCoder", "C", r"results_archive\unixcoder\C\kicl_finetune_unixcoder_C_best.pt", "microsoft/unixcoder-base"),
]

results = []

for model, exp, ckpt, hf_name in checkpoints:
    print(f"Evaluating {model} {exp}...", flush=True)
    if not os.path.exists(ckpt):
        print(f"Missing {ckpt}", flush=True)
        continue
        
    out_dir = f"results_archive/{model.lower()}/{exp}"
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "python", "scripts/evaluate.py",
        "--model_path", ckpt,
        "--test_file", r"C:\Users\singh\OneDrive\Documents\BugRe\bugsrepo-kicl-jsonl\test.jsonl",
        "--model_name", hf_name,
        "--output_dir", out_dir,
        "--experiment", exp,
        "--batch_size", "16"
    ]
    subprocess.run(cmd, check=True)
    
    with open(os.path.join(out_dir, "baseline_results.json"), "r") as f:
        metrics = json.load(f)
        
    results.append({
        "Model": model,
        "Exp": exp,
        "W-Recall": metrics.get("recall_weighted"),
        "W-F1": metrics.get("f1_weighted"),
        "W-AUC": metrics.get("auc_weighted"),
        "MCC": metrics.get("mcc"),
        "G-Mean": metrics.get("g_mean"),
    })

print("| Model | Exp | W. Recall | W. F1 | W. ROC-AUC | MCC | G-Mean |")
print("|-------|-----|-----------|-------|------------|-----|--------|")
for r in results:
    w_rec = f"{r['W-Recall']:.4f}" if r['W-Recall'] is not None else "N/A"
    w_f1 = f"{r['W-F1']:.4f}" if r['W-F1'] is not None else "N/A"
    w_auc = f"{r['W-AUC']:.4f}" if r['W-AUC'] is not None else "N/A"
    mcc = f"{r['MCC']:.4f}" if r['MCC'] is not None else "N/A"
    g_mean = f"{r['G-Mean']:.4f}" if r['G-Mean'] is not None else "N/A"
    print(f"| {r['Model']} | {r['Exp']} | {w_rec} | {w_f1} | {w_auc} | {mcc} | {g_mean} |")
