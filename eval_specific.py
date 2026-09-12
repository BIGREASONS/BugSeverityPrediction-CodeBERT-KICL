import os
import subprocess
import json

checkpoints = [
    ("CodeBERT", "C", r"C:\Users\singh\OneDrive\Documents\BugRe\results_archive\codebert\C\best_checkpoint.pt", "microsoft/codebert-base"),
    ("UniXCoder", "C", r"C:\Users\singh\OneDrive\Documents\BugRe\results_archive\unixcoder\C\kicl_finetune_unixcoder_C_best.pt", "microsoft/unixcoder-base"),
    ("CodeT5+ Finetune-only", "C", r"C:\Users\singh\OneDrive\Documents\BugRe\results_archive\codet5p\kicl_finetune_codet5p_C_best.pt", "Salesforce/codet5p-220m"),
    ("CodeT5+ KICL", "C", r"C:\Users\singh\Downloads\codet5p_kicl_results\kicl_finetune_codet5p_C_kicl_best.pt", "Salesforce/codet5p-220m"),
]

results = []

for model_name, exp, ckpt, hf_name in checkpoints:
    print(f"Evaluating {model_name} {exp}...", flush=True)
    if not os.path.exists(ckpt):
        print(f"Missing {ckpt}", flush=True)
        continue
        
    out_dir = f"results_eval/{model_name.replace(' ', '_').replace('+', 'p')}"
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
        "Model": model_name,
        "Accuracy": metrics.get("accuracy"),
        "Macro-F1": metrics.get("classification_report", {}).get("macro avg", {}).get("f1-score"),
        "W-F1": metrics.get("f1_weighted"),
        "ROC-AUC": metrics.get("auc_weighted"),
        "MCC": metrics.get("mcc"),
    })

print("\n--- TEST METRICS ---")
print("| Model | Accuracy | W-F1 | ROC-AUC | MCC | Macro-F1 |")
print("|-------|----------|------|---------|-----|----------|")
for r in results:
    acc = f"{r['Accuracy']:.4f}" if r['Accuracy'] is not None else "N/A"
    w_f1 = f"{r['W-F1']:.4f}" if r['W-F1'] is not None else "N/A"
    roc = f"{r['ROC-AUC']:.4f}" if r['ROC-AUC'] is not None else "N/A"
    mcc = f"{r['MCC']:.4f}" if r['MCC'] is not None else "N/A"
    mac_f1 = f"{r['Macro-F1']:.4f}" if r['Macro-F1'] is not None else "N/A"
    print(f"| {r['Model']} | {acc} | {w_f1} | {roc} | {mcc} | {mac_f1} |")
