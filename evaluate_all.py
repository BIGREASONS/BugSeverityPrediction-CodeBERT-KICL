import os
import subprocess
import json

test_file = r"C:\Users\singh\OneDrive\Documents\BugRe\bugsrepo-kicl-jsonl\test.jsonl"

checkpoints = [
    # Backbone, Exp, ckpt_path, hf_name, out_json_name
    ("CodeBERT", "A", r"results_archive\codebert\A\best_checkpoint.pt", "microsoft/codebert-base", "codebert_A_metrics.json"),
    ("CodeBERT", "B", r"models\best_model.pt", "microsoft/codebert-base", "codebert_B_metrics.json"),
    ("CodeBERT", "C", r"results_archive\codebert\C\best_checkpoint.pt", "microsoft/codebert-base", "codebert_C_metrics.json"),
    
    ("UniXCoder", "A", r"results_archive\unixcoder\A\kicl_finetune_unixcoder_A_best.pt", "microsoft/unixcoder-base", "unixcoder_A_metrics.json"),
    ("UniXCoder", "B", r"results_archive\unixcoder\B\kicl_finetune_unixcoder_B_best.pt", "microsoft/unixcoder-base", "unixcoder_B_metrics.json"),
    ("UniXCoder", "C", r"results_archive\unixcoder\C\kicl_finetune_unixcoder_C_best.pt", "microsoft/unixcoder-base", "unixcoder_C_metrics.json"),
    
    ("CodeT5+", "A", r"results_archive\codet5p\kicl_finetune_codet5p_A_best.pt", "Salesforce/codet5p-220m", "codet5p_A_metrics.json"),
    ("CodeT5+", "B", r"results_archive\codet5p\kicl_finetune_codet5p_B_best.pt", "Salesforce/codet5p-220m", "codet5p_B_metrics.json"),
    ("CodeT5+", "C", r"results_archive\codet5p\kicl_finetune_codet5p_C_best.pt", "Salesforce/codet5p-220m", "codet5p_C_metrics.json"),
    
    ("CodeT5+ KICL", "KICL", r"C:\Users\singh\Downloads\codet5p_kicl_results\kicl_finetune_codet5p_C_kicl_best.pt", "Salesforce/codet5p-220m", "codet5p_kicl_metrics.json"),
]

results_out = []
os.makedirs("results", exist_ok=True)

for backbone, exp, ckpt, hf_name, json_name in checkpoints:
    print(f"==================================================", flush=True)
    print(f"Evaluating {backbone} - {exp}", flush=True)
    print(f"==================================================", flush=True)
    
    if not os.path.exists(ckpt):
        print(f"ERROR: Missing {ckpt}", flush=True)
        continue
        
    out_dir = f"temp_eval_{json_name.split('.')[0]}"
    os.makedirs(out_dir, exist_ok=True)
    
    # KICL experiment defaults to C if not provided, but KICL should use C's config.
    exp_arg = "C" if exp == "KICL" else exp

    cmd = [
        "python", "scripts/evaluate.py",
        "--model_path", ckpt,
        "--test_file", test_file,
        "--model_name", hf_name,
        "--output_dir", out_dir,
        "--experiment", exp_arg,
        "--batch_size", "16"
    ]
    subprocess.run(cmd, check=True)
    
    # Read the output JSON
    baseline_json = os.path.join(out_dir, "baseline_results.json")
    with open(baseline_json, "r") as f:
        metrics = json.load(f)
        
    # Copy to results directory
    final_json = os.path.join("results", json_name)
    with open(final_json, "w") as f:
        json.dump(metrics, f, indent=2)
        
    results_out.append({
        "Backbone": backbone,
        "Experiment": exp,
        "Accuracy": metrics.get("accuracy", 0.0),
        "W_Precision": metrics.get("precision_weighted", 0.0),
        "W_Recall": metrics.get("recall_weighted", 0.0),
        "W_F1": metrics.get("f1_weighted", 0.0),
        "W_ROC_AUC": metrics.get("auc_weighted", 0.0) if metrics.get("auc_weighted") is not None else 0.0,
        "MCC": metrics.get("mcc", 0.0),
        "G_Mean": metrics.get("g_mean", 0.0),
    })

# Sort by W_F1 then MCC
results_out.sort(key=lambda x: (x["W_F1"], x["MCC"]), reverse=True)

md = "# Final Leaderboard\n\n"
md += "| Backbone | Experiment | Accuracy | W.Precision | W.Recall | W.F1 | W.ROC-AUC | MCC | G-Mean |\n"
md += "|----------|------------|----------|-------------|----------|------|-----------|-----|--------|\n"

for r in results_out:
    b = r["Backbone"]
    e = r["Experiment"]
    
    # Bold the top row
    if r == results_out[0]:
        b = f"**{b}**"
        e = f"**{e}**"
        
    row = f"| {b} | {e} | {r['Accuracy']:.4f} | {r['W_Precision']:.4f} | {r['W_Recall']:.4f} | {r['W_F1']:.4f} | {r['W_ROC_AUC']:.4f} | {r['MCC']:.4f} | {r['G_Mean']:.4f} |"
    # Highlight the best W_F1 specifically if it's the top row
    if r == results_out[0]:
        row = row.replace(f" {r['W_F1']:.4f} ", f" **{r['W_F1']:.4f}** ")
    md += row + "\n"

# Write leaderboard.md
with open("results/leaderboard.md", "w") as f:
    f.write(md)

print("\nDone! Evaluated all models and created results/leaderboard.md.")
print(md)
