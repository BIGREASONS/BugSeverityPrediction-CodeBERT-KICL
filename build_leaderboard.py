import os
import json

checkpoints = [
    ("CodeBERT", "A", "codebert_A_metrics.json"),
    ("CodeBERT", "B", "codebert_B_metrics.json"),
    ("CodeBERT", "C", "codebert_C_metrics.json"),
    ("UniXCoder", "A", "unixcoder_A_metrics.json"),
    ("UniXCoder", "B", "unixcoder_B_metrics.json"),
    ("UniXCoder", "C", "unixcoder_C_metrics.json"),
    ("CodeT5+", "A", "codet5p_A_metrics.json"),
    ("CodeT5+", "B", "codet5p_B_metrics.json"),
    ("CodeT5+", "C", "codet5p_C_metrics.json"),
    ("CodeT5+ KICL", "KICL", "codet5p_kicl_metrics.json"),
]

results_out = []
for backbone, exp, json_name in checkpoints:
    path = os.path.join("results", json_name)
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        metrics = json.load(f)
        
    results_out.append({
        "Backbone": backbone,
        "Experiment": exp,
        "Accuracy": metrics.get("accuracy", 0.0),
        "W_Precision": metrics.get("precision_weighted", 0.0),
        "W_Recall": metrics.get("recall_weighted", 0.0),
        "W_F1": metrics.get("f1_weighted", 0.0),
        "M_Precision": metrics.get("precision_macro", 0.0),
        "M_Recall": metrics.get("recall_macro", 0.0),
        "M_F1": metrics.get("f1_macro", 0.0),
        "W_ROC_AUC": metrics.get("auc_weighted", 0.0) if metrics.get("auc_weighted") is not None else 0.0,
        "MCC": metrics.get("mcc", 0.0),
        "G_Mean": metrics.get("g_mean", 0.0),
    })

# Sort by W_F1 then MCC
results_out.sort(key=lambda x: (x["W_F1"], x["MCC"]), reverse=True)

md = "### Final Leaderboard (Test Set Only)\n\n"
md += "| Backbone | Experiment | Accuracy | W.Precision | W.Recall | W.F1 | M.Precision | M.Recall | M.F1 | W.ROC-AUC | MCC | G-Mean |\n"
md += "|----------|------------|----------|-------------|----------|------|-------------|----------|------|-----------|-----|--------|\n"

for r in results_out:
    b = r["Backbone"]
    e = r["Experiment"]
    
    # Bold the top row
    if r == results_out[0]:
        b = f"**{b}**"
        e = f"**{e}**"
        
    row = f"| {b} | {e} | {r['Accuracy']:.4f} | {r['W_Precision']:.4f} | {r['W_Recall']:.4f} | {r['W_F1']:.4f} | {r['M_Precision']:.4f} | {r['M_Recall']:.4f} | {r['M_F1']:.4f} | {r['W_ROC_AUC']:.4f} | {r['MCC']:.4f} | {r['G_Mean']:.4f} |"
    
    # Highlight best metrics for top row
    if r == results_out[0]:
        row = row.replace(f" {r['W_F1']:.4f} ", f" **{r['W_F1']:.4f}** ")
        
    md += row + "\n"

print(md)

with open("results/leaderboard.md", "w") as f:
    f.write(md)

# Update results_archive/leaderboard.md too
with open("results_archive/leaderboard.md", "w") as f:
    f.write("# KICL Leaderboard Archive\n\n" + md)
