import os
import shutil

base = "results_archive"
for m in ["codebert", "unixcoder"]:
    for e in ["A", "B", "C"]:
        os.makedirs(os.path.join(base, m, e), exist_ok=True)

def safe_copy(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)

for e in ["A", "B", "C"]:
    src_dir = f"baselines/codebert/{e}"
    dst_dir = f"results_archive/codebert/{e}"
    if os.path.exists(src_dir):
        for f in os.listdir(src_dir):
            safe_copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

u_src = "C:/Users/singh/Downloads/unixcoder_results"
for e in ["A", "B", "C"]:
    safe_copy(f"{u_src}/kicl_finetune_unixcoder_{e}_best.pt", f"results_archive/unixcoder/{e}/kicl_finetune_unixcoder_{e}_best.pt")
    safe_copy(f"{u_src}/kicl_finetune_unixcoder_{e}_history.json", f"results_archive/unixcoder/{e}/kicl_finetune_unixcoder_{e}_history.json")

leaderboard = """## CodeBERT (Finetune Only)

A

* F1 Macro = 0.3612
* MCC = 0.3086

B

* F1 Macro = 0.4034
* MCC = 0.3637

C

* F1 Macro = 0.3886
* MCC = 0.3555

## UniXCoder (Finetune Only)

A

* F1 Macro = 0.3901
* MCC = 0.3249

B

* F1 Macro = 0.3746
* MCC = 0.3235

C

* F1 Macro = 0.3782
* MCC = 0.3347

Current Best:

* Model: CodeBERT
* Experiment: B
* F1 Macro: 0.4034
* MCC: 0.3637
"""

with open("results_archive/leaderboard.md", "w") as f:
    f.write(leaderboard)

print("Archive complete.")
