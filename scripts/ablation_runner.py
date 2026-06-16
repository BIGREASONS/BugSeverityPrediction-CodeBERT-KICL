import subprocess
import json
import os
import pandas as pd

def run_ablation(max_samples=100, epochs=1):
    fusion_types = ['none', 'concat10', 'metric_encoder64']
    results = []

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    for ftype in fusion_types:
        print(f"\n{'='*50}")
        print(f"Running Ablation for fusion_type: {ftype}")
        print(f"{'='*50}\n")
        
        # In a real run, you might want to run mlm and contrastive stages first.
        # Here we assume we're just comparing the finetuning stage.
        
        # We rename the output history file so it doesn't get overwritten
        hist_file = f"models/kicl_finetune_{ftype}_history.json"
        
        cmd = [
            "python", os.path.join("scripts", "kicl_pretrain.py"),
            "--stage", "finetune",
            "--fusion_type", ftype,
            "--epochs", str(epochs),
            "--max_train_samples", str(max_samples)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Move the default history file to the specific one
        default_hist = "models/kicl_finetune_history.json"
        if os.path.exists(default_hist):
            os.rename(default_hist, hist_file)
            
        if os.path.exists(hist_file):
            with open(hist_file, 'r') as f:
                history = json.load(f)
                # Get the epoch with the best F1 Macro or lowest loss
                best_epoch = min(history, key=lambda x: x['loss'])
                best_epoch['fusion_type'] = ftype
                results.append(best_epoch)

    # Output results
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns for readability
        cols = ['fusion_type', 'acc', 'f1_macro', 'f1_weight', 'mcc', 'loss', 'epoch']
        cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols and c != 'fusion_type']
        df = df[cols]
        
        df.to_csv("models/ablation_results.csv", index=False)
        print("\n" + "="*50)
        print("Ablation Results")
        print("="*50)
        print(df.to_string(index=False))
        
        try:
            print("\n" + "="*50)
            print("Ablation Results (Markdown)")
            print("="*50)
            print(df.to_markdown(index=False))
        except ImportError:
            pass
        
        try:
            print("\n" + "="*50)
            print("Ablation Results (LaTeX)")
            print("="*50)
            print(df.style.to_latex())
        except ImportError:
            pass
        
if __name__ == "__main__":
    # For a quick test, use very few samples and epochs.
    # In production, use max_samples=None and epochs=10
    run_ablation(max_samples=20, epochs=1)
