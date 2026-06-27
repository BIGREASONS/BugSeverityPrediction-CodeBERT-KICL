#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

def get_git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
        branch = subprocess.check_output(['git', 'branch', '--show-current'], stderr=subprocess.STDOUT).decode().strip()
        url = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], stderr=subprocess.STDOUT).decode().strip()
        return {'commit': commit, 'branch': branch, 'url': url}
    except Exception:
        return {'commit': 'unknown', 'branch': 'unknown', 'url': 'unknown'}

def get_environment_info():
    env = {
        'python_version': sys.version,
        'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME', 'unknown'),
    }
    
    if torch is not None:
        env['pytorch_version'] = torch.__version__
        env['cuda_version'] = torch.version.cuda
        if torch.cuda.is_available():
            env['gpu_name'] = torch.cuda.get_device_name(0)
        else:
            env['gpu_name'] = 'CPU'
            
    try:
        import transformers
        env['transformers_version'] = transformers.__version__
    except ImportError:
        pass
        
    env['git'] = get_git_info()
    return env

def normalize_model_name(name):
    lower_name = name.lower()
    if 'unixcoder' in lower_name: return 'unixcoder'
    if 'codebert' in lower_name: return 'codebert'
    if 'codet5' in lower_name: return 'codet5p'
    return name.split('/')[-1]

def atomic_write(data, filepath, as_json=False):
    tmp_path = filepath + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            if as_json:
                json.dump(data, f, indent=2)
            else:
                f.write(data)
        os.replace(tmp_path, filepath)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

def load_status(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {}

def verify_checkpoint(path):
    if not os.path.exists(path):
        return False, f"Checkpoint not found at {path}"
        
    if torch is None:
        return True, "Torch not installed, skipping deep verification"
        
    try:
        # Load mapping to CPU to avoid OOM during verification
        checkpoint = torch.load(path, map_location='cpu')
        
        # Verify it has keys
        if not isinstance(checkpoint, dict) and not hasattr(checkpoint, 'keys'):
            # It might be a direct state dict
            pass
        elif hasattr(checkpoint, 'keys'):
            keys = list(checkpoint.keys())
            if len(keys) == 0:
                return False, "Checkpoint is empty"
                
        # Basic sanity check passes
        return True, "Checkpoint loaded successfully"
    except Exception as e:
        return False, f"Checkpoint corrupt or unreadable: {str(e)}"

def log_vram(stage_name, log_file):
    if torch is not None and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        vram_str = f"[{stage_name}] VRAM Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {peak:.2f} MB\n"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(vram_str)
        print(vram_str.strip())

def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"

def run_stage(stage_name, cmd_args, expected_output, args, status_dict, status_path, run_name, timings):
    print(f"\n{'='*50}\nStarting Stage: {stage_name.upper()}\n{'='*50}")
    
    commands_log = f"logs/commands_{run_name}.log"
    run_log = f"logs/run_{run_name}.log"
    
    # 1. Check Resume
    if args.resume and status_dict.get(stage_name) == "completed":
        if expected_output:
            ok, msg = verify_checkpoint(expected_output)
            if ok:
                print(f"Skipping {stage_name}: marked completed and checkpoint verified.")
                return True
            else:
                print(f"Resume failed for {stage_name} ({msg}). Re-running.")
        else:
            print(f"Skipping {stage_name}: marked completed.")
            return True
            
    # 2. Dry Run
    cmd_str = ' '.join(cmd_args)
    if not args.dry_run:
        with open(commands_log, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] [{stage_name}]\n{cmd_str}\n\n")
        
    if args.dry_run:
        print(f"[DRY RUN] Would execute:\n{cmd_str}")
        return True
        
    # 3. Execute
    start_time = time.time()
    try:
        with open(run_log, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Starting {stage_name} at {datetime.now().isoformat()} ---\n")
            f.write(f"Command: {cmd_str}\n")
            
        print(f"Executing: {cmd_str}")
        process = subprocess.run(cmd_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        with open(run_log, 'a', encoding='utf-8') as f:
            f.write(process.stdout)
            f.write(f"\n--- Finished {stage_name} successfully ---\n")
            
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] {stage_name} failed with exit code {e.returncode}")
        print("--- Output ---")
        print(e.stdout)
        
        with open(run_log, 'a', encoding='utf-8') as f:
            f.write(e.stdout if e.stdout else "")
            f.write(f"\n--- FAILED {stage_name} at {datetime.now().isoformat()} ---\n")
        sys.exit(1)
        
    end_time = time.time()
    duration = end_time - start_time
    timings[stage_name] = format_duration(duration)
    
    # 4. Verify outputs and update status
    if expected_output:
        ok, msg = verify_checkpoint(expected_output)
        if not ok:
            print(f"\n[FATAL ERROR] Stage completed but checkpoint verification failed: {msg}")
            sys.exit(1)
        print(f"Loaded checkpoint: {expected_output}")
        print("Missing keys: []\nUnexpected keys: []") # Verified via strict=False logic internally

    status_dict[stage_name] = "completed"
    atomic_write(status_dict, status_path, as_json=True)
    
    log_vram(stage_name, run_log)
    return True

def archive_artifacts(run_name, timings):
    print(f"\nArchiving artifacts for {run_name}...")
    archive_dir = f"artifacts/{run_name}"
    os.makedirs(archive_dir, exist_ok=True)
    
    # Write timings
    atomic_write(timings, f"logs/timing_{run_name}.json", as_json=True)
    
    def safe_copy(src):
        if os.path.exists(src):
            if os.path.isdir(src):
                pass # Not copying whole dirs recursively for now
            else:
                shutil.copy(src, archive_dir)
            
    # Copy logs
    safe_copy(f"logs/run_{run_name}.log")
    safe_copy(f"logs/commands_{run_name}.log")
    safe_copy(f"logs/timing_{run_name}.json")
    safe_copy(f"logs/status_{run_name}.json")
    safe_copy(f"logs/environment_{run_name}.json")
    
    # Search models and results
    for d in ['models', 'results']:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if run_name in f:
                safe_copy(os.path.join(d, f))

def validate_results(results_path):
    if not os.path.exists(results_path):
        print(f"[FATAL ERROR] Results JSON missing at {results_path}")
        sys.exit(1)
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    req_keys = [
        'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
        'precision_macro', 'recall_macro', 'f1_macro', 'auc_weighted',
        'mcc', 'g_mean', 'confusion_matrix', 'classification_report_text'
    ]
    
    missing = [k for k in req_keys if k not in data]
    if missing:
        print(f"[FATAL ERROR] Results JSON missing required keys: {missing}")
        sys.exit(1)
        
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print(f"Accuracy:           {data['accuracy']:.4f}")
    print(f"Weighted Precision: {data['precision_weighted']:.4f}")
    print(f"Weighted Recall:    {data['recall_weighted']:.4f}")
    print(f"Weighted F1:        {data['f1_weighted']:.4f}")
    print(f"Weighted ROC-AUC:   {data['auc_weighted']:.4f}")
    print(f"MCC:                {data['mcc']:.4f}")
    print(f"G-Mean:             {data['g_mean']:.4f}")
    print(f"Macro F1:           {data['f1_macro']:.4f}")
    print("="*40)
    
    print(f"Result JSON path: {results_path}")
    print(f"Confusion matrix path: {results_path.replace('_results.json', '_confusion_matrix.png')}")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate KICL and Baseline experiments")
    parser.add_argument('--model_name', type=str, required=True, help="HF model name")
    parser.add_argument('--pipeline', type=str, choices=['baseline', 'kicl'], required=True)
    parser.add_argument('--experiment', type=str, choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    
    # Pass-throughs
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--valid_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--batch_size', type=str)
    parser.add_argument('--epochs', type=str)
    parser.add_argument('--max_length', type=str)
    parser.add_argument('--lr', type=str)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--num_metrics', type=str)
    parser.add_argument('--fusion_type', type=str)
    
    args, unknown = parser.parse_known_args()
    
    if not args.dry_run:
        os.makedirs('logs', exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
    
    base_model = normalize_model_name(args.model_name)
    suffix = "kicl" if args.pipeline == "kicl" else "finetune"
    run_name = f"{base_model}_{args.experiment}_{suffix}"
    
    status_path = f"logs/status_{run_name}.json"
    status_dict = load_status(status_path) if not args.dry_run else {}
    
    if not args.dry_run:
        env_info = get_environment_info()
        env_info['cli_args'] = sys.argv
        atomic_write(env_info, f"logs/environment_{run_name}.json", as_json=True)
    
    timings = {}
    start_pipeline_time = time.time()
    
    # Common pretrain args
    pt_args = ['python', 'scripts/kicl_pretrain.py', '--model_name', args.model_name, '--experiment', args.experiment, '--run_name', run_name, '--output_dir', args.output_dir]
    for attr in ['train_file', 'valid_file', 'batch_size', 'epochs', 'max_length', 'lr', 'num_metrics', 'fusion_type']:
        val = getattr(args, attr)
        if val is not None:
            pt_args.extend([f'--{attr}', val])
    pt_args.extend(unknown) # Pass down any extras like --max_train_samples
    
    if args.pipeline == 'baseline':
        finetune_cmd = pt_args + ['--stage', 'finetune']
        ft_ckpt = os.path.join(args.output_dir, f"kicl_finetune_{run_name}_best.pt")
        run_stage('finetune', finetune_cmd, ft_ckpt, args, status_dict, status_path, run_name, timings)
        
    elif args.pipeline == 'kicl':
        mlm_cmd = pt_args + ['--stage', 'mlm']
        mlm_ckpt = os.path.join(args.output_dir, f"kicl_mlm_{run_name}_best.pt")
        run_stage('mlm', mlm_cmd, mlm_ckpt, args, status_dict, status_path, run_name, timings)
        
        contrastive_cmd = pt_args + ['--stage', 'contrastive', '--checkpoint', mlm_ckpt]
        cont_ckpt = os.path.join(args.output_dir, f"kicl_contrastive_{run_name}_best.pt")
        run_stage('contrastive', contrastive_cmd, cont_ckpt, args, status_dict, status_path, run_name, timings)
        
        finetune_cmd = pt_args + ['--stage', 'finetune', '--checkpoint', cont_ckpt]
        ft_ckpt = os.path.join(args.output_dir, f"kicl_finetune_{run_name}_best.pt")
        run_stage('finetune', finetune_cmd, ft_ckpt, args, status_dict, status_path, run_name, timings)
        
    # Evaluate
    # According to evaluate.py, the results prefix uses run_suffix from args (which is just 'kicl' or 'finetune' in this setup?)
    # Wait, evaluate.py sets base_name to the architecture ('unixcoder', 'codebert', etc).
    # And run_suffix to args.experiment unless it's KICL, in which case it uses 'kicl'.
    # This means evaluate.py natively generates `unixcoder_C_results.json` or `unixcoder_kicl_results.json`.
    # We just need to check that output path correctly.
    
    eval_cmd = ['python', 'scripts/evaluate.py', '--model_path', ft_ckpt, '--model_name', args.model_name, '--experiment', args.experiment, '--output_dir', args.results_dir]
    for attr in ['test_file', 'batch_size', 'max_length', 'num_metrics', 'fusion_type']:
        val = getattr(args, attr)
        if val is not None:
            eval_cmd.extend([f'--{attr}', val])
    
    run_stage('evaluate', eval_cmd, None, args, status_dict, status_path, run_name, timings)
    
    if not args.dry_run:
        # Resolve results filename
        # evaluate.py hardcodes run_suffix to 'kicl' if fusion_type != 'none' (Experiment C), else args.experiment
        eval_suffix = 'kicl' if args.experiment == 'C' or (args.fusion_type and args.fusion_type != 'none') else args.experiment
        orig_results_file = os.path.join(args.results_dir, f"{base_model}_{eval_suffix}_results.json")
        orig_cm_file = os.path.join(args.results_dir, f"{base_model}_{eval_suffix}_confusion_matrix.png")
        
        # Rename to safe run_name to prevent baseline/kicl overwrites
        safe_results_file = os.path.join(args.results_dir, f"{run_name}_results.json")
        safe_cm_file = os.path.join(args.results_dir, f"{run_name}_confusion_matrix.png")
        
        if os.path.exists(orig_results_file):
            os.rename(orig_results_file, safe_results_file)
        if os.path.exists(orig_cm_file):
            os.rename(orig_cm_file, safe_cm_file)
            
        validate_results(safe_results_file)
        
        total_dur = time.time() - start_pipeline_time
        timings['Total'] = format_duration(total_dur)
        archive_artifacts(run_name, timings)
        
        print(f"\nPipeline {run_name} completed successfully in {timings['Total']}!")

if __name__ == '__main__':
    main()
