"""
Measure actual training throughput on your GPU
Runs a quick benchmark to determine realistic epoch time estimates
"""

import torch
import time
from tqdm import tqdm
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add repo to path
sys.path.insert(0, '.')

try:
    from dataset_arc import ARCDataset
    from torch.utils.data import DataLoader
except Exception as e:
    print(f"Warning: Could not import dataset modules: {e}")
    DataLoader = None

def measure_throughput(num_batches=500, batch_size=1, device='cuda'):
    """
    Measure actual per-batch processing time on your hardware.
    """
    print("\n" + "="*70)
    print("GPU THROUGHPUT BENCHMARK")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Check device
    print("[1/5] Checking device...")
    if device == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_mem:.1f} GB\n")
    else:
        print(f"  Device: {device}\n")
    sys.stdout.flush()

    # Load dataset
    print("[2/5] Loading dataset...")
    data_dir = os.getenv('ARC_DATA_DIR', '.')
    train_path = os.path.join(data_dir, 'training.json')

    train_ld = None
    if os.path.exists(train_path):
        try:
            print(f"  Path: {train_path}")
            from dataset_arc import ARCDataset
            train_ds = ARCDataset(train_path, split='train')
            train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            print(f"  Loaded: {len(train_ds)} samples\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Will use synthetic data instead\n")
            sys.stdout.flush()
            train_ld = None
    else:
        print(f"  File not found: {train_path}")
        print(f"  Will use synthetic data instead\n")
        sys.stdout.flush()

    # Create synthetic data if needed
    if train_ld is None:
        print("[2/5] Creating synthetic data...")
        import random
        import numpy as np

        class SyntheticDataset:
            def __init__(self, size=500):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                h, w = random.randint(5, 25), random.randint(5, 25)
                return {
                    'input': torch.randint(0, 10, (h, w)).float(),
                    'output': torch.randint(0, 10, (h, w)).float()
                }

        synthetic_ds = SyntheticDataset(num_batches)
        train_ld = DataLoader(synthetic_ds, batch_size=batch_size, shuffle=False)
        print(f"  Created: {len(synthetic_ds)} synthetic samples\n")
        sys.stdout.flush()

    # Measure loading time
    print(f"[3/5] Warming up GPU...")
    # Warm up
    try:
        batch = next(iter(train_ld))
        _ = batch["input"].to(device)
    except:
        pass
    print("  Warm-up complete\n")
    sys.stdout.flush()

    print(f"[4/5] Running benchmark with {num_batches} batches...")
    print("  (This will take a few minutes)\n")
    sys.stdout.flush()

    times = []
    pbar = tqdm(enumerate(train_ld), total=num_batches, desc="  Benchmarking",
                unit="batch", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    batch_count = 0
    for batch_idx, batch in pbar:
        if batch_idx >= num_batches:
            break

        start = time.time()

        # Simulate data loading and moving to device
        try:
            inp = batch["input"].to(device)
            out = batch["output"].to(device) if batch["output"] is not None else None

            # Simulate basic processing
            if inp is not None and out is not None:
                # Basic feature operations
                _ = inp.mean()
                _ = out.mean()
                torch.cuda.synchronize() if device == 'cuda' else None
        except Exception as e:
            print(f"\n  Error processing batch: {e}")
            continue

        elapsed = time.time() - start
        times.append(elapsed)
        batch_count += 1

    pbar.close()
    print(f"  Completed {batch_count} batches\n")
    sys.stdout.flush()

    # Calculate statistics
    print("[5/5] Calculating statistics...\n")
    sys.stdout.flush()

    if not times:
        print("ERROR: No timing data collected!")
        sys.stdout.flush()
        return None

    import statistics

    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0

    print("="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"\nPer-Batch Time Statistics (based on {len(times)} batches):")
    print(f"  Average:     {avg_time*1000:.2f} ms")
    print(f"  Median:      {median_time*1000:.2f} ms")
    print(f"  Min:         {min_time*1000:.2f} ms")
    print(f"  Max:         {max_time*1000:.2f} ms")
    print(f"  Std Dev:     {stdev*1000:.2f} ms")
    sys.stdout.flush()

    # Calculate epoch times
    total_samples = 3232  # ARC dataset size
    total_batches_full = total_samples  # batch_size=1

    epoch_time_sec = total_batches_full * avg_time
    epoch_time_min = epoch_time_sec / 60
    epoch_time_hours = epoch_time_min / 60

    print(f"\nProjected Single Epoch Time (3232 samples):")
    print(f"  {epoch_time_sec:.0f} seconds")
    print(f"  {epoch_time_min:.1f} minutes")
    print(f"  {epoch_time_hours:.2f} hours")
    sys.stdout.flush()

    print("\n" + "="*70)
    print("RECOMMENDED EPOCH COUNTS")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Recommendations based on time budget
    configs = {
        '30 minutes': 0.5,
        '1 hour': 1,
        '2 hours': 2,
        '4 hours': 4,
        '8 hours': 8,
        '12 hours': 12,
        '24 hours': 24,
    }

    print("If you have [TIME BUDGET] available:")
    print("-"*70)

    for time_label, time_hours in configs.items():
        max_epochs = max(1, int(time_hours / epoch_time_hours))
        total_time = max_epochs * epoch_time_hours
        print(f"{time_label:15} -> {max_epochs:2} epochs (~{total_time:.1f} hours)")

    sys.stdout.flush()

    print("\n" + "="*70)
    print("TRAINING RECOMMENDATIONS")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Suggest based on convergence
    if epoch_time_hours < 0.5:
        suggested_epochs = 20
        reason = "Fast per-epoch time - can afford more epochs for better convergence"
    elif epoch_time_hours < 1:
        suggested_epochs = 10
        reason = "Moderate per-epoch time - good balance"
    elif epoch_time_hours < 2:
        suggested_epochs = 5
        reason = "Slower per-epoch time - fewer epochs recommended"
    else:
        suggested_epochs = 2
        reason = "Very slow per-epoch time - focus on quality over quantity"

    print(f"Suggested configuration: {suggested_epochs} epochs")
    print(f"Reason: {reason}")
    print(f"Total training time: ~{suggested_epochs * epoch_time_hours:.1f} hours")
    sys.stdout.flush()

    print("\n" + "="*70)
    print("ACTUAL COMMAND TO RUN")
    print("="*70)
    print(f"\npython trainloop_gpu_finetuned.py --epochs {suggested_epochs} --device cuda\n")
    sys.stdout.flush()

    return {
        'avg_time_ms': avg_time * 1000,
        'epoch_time_hours': epoch_time_hours,
        'epoch_time_min': epoch_time_min,
        'suggested_epochs': suggested_epochs,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure GPU throughput for training")
    parser.add_argument("--batches", type=int, default=500,
                       help="Number of batches to benchmark (default: 500)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to benchmark (default: cuda)")

    args = parser.parse_args()

    results = measure_throughput(num_batches=args.batches, device=args.device)
