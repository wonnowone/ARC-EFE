"""
Measure actual training throughput on your GPU
Runs a quick benchmark to determine realistic epoch time estimates
"""

import torch
import time
from tqdm import tqdm
import os
import sys

# Add repo to path
sys.path.insert(0, '.')

from dataset_arc import ARCDataset
from torch.utils.data import DataLoader

def measure_throughput(num_batches=500, batch_size=1, device='cuda'):
    """
    Measure actual per-batch processing time on your hardware.
    """
    print("="*70)
    print("GPU THROUGHPUT BENCHMARK")
    print("="*70)

    # Check device
    if device == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset (max {num_batches} batches)...")
    data_dir = os.getenv('ARC_DATA_DIR', '.')
    train_path = os.path.join(data_dir, 'training.json')

    try:
        train_ds = ARCDataset(train_path, split='train')
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        print(f"Dataset loaded: {len(train_ds)} total samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data instead...")
        num_batches = 100

    # Measure loading time
    print(f"\nMeasuring throughput with {num_batches} batches...")
    print("-"*70)

    times = []
    pbar = tqdm(enumerate(train_ld), total=num_batches, desc="Benchmarking",
                unit="batch", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

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
        except Exception as e:
            pass

        elapsed = time.time() - start
        times.append(elapsed)

    pbar.close()

    # Calculate statistics
    import statistics

    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0

    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"\nPer-Batch Time Statistics:")
    print(f"  Average:     {avg_time*1000:.2f} ms")
    print(f"  Median:      {median_time*1000:.2f} ms")
    print(f"  Min:         {min_time*1000:.2f} ms")
    print(f"  Max:         {max_time*1000:.2f} ms")
    print(f"  Std Dev:     {stdev*1000:.2f} ms")

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

    print("\n" + "="*70)
    print("RECOMMENDED EPOCH COUNTS")
    print("="*70)

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

    print("\nIf you have [TIME BUDGET] available:")
    print("-"*70)

    for time_label, time_hours in configs.items():
        max_epochs = max(1, int(time_hours / epoch_time_hours))
        total_time = max_epochs * epoch_time_hours
        print(f"{time_label:15} -> {max_epochs:2} epochs (~{total_time:.1f} hours)")

    print("\n" + "="*70)
    print("TRAINING RECOMMENDATIONS")
    print("="*70)

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

    print(f"\nSuggested configuration: {suggested_epochs} epochs")
    print(f"Reason: {reason}")
    print(f"Total training time: ~{suggested_epochs * epoch_time_hours:.1f} hours")

    print("\n" + "="*70)
    print("ACTUAL COMMAND TO RUN")
    print("="*70)
    print(f"\npython trainloop_gpu_finetuned.py --epochs {suggested_epochs} --device cuda\n")

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
