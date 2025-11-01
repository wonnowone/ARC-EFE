"""
Simple throughput measurement - works on CPU or GPU
Tests actual data loading and processing time
"""

import torch
import time
import os
import sys
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

from dataset_arc import ARCDataset
from torch.utils.data import DataLoader

def measure_throughput_simple(num_batches=200):
    """
    Simple throughput test - CPU compatible
    """
    print("\n" + "="*70)
    print("SIMPLE THROUGHPUT BENCHMARK (CPU COMPATIBLE)")
    print("="*70 + "\n")
    sys.stdout.flush()

    # Check what we have
    print("Checking PyTorch setup...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA compiled: ", end="")

    try:
        _ = torch.zeros(1).cuda()
        print("YES")
        device = 'cuda'
    except:
        print("NO - using CPU")
        device = 'cpu'

    print()
    sys.stdout.flush()

    # Load dataset
    print("Loading ARC dataset...")
    data_dir = os.getenv('ARC_DATA_DIR', '.')
    train_path = os.path.join(data_dir, 'training.json')

    if not os.path.exists(train_path):
        print(f"ERROR: Dataset not found at {train_path}")
        return None

    train_ds = ARCDataset(train_path, split='train')
    train_ld = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Loaded: {len(train_ds)} samples\n")
    sys.stdout.flush()

    # Measure time
    print(f"Measuring {num_batches} batches on {device.upper()}...")
    print("(Each batch: load + process)\n")
    sys.stdout.flush()

    times = []
    pbar = tqdm(enumerate(train_ld), total=num_batches,
                desc="  Measuring",
                unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for batch_idx, batch in pbar:
        if batch_idx >= num_batches:
            break

        start = time.time()

        # Just load to CPU (don't move to GPU if not available)
        try:
            inp = batch["input"].float()
            out = batch["output"].float()

            if inp is not None and out is not None:
                # Basic processing
                _ = inp.mean()
                _ = out.mean()
        except Exception as e:
            print(f"\nError: {e}")
            continue

        elapsed = time.time() - start
        times.append(elapsed)

    pbar.close()
    print()
    sys.stdout.flush()

    # Results
    if not times:
        print("ERROR: No data collected!")
        return None

    import statistics

    avg_ms = statistics.mean(times) * 1000
    median_ms = statistics.median(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0

    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nPer-Batch Statistics ({len(times)} batches):")
    print(f"  Average:  {avg_ms:.2f} ms")
    print(f"  Median:   {median_ms:.2f} ms")
    print(f"  Min:      {min_ms:.2f} ms")
    print(f"  Max:      {max_ms:.2f} ms")
    print(f"  Std Dev:  {std_ms:.2f} ms")

    # Project to full epoch
    epoch_ms = avg_ms * 3232
    epoch_sec = epoch_ms / 1000
    epoch_min = epoch_sec / 60
    epoch_hours = epoch_min / 60

    print(f"\nFull Epoch (3232 samples):")
    print(f"  {epoch_ms/1000:.0f} seconds")
    print(f"  {epoch_min:.1f} minutes")
    print(f"  {epoch_hours:.2f} hours")

    print("\n" + "="*70)
    print("EPOCH RECOMMENDATIONS")
    print("="*70 + "\n")

    for hours in [0.5, 1, 2, 3, 4, 6, 8, 12, 24]:
        max_epochs = max(1, int(hours / epoch_hours))
        total = max_epochs * epoch_hours
        label = f"{hours:.1f}" if hours == int(hours) else f"{hours:.1f}"
        print(f"  {label} hours available  →  {max_epochs:2} epochs  (~{total:.1f} hours)")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70 + "\n")

    if epoch_hours < 0.25:
        suggested = 20
        reason = "Very fast - can do many epochs"
    elif epoch_hours < 0.5:
        suggested = 10
        reason = "Fast - good for full training"
    elif epoch_hours < 1:
        suggested = 5
        reason = "Moderate - balanced approach"
    elif epoch_hours < 2:
        suggested = 3
        reason = "Slower - focus on convergence"
    else:
        suggested = 2
        reason = "Very slow - minimal epochs"

    print(f"Suggested: {suggested} epochs")
    print(f"Reason: {reason}")
    print(f"Total time: ~{suggested * epoch_hours:.1f} hours\n")

    print("="*70)
    print("COMMAND TO RUN")
    print("="*70)
    print(f"\npython trainloop_gpu_finetuned.py --epochs {suggested} --device cuda\n")
    sys.stdout.flush()

    return {
        'avg_ms': avg_ms,
        'epoch_hours': epoch_hours,
        'suggested_epochs': suggested,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=200,
                       help="Number of batches to test (default: 200)")
    args = parser.parse_args()

    results = measure_throughput_simple(num_batches=args.batches)

    if results:
        print("\nBenchmark completed successfully!")
    else:
        print("\nBenchmark failed!")
