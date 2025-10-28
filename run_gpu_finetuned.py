# -*- coding: utf-8 -*-
"""
Unified GPU Training Entry Point

Supports:
- Local GPU (NVIDIA GPU with CUDA)
- Google Colab (with setup instructions)
- Cloud GPU (Lambda Labs, Vast.ai, etc)

Features:
- Qwen fine-tuning (full LM trainable)
- Grid accuracy loss (direct optimization)
- Binary accuracy validation (strict ARC)
- Mixed precision (for memory efficiency)
"""

import os
import sys
import torch
import argparse
from pathlib import Path


def check_gpu_availability():
    """Check GPU availability and print info"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        print("   This script requires a GPU")
        print("\n   Options:")
        print("   1. Run locally if you have NVIDIA GPU")
        print("   2. Use Google Colab (free GPU)")
        print("   3. Use cloud GPU provider (Lambda Labs, Vast.ai, etc)")
        return False

    print("✓ CUDA available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  PyTorch version: {torch.__version__}")
    return True


def check_dependencies():
    """Check required dependencies"""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers (for Qwen)',
        'numpy': 'NumPy',
    }

    print("\nChecking dependencies...")
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing.append(module)

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        for module in missing:
            print(f"  pip install {module}")
        return False

    return True


def check_hf_credentials():
    """Check HuggingFace credentials for Qwen access"""
    print("\nChecking HuggingFace access...")

    # Check if token exists
    hf_token_path = Path.home() / ".huggingface" / "token"
    if hf_token_path.exists():
        print("  ✓ HuggingFace token found")
        return True

    print("  ⚠ No HuggingFace token found")
    print("\n  Qwen requires HuggingFace authentication.")
    print("  Set it up with:")
    print("    huggingface-cli login")
    print("\n  Or export token:")
    print("    export HUGGING_FACE_HUB_TOKEN='your-token-here'")
    return False


def setup_colab():
    """Print Colab setup instructions"""
    print("\n" + "="*70)
    print("GOOGLE COLAB SETUP".center(70))
    print("="*70)

    print("""
1. Create a new Colab notebook at https://colab.research.google.com

2. Install dependencies (first cell):
   !pip install torch transformers datasets

3. Clone or upload the code:
   !git clone https://github.com/your-repo/arc-solver.git
   # or
   # Upload files to Colab

4. Setup HuggingFace auth (in Colab):
   from huggingface_hub import notebook_login
   notebook_login()

5. Run training (main cell):
   %cd /content/arc-solver/experiment/final
   !python trainloop_gpu_finetuned.py --epochs 5 --device cuda

6. Download results:
   # Files saved in runs/arc_gpu_finetuned_*
   # Use Colab's download feature
""")


def setup_cloud_gpu(provider="lambda"):
    """Print cloud GPU setup instructions"""
    print("\n" + "="*70)
    if provider == "lambda":
        print("LAMBDA LABS SETUP".center(70))
    elif provider == "vast":
        print("VAST.AI SETUP".center(70))
    else:
        print("CLOUD GPU SETUP".center(70))
    print("="*70)

    if provider == "lambda":
        print("""
1. Create account at https://lambdalabs.com/

2. Launch an instance:
   - Select: GPU Cloud > On-Demand
   - Instance type: GPU 1x A100 or RTX 4090 (12GB+ VRAM)
   - OS: Ubuntu 20.04 with CUDA

3. SSH into instance:
   ssh ubuntu@<instance-ip>

4. Clone code:
   git clone https://github.com/your-repo/arc-solver.git
   cd arc-solver/experiment/final

5. Install dependencies:
   pip install -r requirements.txt
   huggingface-cli login  # Enter your HF token

6. Run training:
   python trainloop_gpu_finetuned.py --epochs 10 --device cuda

7. Monitor training:
   # In another SSH window:
   tail -f runs/arc_gpu_finetuned_*/training.log
""")
    elif provider == "vast":
        print("""
1. Create account at https://vast.ai/

2. Find instance:
   - Filter: NVIDIA GPU, 12GB+ VRAM, CUDA 12.0+
   - Recommended: RTX 4090, RTX 4080, A100

3. Launch and SSH into instance

4. Setup same as Lambda Labs above
""")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Training with Qwen Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:

  # Local GPU
  python run_gpu_finetuned.py --mode train --epochs 5

  # Setup Colab
  python run_gpu_finetuned.py --mode colab

  # Setup cloud GPU
  python run_gpu_finetuned.py --mode cloud --provider lambda

REQUIREMENTS:

  GPU: NVIDIA GPU with 12GB+ VRAM
  CUDA: 11.8 or newer
  Python: 3.9+
  HuggingFace: Token required for Qwen access

ACCURACY IMPROVEMENTS:

  Frozen Qwen (CPU):    0.41 accuracy (plateaued)
  Fine-tuned Qwen (GPU): 0.50+ accuracy (expected)

TRAINING TIME:

  5 epochs:  ~30-45 min (GPU)
  10 epochs: ~60-90 min (GPU)
  vs 200+ min (CPU per epoch)
        """
    )

    parser.add_argument("--mode", choices=["train", "check", "colab", "cloud"],
                       default="train",
                       help="Mode: train (run training), check (verify setup), "
                            "colab (show Colab setup), cloud (show cloud setup)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs (default: 5)")
    parser.add_argument("--max-batches", type=int, default=None,
                       help="Max batches per epoch (default: None = full)")
    parser.add_argument("--provider", choices=["lambda", "vast"],
                       default="lambda",
                       help="Cloud provider for setup instructions")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")

    args = parser.parse_args()

    # Mode: Check setup
    if args.mode == "check":
        print("\n" + "="*70)
        print("CHECKING SETUP".center(70))
        print("="*70 + "\n")

        gpu_ok = check_gpu_availability()
        deps_ok = check_dependencies()
        hf_ok = check_hf_credentials()

        print("\n" + "="*70)
        if gpu_ok and deps_ok and hf_ok:
            print("✓ ALL CHECKS PASSED - Ready to train!".center(70))
            print("="*70 + "\n")
            print("Run training with:")
            print("  python run_gpu_finetuned.py --mode train --epochs 5")
        else:
            print("⚠ SETUP ISSUES DETECTED".center(70))
            print("="*70)
            return 1

        return 0

    # Mode: Colab setup
    if args.mode == "colab":
        setup_colab()
        return 0

    # Mode: Cloud setup
    if args.mode == "cloud":
        setup_cloud_gpu(args.provider)
        return 0

    # Mode: Train
    if args.mode == "train":
        print("\n" + "="*70)
        print("PREPARING TRAINING".center(70))
        print("="*70 + "\n")

        # Check setup
        if not check_gpu_availability():
            print("\n❌ GPU not available")
            print("   Use --mode colab or --mode cloud for setup instructions")
            return 1

        if not check_dependencies():
            print("\n❌ Missing dependencies")
            return 1

        if not check_hf_credentials():
            print("\n⚠ HuggingFace credentials not found")
            print("   Run: huggingface-cli login")
            # Don't fail, might work with public access

        print("\n✓ All checks passed, starting training...\n")

        # Import and run training
        try:
            from trainloop_gpu_finetuned import main
            output_dir, final_acc = main(
                epochs=args.epochs,
                max_batches_per_epoch=args.max_batches,
                device=args.device
            )
            print(f"\n✓ Training complete!")
            print(f"  Output: {output_dir}")
            print(f"  Best accuracy: {final_acc:.4f}")
            return 0

        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
