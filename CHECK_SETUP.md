# Critical Setup Issue: CPU-Only PyTorch Detected

## The Problem

Your PyTorch installation is **CPU-ONLY**. The version installed is:
```
PyTorch 2.8.0+cpu
```

This means:
- ❌ **Cannot** run on GPU (CUDA not compiled)
- ❌ **Training will be EXTREMELY slow** (100x slower than GPU)
- ❌ Will fail when trying to move tensors to CUDA
- ✓ Can only run on CPU (very limited)

## What You Need to Do

### Option 1: Reinstall PyTorch with CUDA Support (Recommended)

If you're on **Google Colab**:
```python
# Run in a Colab cell
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you're on **Local Machine**:

**For Windows with NVIDIA GPU:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Linux with NVIDIA GPU:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For Apple Silicon (M1/M2/M3):**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Option 2: Use Google Colab (Easy, Free GPU)

1. Upload your code to Google Drive or GitHub
2. Create a Colab notebook
3. It automatically has CUDA-enabled PyTorch installed
4. GPU: T4, V100, A100 available

## How to Verify Fix

After reinstalling, run this:

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
    x = torch.zeros(1).cuda()
    print('SUCCESS: CUDA works!')
else:
    print('ERROR: Still CPU-only!')
"
```

**Good output:**
```
PyTorch version: 2.8.0+cu118
CUDA available: True
CUDA device: NVIDIA RTX 3060
SUCCESS: CUDA works!
```

**Bad output (your current state):**
```
PyTorch version: 2.8.0+cpu
CUDA available: False
ERROR: Still CPU-only!
```

## Expected Training Times AFTER Fix

Once you have CUDA-enabled PyTorch:

### On T4 GPU:
- Per epoch: 10-30 minutes
- 5 epochs: 50-150 minutes (~1-2.5 hours)
- 10 epochs: 100-300 minutes (~1.5-5 hours)

### On CPU (your current setup):
- Per epoch: 5-8 HOURS
- 5 epochs: 25-40 hours (overnight!)
- **NOT RECOMMENDED**

## Step-by-Step Fix

### Step 1: Uninstall Old PyTorch
```bash
pip uninstall -y torch torchvision torchaudio
```

### Step 2: Install CUDA Version
```bash
# For CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

or

```bash
# For CUDA 12.1 (newer GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 4: Run the Benchmark
```bash
python measure_throughput_simple.py --batches 100 --device cuda
```

### Step 5: Run Training
```bash
python trainloop_gpu_finetuned.py --epochs 5 --device cuda
```

## Which CUDA Version Do I Need?

Check your GPU's CUDA compatibility:

**NVIDIA GPU Models:**
- **RTX 30/40 series** → CUDA 11.8 or 12.1
- **RTX 20 series** → CUDA 11.8
- **GTX 16 series** → CUDA 11.8
- **Tesla T4** → CUDA 11.8 or 12.1
- **A100/V100** → CUDA 11.8 or 12.1

If unsure, use **CUDA 11.8** (most compatible):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## On Google Colab

Colab already has CUDA PyTorch pre-installed. Just run:

```bash
python trainloop_gpu_finetuned.py --epochs 5 --device cuda
```

No reinstall needed!

## For Intel Arc GPUs

Use ROCm version instead:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## Troubleshooting

### Error: "Torch not compiled with CUDA enabled"
- Still using CPU-only PyTorch
- Run the uninstall + reinstall steps above

### Error: "CUDA out of memory"
- GPU memory full
- Use: `--max-batches 50` to validate first
- Or reduce model: `--model-name Qwen/Qwen1.5-0.5B`

### GPU device shows "No NVIDIA GPU"
- Check GPU drivers are installed
- Run: `nvidia-smi` in terminal
- If no output, update drivers

## Timeline After Fix

```
Install correct PyTorch      → 5-10 minutes
Verify CUDA works            → 1 minute
Run quick benchmark (100x)   → 3-5 minutes
Run short training (1 epoch) → 15-30 minutes
Run full training (5 epochs) → 1.5-3 hours
```

## Common PyTorch Installation Commands

```bash
# Stable CUDA 11.8 (most compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Latest CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (no index needed)
pip install torch torchvision torchaudio

# ROCm for AMD
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# CPU only (NOT what you want)
pip install torch torchvision torchaudio
```

## Need Help?

After installing, run this diagnostic:

```bash
python -c "
import torch
print('='*60)
print('PyTorch Setup Diagnostic')
print('='*60)
print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

    # Test if it actually works
    try:
        x = torch.zeros(10, 10).cuda()
        y = x + 1
        print('GPU Test: PASSED')
    except Exception as e:
        print(f'GPU Test: FAILED - {e}')
else:
    print('ERROR: CUDA not available!')
    print('Need to reinstall PyTorch with CUDA support')
print('='*60)
"
```

This will show you exactly what's installed and if GPU works.

## Final Checklist

- [ ] Understand you have CPU-only PyTorch
- [ ] Installed CUDA-enabled PyTorch
- [ ] Verified with diagnostic script
- [ ] nvidia-smi shows GPU
- [ ] torch.cuda.is_available() returns True
- [ ] Ready to run training!

---

**Bottom line**: Install CUDA PyTorch first, then everything will work. Without it, you're limited to extremely slow CPU training.
