"""
Quick GPU diagnostic script.
Checks if PyTorch can detect your GPU.
"""

import torch
import sys

print("="*60)
print("GPU DIAGNOSTIC")
print("="*60)

# Check PyTorch version
print(f"\nPyTorch Version: {torch.__version__}")

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Test GPU
    print("\nTesting GPU...")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print("✅ GPU is working!")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("\n❌ CUDA not available!")
    print("\nPossible reasons:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. CUDA drivers not installed")
    print("3. GPU not detected by system")

    print("\n" + "="*60)
    print("SOLUTION:")
    print("="*60)
    print("\nReinstall PyTorch with CUDA support:")
    print("\nFor CUDA 11.8:")
    print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor CUDA 12.1:")
    print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\nFor CUDA 12.4:")
    print("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

print("\n" + "="*60)
