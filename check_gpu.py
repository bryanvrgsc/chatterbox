import torch
import sys
import platform

print(f"Python Platform: {sys.platform}")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 50)

# Check CUDA (NVIDIA)
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}") 
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n📊 GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   Multi-Processor Count: {props.multi_processor_count}")
    
    # CUDA 13.1 Compatibility Check
    cuda_version = torch.version.cuda
    if cuda_version:
        major, minor = map(int, cuda_version.split('.')[:2])
        print(f"\n🔍 CUDA Compatibility Check:")
        if major >= 13:
            print(f"   ✅ CUDA {cuda_version} - Fully compatible with CUDA Toolkit 13.1")
            if minor >= 1:
                print(f"   🎯 CUDA {major}.{minor} - Optimal version detected!")
        elif major == 12 and minor >= 6:
            print(f"   ✅ CUDA {cuda_version} - Compatible (12.6+ recommended for 13.x)")
        elif major == 12:
            print(f"   ⚠️  CUDA {cuda_version} - Update to 12.6+ recommended for better 13.x compatibility")
        else:
            print(f"   ⚠️  CUDA {cuda_version} - Upgrade to CUDA 12.6+ or 13.0+ strongly recommended")
    
    # Test CUDA 13.1 features
    print(f"\n🔧 CUDA Features:")
    print(f"   TF32 Allowed (matmul): {torch.backends.cuda.matmul.allow_tf32}")
    print(f"   TF32 Allowed (cudnn): {torch.backends.cudnn.allow_tf32}")
    print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    
    # Check Compute Capability for architecture features
    compute_cap = torch.cuda.get_device_capability(0)
    cc_major, cc_minor = compute_cap
    print(f"\n🏗️  GPU Architecture:")
    if cc_major >= 9:
        print(f"   ✅ Hopper+ architecture (SM {cc_major}.{cc_minor}) - Latest features available")
        print(f"   💡 Supports: FP8, TF32, BF16, INT8, Tensor Cores")
    elif cc_major >= 8:
        print(f"   ✅ Ampere+ architecture (SM {cc_major}.{cc_minor}) - Modern features available")
        print(f"   💡 Supports: TF32, BF16, INT8, Tensor Cores")
    elif cc_major >= 7:
        print(f"   ℹ️  Volta/Turing architecture (SM {cc_major}.{cc_minor})")
        print(f"   💡 Supports: FP16, INT8, Tensor Cores")
    else:
        print(f"   ⚠️  Older GPU architecture (SM {cc_major}.{cc_minor})")
        print(f"   💡 Limited modern features - consider upgrading for better performance")

# Check MPS (Mac GPU)
print(f"\n🍎 MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")

# Performance test
print("\n⚡ Performance Test:")
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Test tensor operations
        x = torch.randn(1000, 1000, device=device)
        
        import time
        start = time.time()
        for _ in range(100):
            y = torch.matmul(x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✅ CUDA tensor test passed")
        print(f"   100 matrix multiplications (1000x1000): {elapsed:.3f}s")
        print(f"   GFLOPS estimate: {(2 * 1000**3 * 100) / elapsed / 1e9:.2f}")
        
        # Memory test
        print(f"\n💾 Memory Status:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"   Max Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
        
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print("✅ Successfully created tensor on MPS device.")
    else:
        print("⚠️  No GPU acceleration detected. Falling back to CPU.")
except Exception as e:
    print(f"❌ Error using device: {e}")

# Recommendations
print("\n📝 Recommendations:")
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version:
        major = int(cuda_version.split('.')[0])
        if major >= 13:
            print("   ✅ Your CUDA setup is optimal for Chatterbox TTS")
            print("   💡 Recommended settings for CUDA 13.1:")
            print("      - torch.backends.cuda.matmul.allow_tf32 = True")
            print("      - torch.backends.cudnn.allow_tf32 = True")
            print("      - torch.backends.cudnn.benchmark = True")
        elif major >= 12:
            print("   ✅ Your CUDA version is supported")
            print("   💡 Consider upgrading to CUDA 13.0+ for latest features")
        else:
            print("   ⚠️  Consider upgrading to CUDA 13.0+ for best performance")
    
    compute_cap = torch.cuda.get_device_capability(0)
    if compute_cap[0] >= 8:
        print("   💡 Ampere+ GPU detected - Enable TF32 for faster inference:")
        print("      torch.backends.cuda.matmul.allow_tf32 = True")
        print("      torch.backends.cudnn.allow_tf32 = True")
    
    # CUDA Toolkit recommendations
    print("\n🔧 For CUDA Toolkit 13.1 installation:")
    print("   - Download from: https://developer.nvidia.com/cuda-13-1-0-download-archive")
    print("   - Ensure cuDNN 9.10+ is installed")
    print("   - Update PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu131")
    
elif torch.backends.mps.is_available():
    print("   ✅ Apple Silicon detected - Use device='mps' for GPU acceleration")
    print("   💡 MPS is optimized for Apple M1/M2/M3/M4 chips")
else:
    print("   ⚠️  No GPU detected - Performance will be limited on CPU")
    print("   💡 For better performance, consider using a system with:")
    print("      - NVIDIA GPU with CUDA 13.1 support")
    print("      - Apple Silicon Mac (M1/M2/M3/M4)")
