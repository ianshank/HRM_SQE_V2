#!/usr/bin/env python3
"""
Validation script for CUDA initialization and indexing fixes.

This script performs a series of checks to validate that the CUDA initialization
and indexing fixes are working correctly.
"""

import sys
import torch
import traceback


def test_cuda_availability():
    """Test if CUDA is available."""
    print("=" * 60)
    print("Test 1: CUDA Availability")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")

    return cuda_available


def test_cuda_initialization(device_id=0):
    """Test CUDA context initialization."""
    print("\n" + "=" * 60)
    print("Test 2: CUDA Context Initialization")
    print("=" * 60)

    try:
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)

        # Synchronize to ensure context is created
        torch.cuda.synchronize()
        print(f"CUDA context initialized on device {device_id}")

        # Warm up with a small tensor
        test_tensor = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        print(f"Test tensor created successfully on {device}")

        return True
    except Exception as e:
        print(f"ERROR: CUDA initialization failed: {e}")
        traceback.print_exc()
        return False


def test_tensor_operations(device_id=0):
    """Test basic tensor operations on CUDA."""
    print("\n" + "=" * 60)
    print("Test 3: Basic Tensor Operations")
    print("=" * 60)

    try:
        device = torch.device(f"cuda:{device_id}")

        # Create tensors
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)

        # Matrix multiplication (uses cuBLAS)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

        print(f"Matrix multiplication successful")
        print(f"Result shape: {c.shape}")
        print(f"Result device: {c.device}")

        return True
    except Exception as e:
        print(f"ERROR: Tensor operations failed: {e}")
        traceback.print_exc()
        return False


def test_embedding_operations(device_id=0):
    """Test embedding operations."""
    print("\n" + "=" * 60)
    print("Test 4: Embedding Operations")
    print("=" * 60)

    try:
        device = torch.device(f"cuda:{device_id}")

        # Create embedding layer
        num_embeddings = 1000
        embedding_dim = 128
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)

        # Test valid indices
        valid_indices = torch.randint(0, num_embeddings, (32,), device=device)
        output = embedding(valid_indices)

        print(f"Valid embedding lookup successful")
        print(f"Output shape: {output.shape}")

        # Test bounds validation (should fail)
        try:
            invalid_indices = torch.tensor([num_embeddings], device=device)
            _ = embedding(invalid_indices)
            print("WARNING: Out-of-bounds index was not caught!")
            return False
        except Exception:
            print("Bounds checking working correctly (out-of-bounds caught)")

        return True
    except Exception as e:
        print(f"ERROR: Embedding operations failed: {e}")
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test model initialization with CUDA."""
    print("\n" + "=" * 60)
    print("Test 5: Model Initialization")
    print("=" * 60)

    try:
        # Import model components
        from models.layers import RotaryEmbedding, Attention

        # Test RotaryEmbedding initialization
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512, base=10000)
        print("RotaryEmbedding initialized on CPU")

        # Move to CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            rope = rope.to(device)
            torch.cuda.synchronize()
            print(f"RotaryEmbedding moved to {device}")

            # Test forward pass
            cos, sin = rope()
            print(f"RoPE forward pass successful: cos device={cos.device}, sin device={sin.device}")

        return True
    except Exception as e:
        print(f"ERROR: Model initialization failed: {e}")
        traceback.print_exc()
        return False


def test_distributed_setup():
    """Test distributed training setup."""
    print("\n" + "=" * 60)
    print("Test 6: Distributed Training Setup")
    print("=" * 60)

    import os

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Distributed environment detected: LOCAL_RANK={local_rank}")

        try:
            import torch.distributed as dist

            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

            rank = dist.get_rank()
            world_size = dist.get_world_size()

            print(f"Distributed initialized: rank={rank}, world_size={world_size}, backend={backend}")

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)

                # Initialize CUDA context
                torch.cuda.synchronize()
                test_tensor = torch.zeros(1, device=device)
                torch.cuda.synchronize()

                print(f"CUDA initialized on device {device}")

            dist.destroy_process_group()
            return True

        except Exception as e:
            print(f"ERROR: Distributed setup failed: {e}")
            traceback.print_exc()
            return False
    else:
        print("Not in distributed environment (LOCAL_RANK not set)")
        return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("CUDA INITIALIZATION AND INDEXING FIXES VALIDATION")
    print("=" * 60)

    results = []

    # Test 1: CUDA Availability
    cuda_available = test_cuda_availability()
    results.append(("CUDA Availability", cuda_available))

    if cuda_available:
        # Test 2: CUDA Initialization
        cuda_init = test_cuda_initialization()
        results.append(("CUDA Initialization", cuda_init))

        if cuda_init:
            # Test 3: Tensor Operations
            tensor_ops = test_tensor_operations()
            results.append(("Tensor Operations", tensor_ops))

            # Test 4: Embedding Operations
            embedding_ops = test_embedding_operations()
            results.append(("Embedding Operations", embedding_ops))
    else:
        print("\nSkipping CUDA-specific tests (CUDA not available)")

    # Test 5: Model Initialization (works on CPU too)
    model_init = test_model_initialization()
    results.append(("Model Initialization", model_init))

    # Test 6: Distributed Setup
    dist_setup = test_distributed_setup()
    results.append(("Distributed Setup", dist_setup))

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[OK]" if result else "[FAIL]"
        print(f"{symbol} {test_name}: {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
