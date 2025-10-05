"""
Unit tests for CUDA initialization fixes.

Tests the explicit CUDA context initialization and device handling
to ensure CUBLAS_STATUS_NOT_INITIALIZED errors are prevented.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCUDAInitialization(unittest.TestCase):
    """Test CUDA initialization and context creation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cuda_available = torch.cuda.is_available()

    def test_cuda_availability(self):
        """Test CUDA availability detection."""
        if self.cuda_available:
            self.assertGreater(torch.cuda.device_count(), 0)
            self.assertIsNotNone(torch.cuda.get_device_name(0))

    def test_cuda_synchronization(self):
        """Test CUDA synchronization works without errors."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        # This should not raise CUBLAS_STATUS_NOT_INITIALIZED
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            self.fail(f"CUDA synchronization failed: {e}")

    def test_cuda_warmup_tensor(self):
        """Test CUDA warmup with small tensor allocation."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        device = torch.device("cuda:0")

        # Warmup: allocate and synchronize
        try:
            test_tensor = torch.zeros(1, device=device)
            torch.cuda.synchronize()
            self.assertEqual(test_tensor.device.type, "cuda")
        except RuntimeError as e:
            self.fail(f"CUDA warmup failed: {e}")

    def test_device_initialization_sequence(self):
        """Test the complete CUDA initialization sequence from pretrain.py."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        device = torch.device("cuda:0")

        # Simulate pretrain.py initialization sequence
        try:
            # Step 1: Synchronize
            torch.cuda.synchronize()

            # Step 2: Warmup tensor
            _ = torch.zeros(1, device=device)

            # Step 3: Synchronize again
            torch.cuda.synchronize()

            # Step 4: Test cuBLAS operation (matrix multiply)
            a = torch.randn(10, 10, device=device)
            b = torch.randn(10, 10, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()

            self.assertEqual(c.device.type, "cuda")
            self.assertEqual(c.shape, (10, 10))

        except RuntimeError as e:
            self.fail(f"CUDA initialization sequence failed: {e}")

    def test_multi_device_initialization(self):
        """Test initialization works on multiple GPUs if available."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            device = torch.device(f"cuda:{device_id}")

            try:
                torch.cuda.set_device(device_id)
                torch.cuda.synchronize()
                test_tensor = torch.zeros(1, device=device)
                torch.cuda.synchronize()

                self.assertEqual(test_tensor.device.index, device_id)

            except RuntimeError as e:
                self.fail(f"CUDA initialization failed on device {device_id}: {e}")

    def test_model_to_device_with_sync(self):
        """Test model.to(device) with synchronization."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        device = torch.device("cuda:0")

        # Create simple model
        model = torch.nn.Linear(10, 10)

        # Move to device with synchronization (as in pretrain.py)
        try:
            model = model.to(device)
            torch.cuda.synchronize(device)

            # Verify model parameters are on correct device
            for param in model.parameters():
                self.assertEqual(param.device.type, "cuda")

        except RuntimeError as e:
            self.fail(f"Model to device transfer failed: {e}")

    def test_distributed_cuda_initialization(self):
        """Test CUDA initialization in distributed setting."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        # Simulate distributed environment variables
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device("cuda", local_rank)

            try:
                torch.cuda.set_device(local_rank)
                torch.cuda.synchronize()
                test_tensor = torch.zeros(1, device=device)
                torch.cuda.synchronize()

                self.assertEqual(test_tensor.device.index, local_rank)

            except RuntimeError as e:
                self.fail(f"Distributed CUDA initialization failed: {e}")

    def test_cublas_operations_after_init(self):
        """Test cuBLAS operations work correctly after initialization."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        device = torch.device("cuda:0")

        # Initialize CUDA context
        torch.cuda.synchronize()
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()

        # Test various cuBLAS operations
        try:
            # Matrix multiplication
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = torch.matmul(a, b)

            # Batch matrix multiplication
            batch_a = torch.randn(10, 50, 50, device=device)
            batch_b = torch.randn(10, 50, 50, device=device)
            batch_c = torch.bmm(batch_a, batch_b)

            # Linear layer (uses cuBLAS)
            linear = torch.nn.Linear(100, 100).to(device)
            x = torch.randn(32, 100, device=device)
            y = linear(x)

            torch.cuda.synchronize()

            self.assertEqual(c.device.type, "cuda")
            self.assertEqual(batch_c.device.type, "cuda")
            self.assertEqual(y.device.type, "cuda")

        except RuntimeError as e:
            if "CUBLAS" in str(e):
                self.fail(f"cuBLAS operation failed after initialization: {e}")
            raise


class TestDeviceManagement(unittest.TestCase):
    """Test device management and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.cuda_available = torch.cuda.is_available()

    def test_device_dtype_compatibility(self):
        """Test forward_dtype selection based on device capability."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        from pretrain import _pick_forward_dtype

        device = torch.device("cuda:0")
        major, _ = torch.cuda.get_device_capability(device)

        # Test bfloat16 fallback on older GPUs
        if major < 8:
            picked = _pick_forward_dtype("bfloat16", device)
            self.assertEqual(picked, "float16")
        else:
            picked = _pick_forward_dtype("bfloat16", device)
            self.assertEqual(picked, "bfloat16")

    def test_cpu_device_handling(self):
        """Test that CPU device works correctly."""
        device = torch.device("cpu")

        from pretrain import _pick_forward_dtype

        # Should not modify dtype for CPU
        picked = _pick_forward_dtype("bfloat16", device)
        self.assertEqual(picked, "bfloat16")

    def test_error_handling_invalid_device(self):
        """Test error handling for invalid device."""
        # This should raise an error
        with self.assertRaises((RuntimeError, ValueError)):
            device = torch.device("cuda:999")  # Invalid device ID
            _ = torch.zeros(1, device=device)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
