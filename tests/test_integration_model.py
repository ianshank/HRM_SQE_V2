"""
Integration tests for model initialization with CUDA fixes.

Tests the complete model initialization flow including CUDA context,
device placement, and carry initialization.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.layers import RotaryEmbedding, Attention, SwiGLU
from pretrain import create_model, PretrainConfig, LossConfig, ArchConfig
from dataset.common import PuzzleDatasetMetadata


class TestModelInitialization(unittest.TestCase):
    """Test complete model initialization flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()

    def test_rotary_embedding_cpu_init_then_cuda(self):
        """Test RoPE initializes on CPU then moves to CUDA."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        # Initialize on CPU (default behavior)
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512, base=10000)

        # Check buffers are on CPU
        self.assertEqual(rope.cos_cached.device.type, "cpu")
        self.assertEqual(rope.sin_cached.device.type, "cpu")

        # Move to CUDA
        rope = rope.to(self.device)
        torch.cuda.synchronize()

        # Check buffers moved to CUDA
        self.assertEqual(rope.cos_cached.device.type, "cuda")
        self.assertEqual(rope.sin_cached.device.type, "cuda")

        # Test forward pass
        cos, sin = rope()
        self.assertEqual(cos.device.type, "cuda")
        self.assertEqual(sin.device.type, "cuda")

    def test_hrm_model_initialization(self):
        """Test HRM model initialization with CUDA."""
        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=4,
            seq_len=32,
            puzzle_emb_ndim=64,
            num_puzzle_identifiers=10,
            vocab_size=50,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,
            L_layers=2,
            hidden_size=128,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32"
        )

        model = HierarchicalReasoningModel_ACTV1(config.model_dump())

        # Move to device
        model = model.to(self.device)

        if self.cuda_available:
            torch.cuda.synchronize()

        # Check all parameters are on correct device
        for param in model.parameters():
            self.assertEqual(param.device.type, self.device.type)

        # Check buffers are on correct device
        for buffer in model.buffers():
            self.assertEqual(buffer.device.type, self.device.type)

    def test_carry_initialization_with_cuda(self):
        """Test carry initialization with CUDA synchronization."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=4,
            seq_len=32,
            puzzle_emb_ndim=64,
            num_puzzle_identifiers=10,
            vocab_size=50,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,
            L_layers=2,
            hidden_size=128,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32"
        )

        model = HierarchicalReasoningModel_ACTV1(config.model_dump())
        model = model.to(self.device)
        torch.cuda.synchronize()

        # Test initial_carry
        batch = {
            "inputs": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=self.device),
            "puzzle_identifiers": torch.randint(0, config.num_puzzle_identifiers, (config.batch_size,), device=self.device)
        }

        # Should not raise CUBLAS error
        carry = model.initial_carry(batch)

        # Check carry tensors are on CUDA
        self.assertEqual(carry.inner_carry.z_H.device.type, "cuda")
        self.assertEqual(carry.inner_carry.z_L.device.type, "cuda")
        self.assertEqual(carry.steps.device.type, "cpu")  # Steps starts on CPU
        self.assertEqual(carry.halted.device.type, "cpu")  # Halted starts on CPU

    def test_model_forward_pass(self):
        """Test complete forward pass through model."""
        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=4,
            seq_len=32,
            puzzle_emb_ndim=64,
            num_puzzle_identifiers=10,
            vocab_size=50,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,
            L_layers=2,
            hidden_size=128,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32"
        )

        model = HierarchicalReasoningModel_ACTV1(config.model_dump())
        model = model.to(self.device)

        if self.cuda_available:
            torch.cuda.synchronize()

        batch = {
            "inputs": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=self.device),
            "puzzle_identifiers": torch.randint(0, config.num_puzzle_identifiers, (config.batch_size,), device=self.device)
        }

        carry = model.initial_carry(batch)

        # Forward pass
        new_carry, outputs = model(carry, batch)

        # Check outputs
        self.assertIn("logits", outputs)
        self.assertIn("q_halt_logits", outputs)
        self.assertIn("q_continue_logits", outputs)

        # Check shapes
        self.assertEqual(outputs["logits"].shape, (config.batch_size, config.seq_len, config.vocab_size))

    def test_create_model_with_config(self):
        """Test create_model function from pretrain.py."""
        arch_config = ArchConfig(
            name="hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
            loss=LossConfig(name="losses@ACTLossHead", loss_type="stablemax_cross_entropy"),
            puzzle_emb_ndim=64,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,
            L_layers=2,
            hidden_size=128,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32"
        )

        config = PretrainConfig(
            arch=arch_config,
            data_path="dummy",
            global_batch_size=16,
            accumulation_steps=1,
            epochs=1,
            lr=0.001,
            lr_min_ratio=0.1,
            lr_warmup_steps=100,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            puzzle_emb_lr=0.01,
            puzzle_emb_weight_decay=0.01
        )

        metadata = PuzzleDatasetMetadata(
            vocab_size=50,
            seq_len=32,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=10,
            total_groups=100,
            mean_puzzle_examples=1000,
            sets=["train"]
        )

        # Initialize CUDA if available
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        # Create model
        model, optimizers, optimizer_lrs = create_model(config, metadata, world_size=1, device=self.device)

        # Check model is on correct device
        for param in model.parameters():
            self.assertEqual(param.device.type, self.device.type)

        # Check optimizers were created
        self.assertGreater(len(optimizers), 0)
        self.assertEqual(len(optimizers), len(optimizer_lrs))

    def test_attention_mechanism(self):
        """Test attention mechanism works correctly."""
        hidden_size = 128
        num_heads = 4
        head_dim = hidden_size // num_heads
        seq_len = 32
        batch_size = 4

        attention = Attention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        ).to(self.device)

        # Test with RoPE
        rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len, base=10000).to(self.device)
        cos_sin = rope()

        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=self.device)

        output = attention(cos_sin, hidden_states)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        self.assertEqual(output.device.type, self.device.type)

    def test_swiglu_activation(self):
        """Test SwiGLU activation works correctly."""
        hidden_size = 128
        expansion = 2.0
        batch_size = 4
        seq_len = 32

        swiglu = SwiGLU(hidden_size=hidden_size, expansion=expansion).to(self.device)

        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        output = swiglu(x)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))
        self.assertEqual(output.device.type, self.device.type)


class TestModelDeviceConsistency(unittest.TestCase):
    """Test device consistency across model components."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()

    def test_all_buffers_on_same_device(self):
        """Test all buffers are on the same device after model.to()."""
        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=4,
            seq_len=32,
            puzzle_emb_ndim=64,
            num_puzzle_identifiers=10,
            vocab_size=50,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,
            L_layers=2,
            hidden_size=128,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32"
        )

        model = HierarchicalReasoningModel_ACTV1(config.model_dump())
        model = model.to(self.device)

        # Check all buffers
        for name, buffer in model.named_buffers():
            self.assertEqual(buffer.device.type, self.device.type,
                           f"Buffer {name} on wrong device: {buffer.device}")

    def test_gradient_flow(self):
        """Test gradients flow correctly through model."""
        config = HierarchicalReasoningModel_ACTV1Config(
            batch_size=4,
            seq_len=32,
            puzzle_emb_ndim=64,
            num_puzzle_identifiers=10,
            vocab_size=50,
            H_cycles=2,
            L_cycles=2,
            H_layers=2,
            L_layers=2,
            hidden_size=128,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=4,
            halt_exploration_prob=0.1,
            forward_dtype="float32"
        )

        model = HierarchicalReasoningModel_ACTV1(config.model_dump())
        model = model.to(self.device)
        model.train()

        batch = {
            "inputs": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=self.device),
            "puzzle_identifiers": torch.randint(0, config.num_puzzle_identifiers, (config.batch_size,), device=self.device),
            "labels": torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=self.device)
        }

        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)

        # Compute simple loss
        loss = outputs["logits"].sum()
        loss.backward()

        # Check gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                self.assertEqual(param.grad.device.type, self.device.type)

        self.assertGreater(grad_count, 0, "No gradients computed")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
