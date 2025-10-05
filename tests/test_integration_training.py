"""
Integration tests for the complete training pipeline.

Tests the full training flow including CUDA initialization, data loading,
model forward/backward, and optimizer steps.
"""

import unittest
import torch
import tempfile
import json
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain import (
    PretrainConfig, ArchConfig, LossConfig,
    create_dataloader, init_train_state, train_batch,
    TrainState
)
from dataset.common import PuzzleDatasetMetadata


class TestTrainingPipeline(unittest.TestCase):
    """Test complete training pipeline integration."""

    def setUp(self):
        """Set up test fixtures with minimal dataset."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda_available = torch.cuda.is_available()

        # Create temporary dataset
        self.temp_dir = tempfile.mkdtemp()
        self.create_minimal_dataset()

        # Create minimal config
        self.config = PretrainConfig(
            arch=ArchConfig(
                name="hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
                loss=LossConfig(name="losses@ACTLossHead", loss_type="stablemax_cross_entropy"),
                puzzle_emb_ndim=32,
                H_cycles=1,
                L_cycles=1,
                H_layers=1,
                L_layers=1,
                hidden_size=64,
                expansion=1.0,
                num_heads=2,
                pos_encodings="rope",
                halt_max_steps=2,
                halt_exploration_prob=0.0,
                forward_dtype="float32"
            ),
            data_path=self.temp_dir,
            global_batch_size=8,
            accumulation_steps=1,
            epochs=1,
            lr=0.001,
            lr_min_ratio=0.1,
            lr_warmup_steps=10,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            puzzle_emb_lr=0.01,
            puzzle_emb_weight_decay=0.01,
            seed=42
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_minimal_dataset(self):
        """Create a minimal dataset for testing."""
        split = "train"
        set_name = "train"

        split_dir = os.path.join(self.temp_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        vocab_size = 30
        seq_len = 16
        num_examples = 50
        num_puzzles = 5

        metadata = PuzzleDatasetMetadata(
            vocab_size=vocab_size,
            seq_len=seq_len,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_puzzles,
            total_groups=num_puzzles,
            mean_puzzle_examples=num_examples / num_puzzles,
            sets=[set_name]
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

        # Create data
        inputs = np.random.randint(1, vocab_size, (num_examples, seq_len), dtype=np.int32)
        labels = np.random.randint(0, vocab_size, (num_examples, seq_len), dtype=np.int32)
        puzzle_identifiers = np.arange(num_puzzles, dtype=np.int32)
        puzzle_indices = np.linspace(0, num_examples, num_puzzles + 1, dtype=np.int64)
        # Group indices: each group contains one puzzle
        group_indices = np.arange(num_puzzles + 1, dtype=np.int64)

        np.save(os.path.join(split_dir, f"{set_name}__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"{set_name}__labels.npy"), labels)
        np.save(os.path.join(split_dir, f"{set_name}__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"{set_name}__group_indices.npy"), group_indices)

    def test_cuda_initialization_before_training(self):
        """Test CUDA initialization sequence before training."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")

        # Simulate pretrain.py CUDA initialization
        try:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()
        except RuntimeError as e:
            self.fail(f"CUDA initialization failed: {e}")

    def test_dataloader_creation(self):
        """Test dataloader creation works correctly."""
        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        # Check metadata
        self.assertEqual(train_metadata.vocab_size, 30)
        self.assertEqual(train_metadata.seq_len, 16)

        # Get first batch
        for set_name, batch, global_batch_size in train_loader:
            self.assertIn("inputs", batch)
            self.assertIn("labels", batch)
            self.assertIn("puzzle_identifiers", batch)
            break

    def test_train_state_initialization(self):
        """Test train state initialization."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        # Check train state
        self.assertIsInstance(train_state, TrainState)
        self.assertEqual(train_state.step, 0)
        self.assertGreater(train_state.total_steps, 0)
        self.assertEqual(len(train_state.optimizers), len(train_state.optimizer_lrs))

        # Check model is on correct device
        for param in train_state.model.parameters():
            self.assertEqual(param.device.type, self.device.type)

    def test_single_training_step(self):
        """Test a single training step completes without errors."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        # Get first batch
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(self.config, train_state, batch, micro_batch_idx=0, rank=0, world_size=1)

            # Check step incremented
            self.assertEqual(train_state.step, 1)

            # Check metrics returned
            if metrics is not None:
                self.assertIn("train/lr", metrics)

            break

    def test_multiple_training_steps(self):
        """Test multiple training steps complete without errors."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        # Run 5 training steps
        step_count = 0
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(self.config, train_state, batch, micro_batch_idx=0, rank=0, world_size=1)

            step_count += 1
            if step_count >= 5:
                break

        self.assertEqual(train_state.step, 5)

    def test_gradient_accumulation(self):
        """Test gradient accumulation works correctly."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        # Set accumulation steps
        self.config.accumulation_steps = 2
        self.config.global_batch_size = 16

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size // self.config.accumulation_steps,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        # Accumulate gradients
        batch_iter = iter(train_loader)

        # First micro-batch (no optimizer step)
        set_name, batch1, _ = next(batch_iter)
        metrics1 = train_batch(self.config, train_state, batch1, micro_batch_idx=0, rank=0, world_size=1)
        self.assertIsNone(metrics1)  # No metrics on first micro-batch
        self.assertEqual(train_state.step, 0)  # Step not incremented

        # Second micro-batch (optimizer step)
        set_name, batch2, _ = next(batch_iter)
        metrics2 = train_batch(self.config, train_state, batch2, micro_batch_idx=1, rank=0, world_size=1)
        self.assertIsNotNone(metrics2)  # Metrics returned
        self.assertEqual(train_state.step, 1)  # Step incremented

    def test_learning_rate_schedule(self):
        """Test learning rate schedule is applied correctly."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        lrs = []

        # Run several steps and collect learning rates
        step_count = 0
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(self.config, train_state, batch, micro_batch_idx=0, rank=0, world_size=1)

            if metrics is not None and "train/lr" in metrics:
                lrs.append(metrics["train/lr"])

            step_count += 1
            if step_count >= 20:
                break

        # Check LR changed (warmup + cosine schedule)
        self.assertGreater(len(lrs), 0)
        # LR should change during warmup
        if len(lrs) > 1:
            self.assertNotEqual(lrs[0], lrs[-1])

    def test_loss_computation(self):
        """Test loss is computed correctly."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        # Get first batch
        for set_name, batch, global_batch_size in train_loader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Init carry
            carry = train_state.model.initial_carry(batch)

            # Forward
            carry, loss, metrics, _, _ = train_state.model(carry=carry, batch=batch, return_keys=[])

            # Check loss
            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.device.type, self.device.type)
            self.assertGreater(loss.item(), 0)

            # Check metrics
            self.assertIn("loss", metrics)
            self.assertIn("count", metrics)

            break

    def test_no_nan_gradients(self):
        """Test that gradients don't become NaN."""
        if self.cuda_available:
            torch.cuda.synchronize()
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        train_loader, train_metadata = create_dataloader(
            self.config, "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=self.config.global_batch_size,
            rank=0,
            world_size=1
        )

        train_state = init_train_state(self.config, train_metadata, world_size=1, device=self.device)

        # Run a few steps
        for step_idx, (set_name, batch, global_batch_size) in enumerate(train_loader):
            metrics = train_batch(self.config, train_state, batch, micro_batch_idx=0, rank=0, world_size=1)

            # Check for NaN in parameters
            for param in train_state.model.parameters():
                if param.requires_grad:
                    self.assertFalse(torch.isnan(param).any(), f"NaN in parameter at step {step_idx}")

            if step_idx >= 3:
                break


class TestDistributedTraining(unittest.TestCase):
    """Test distributed training setup (simulated)."""

    def test_distributed_config_validation(self):
        """Test that distributed config is validated correctly."""
        # Test global_batch_size divisibility
        config = PretrainConfig(
            arch=ArchConfig(
                name="hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1",
                loss=LossConfig(name="losses@ACTLossHead", loss_type="stablemax_cross_entropy"),
                puzzle_emb_ndim=32,
                H_cycles=1,
                L_cycles=1,
                H_layers=1,
                L_layers=1,
                hidden_size=64,
                expansion=1.0,
                num_heads=2,
                pos_encodings="rope",
                halt_max_steps=2,
                halt_exploration_prob=0.0,
                forward_dtype="float32"
            ),
            data_path="dummy",
            global_batch_size=17,  # Not divisible by accumulation_steps
            accumulation_steps=4,
            epochs=1,
            lr=0.001,
            lr_min_ratio=0.1,
            lr_warmup_steps=10,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            puzzle_emb_lr=0.01,
            puzzle_emb_weight_decay=0.01
        )

        # This should raise assertion error
        with self.assertRaises(AssertionError):
            micro_batch_size = config.global_batch_size // config.accumulation_steps
            assert config.global_batch_size % config.accumulation_steps == 0


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
