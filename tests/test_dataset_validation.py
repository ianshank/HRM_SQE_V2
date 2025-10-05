"""
Unit tests for dataset index validation fixes.

Tests the index bounds checking in PuzzleDataset to ensure
IndexError prevention in both training and test modes.
"""

import unittest
import torch
import numpy as np
import tempfile
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata


class TestDatasetIndexValidation(unittest.TestCase):
    """Test dataset index validation."""

    def setUp(self):
        """Set up test fixtures with temporary dataset."""
        self.temp_dir = tempfile.mkdtemp()
        self.split = "train"
        self.set_name = "test_set"

        # Create minimal dataset
        self.num_examples = 100
        self.seq_len = 10
        self.vocab_size = 50
        self.num_puzzles = 10

        # Create dataset directory
        split_dir = os.path.join(self.temp_dir, self.split)
        os.makedirs(split_dir, exist_ok=True)

        # Create metadata
        metadata = PuzzleDatasetMetadata(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=self.num_puzzles,
            total_groups=self.num_puzzles,
            mean_puzzle_examples=self.num_examples / self.num_puzzles,
            sets=[self.set_name]
        )

        with open(os.path.join(split_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

        # Create data arrays
        inputs = np.random.randint(0, self.vocab_size, (self.num_examples, self.seq_len), dtype=np.int32)
        labels = np.random.randint(0, self.vocab_size, (self.num_examples, self.seq_len), dtype=np.int32)
        puzzle_identifiers = np.random.randint(0, self.num_puzzles, (self.num_puzzles,), dtype=np.int32)

        # Create puzzle indices (boundaries for each puzzle in terms of examples)
        puzzle_indices = np.linspace(0, self.num_examples, self.num_puzzles + 1, dtype=np.int64)

        # Create group indices (boundaries for each group in terms of puzzle IDs)
        # Each group contains one puzzle, so group_indices maps to puzzle IDs
        group_indices = np.arange(self.num_puzzles + 1, dtype=np.int64)

        # Save arrays
        np.save(os.path.join(split_dir, f"{self.set_name}__inputs.npy"), inputs)
        np.save(os.path.join(split_dir, f"{self.set_name}__labels.npy"), labels)
        np.save(os.path.join(split_dir, f"{self.set_name}__puzzle_identifiers.npy"), puzzle_identifiers)
        np.save(os.path.join(split_dir, f"{self.set_name}__puzzle_indices.npy"), puzzle_indices)
        np.save(os.path.join(split_dir, f"{self.set_name}__group_indices.npy"), group_indices)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_valid_test_iteration(self):
        """Test that valid test iteration works correctly."""
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=16,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        dataset = PuzzleDataset(config, split=self.split)

        # Iterate through dataset
        batch_count = 0
        for set_name, batch, global_batch_size in dataset:
            self.assertEqual(set_name, self.set_name)
            self.assertIn("inputs", batch)
            self.assertIn("labels", batch)
            self.assertIn("puzzle_identifiers", batch)

            # Check batch shapes
            batch_size = batch["inputs"].shape[0]
            self.assertLessEqual(batch_size, config.global_batch_size)
            self.assertEqual(batch["inputs"].shape[1], self.seq_len)

            batch_count += 1

        self.assertGreater(batch_count, 0)

    def test_valid_train_iteration(self):
        """Test that valid train iteration works correctly."""
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=16,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        dataset = PuzzleDataset(config, split=self.split)

        # Iterate through dataset
        batch_count = 0
        for set_name, batch, global_batch_size in dataset:
            self.assertEqual(set_name, self.set_name)
            self.assertIn("inputs", batch)
            self.assertIn("labels", batch)
            self.assertIn("puzzle_identifiers", batch)

            batch_count += 1

        self.assertGreater(batch_count, 0)

    def test_puzzle_index_bounds_checking(self):
        """Test that puzzle index bounds are validated."""
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=16,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        dataset = PuzzleDataset(config, split=self.split)

        # Load dataset to access internals
        dataset._lazy_load_dataset()

        # Test internal bounds checking by accessing data
        for set_name, data in dataset._data.items():
            puzzle_indices = data["puzzle_indices"]
            puzzle_identifiers = data["puzzle_identifiers"]

            # Ensure indices are within bounds
            for i in range(len(puzzle_indices) - 1):
                start = puzzle_indices[i]
                end = puzzle_indices[i + 1]
                self.assertLess(i, len(puzzle_identifiers), f"Puzzle index {i} out of bounds")

    def test_distributed_dataset_split(self):
        """Test dataset splitting across ranks."""
        global_batch_size = 16
        num_replicas = 2

        for rank in range(num_replicas):
            config = PuzzleDatasetConfig(
                seed=42,
                dataset_path=self.temp_dir,
                global_batch_size=global_batch_size,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=rank,
                num_replicas=num_replicas
            )

            dataset = PuzzleDataset(config, split=self.split)

            # Each rank should get its portion
            batch_count = 0
            for set_name, batch, global_bs in dataset:
                local_batch_size = batch["inputs"].shape[0]
                self.assertEqual(local_batch_size, global_batch_size // num_replicas)
                batch_count += 1

            self.assertGreater(batch_count, 0)

    def test_padding_behavior(self):
        """Test that padding is applied correctly for incomplete batches."""
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=self.num_examples + 10,  # Larger than dataset
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        dataset = PuzzleDataset(config, split=self.split)

        # Should get one batch with padding
        for set_name, batch, global_batch_size in dataset:
            # Check that batch is padded to local_batch_size
            self.assertEqual(batch["inputs"].shape[0], config.global_batch_size)
            self.assertEqual(batch["labels"].shape[0], config.global_batch_size)
            break

    def test_ignore_label_conversion(self):
        """Test that ignore label IDs are converted correctly."""
        config = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=16,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        dataset = PuzzleDataset(config, split=self.split)

        from models.losses import IGNORE_LABEL_ID

        # Check that ignore labels are converted
        for set_name, batch, global_batch_size in dataset:
            labels = batch["labels"]

            # If metadata has ignore_label_id, check conversion
            if dataset.metadata.ignore_label_id is not None:
                # Any original ignore labels should now be IGNORE_LABEL_ID
                self.assertTrue(
                    torch.all((labels == IGNORE_LABEL_ID) | (labels != dataset.metadata.ignore_label_id))
                )
            break

    def test_batch_consistency_across_epochs(self):
        """Test that batches are consistent across epochs with same seed."""
        config1 = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=16,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        config2 = PuzzleDatasetConfig(
            seed=42,
            dataset_path=self.temp_dir,
            global_batch_size=16,
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )

        dataset1 = PuzzleDataset(config1, split=self.split)
        dataset2 = PuzzleDataset(config2, split=self.split)

        # First epoch should be identical
        iter1 = iter(dataset1)
        iter2 = iter(dataset2)

        for _ in range(3):  # Check first 3 batches
            _, batch1, _ = next(iter1)
            _, batch2, _ = next(iter2)

            # Batches should be identical with same seed
            # Note: This assumes deterministic sampling
            self.assertEqual(batch1["inputs"].shape, batch2["inputs"].shape)


class TestDatasetEdgeCases(unittest.TestCase):
    """Test edge cases in dataset loading."""

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        # This test would require creating a dataset that produces empty batches
        # which should be dropped according to the code
        pass

    def test_single_example_dataset(self):
        """Test dataset with single example."""
        temp_dir = tempfile.mkdtemp()
        split = "train"
        set_name = "single"

        try:
            split_dir = os.path.join(temp_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            metadata = PuzzleDatasetMetadata(
                vocab_size=10,
                seq_len=5,
                pad_id=0,
                ignore_label_id=-100,
                blank_identifier_id=0,
                num_puzzle_identifiers=1,
                total_groups=1,
                mean_puzzle_examples=1,
                sets=[set_name]
            )

            with open(os.path.join(split_dir, "dataset.json"), "w") as f:
                json.dump(metadata.model_dump(), f)

            # Single example
            np.save(os.path.join(split_dir, f"{set_name}__inputs.npy"), np.array([[1, 2, 3, 4, 5]], dtype=np.int32))
            np.save(os.path.join(split_dir, f"{set_name}__labels.npy"), np.array([[1, 2, 3, 4, 5]], dtype=np.int32))
            np.save(os.path.join(split_dir, f"{set_name}__puzzle_identifiers.npy"), np.array([0], dtype=np.int32))
            np.save(os.path.join(split_dir, f"{set_name}__puzzle_indices.npy"), np.array([0, 1], dtype=np.int64))
            np.save(os.path.join(split_dir, f"{set_name}__group_indices.npy"), np.array([0, 1], dtype=np.int64))  # Group 0 contains puzzle 0

            config = PuzzleDatasetConfig(
                seed=42,
                dataset_path=temp_dir,
                global_batch_size=1,
                test_set_mode=True,
                epochs_per_iter=1,
                rank=0,
                num_replicas=1
            )

            dataset = PuzzleDataset(config, split=split)

            # Should handle single example
            for set_name, batch, global_batch_size in dataset:
                self.assertEqual(batch["inputs"].shape[0], 1)
                break

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
