#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Curriculum Learning Module
========================================================

Tests cover:
1. DifficultyScoreConfig validation
2. CurriculumScheduleConfig validation
3. PuzzleDifficultyAnalyzer functionality
4. CurriculumScheduler scheduling logic
5. CurriculumPuzzleDataset wrapper behavior
6. Integration with mock datasets
7. Edge cases and error handling

Author: Ian Shank
Date: October 2025
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.curriculum_learning import (
    DifficultyScoreConfig,
    CurriculumScheduleConfig,
    PuzzleDifficultyAnalyzer,
    CurriculumScheduler,
    CurriculumPuzzleDataset,
    create_curriculum_dataset
)


# =============================================================================
# Test Configuration Classes
# =============================================================================

class TestDifficultyScoreConfig:
    """Test DifficultyScoreConfig validation."""

    def test_default_config(self):
        """Test default configuration is valid."""
        config = DifficultyScoreConfig()
        assert config.color_weight == 0.4
        assert config.grid_size_weight == 0.3
        assert config.pattern_complexity_weight == 0.3
        assert abs(config.color_weight + config.grid_size_weight +
                  config.pattern_complexity_weight - 1.0) < 1e-5

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = DifficultyScoreConfig(
            color_weight=0.5,
            grid_size_weight=0.3,
            pattern_complexity_weight=0.2
        )
        assert config.color_weight == 0.5
        assert config.grid_size_weight == 0.3
        assert config.pattern_complexity_weight == 0.2

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DifficultyScoreConfig(
                color_weight=0.5,
                grid_size_weight=0.3,
                pattern_complexity_weight=0.3  # Sum = 1.1
            )

    def test_equal_weights(self):
        """Test equal weight configuration."""
        config = DifficultyScoreConfig(
            color_weight=1/3,
            grid_size_weight=1/3,
            pattern_complexity_weight=1/3
        )
        # Should be valid (sums to 1.0 within tolerance)
        assert config is not None


class TestCurriculumScheduleConfig:
    """Test CurriculumScheduleConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = CurriculumScheduleConfig()
        assert config.schedule_type == "exponential"
        assert config.initial_threshold == 0.0
        assert config.final_threshold == 1.0
        assert config.warmup_epochs == 0
        assert config.total_epochs == 20
        assert config.exponential_base == 2.0

    def test_linear_schedule(self):
        """Test linear schedule configuration."""
        config = CurriculumScheduleConfig(
            schedule_type="linear",
            total_epochs=10
        )
        assert config.schedule_type == "linear"
        assert config.total_epochs == 10

    def test_invalid_schedule_type(self):
        """Test invalid schedule type raises error."""
        with pytest.raises(ValueError, match="Invalid schedule_type"):
            CurriculumScheduleConfig(schedule_type="invalid")

    def test_invalid_threshold_range(self):
        """Test invalid threshold values raise errors."""
        with pytest.raises(ValueError, match="initial_threshold must be in"):
            CurriculumScheduleConfig(initial_threshold=-0.1)

        with pytest.raises(ValueError, match="final_threshold must be in"):
            CurriculumScheduleConfig(final_threshold=1.5)

    def test_invalid_epoch_values(self):
        """Test invalid epoch values raise errors."""
        with pytest.raises(ValueError, match="warmup_epochs must be"):
            CurriculumScheduleConfig(warmup_epochs=-1)

        with pytest.raises(ValueError, match="total_epochs must be"):
            CurriculumScheduleConfig(total_epochs=0)


# =============================================================================
# Test PuzzleDifficultyAnalyzer
# =============================================================================

class TestPuzzleDifficultyAnalyzer:
    """Test puzzle difficulty analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default config."""
        return PuzzleDifficultyAnalyzer()

    def test_analyze_simple_grid(self, analyzer):
        """Test analyzing a simple grid."""
        grid = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])

        info = analyzer.analyze_grid(grid)
        assert info["color_count"] == 2
        assert info["grid_size"] == 9
        assert info["height"] == 3
        assert info["width"] == 3
        assert info["unique_colors"] == 2

    def test_analyze_empty_grid(self, analyzer):
        """Test analyzing empty grid."""
        grid = np.zeros((5, 5), dtype=int)

        info = analyzer.analyze_grid(grid)
        assert info["color_count"] == 1  # Only color 0
        assert info["grid_size"] == 25

    def test_analyze_complex_grid(self, analyzer):
        """Test analyzing grid with many colors."""
        grid = np.arange(30).reshape(5, 6)

        info = analyzer.analyze_grid(grid)
        assert info["color_count"] == 30
        assert info["grid_size"] == 30

    def test_invalid_grid_shape(self, analyzer):
        """Test that 1D or 3D grids raise errors."""
        with pytest.raises(ValueError, match="Grid must be 2D"):
            analyzer.analyze_grid(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="Grid must be 2D"):
            analyzer.analyze_grid(np.zeros((3, 3, 3)))

    def test_pattern_complexity_identity(self, analyzer):
        """Test pattern complexity for identity transformation."""
        grid = np.array([[1, 2], [3, 4]])

        # Same input and output
        complexity = analyzer.compute_pattern_complexity(grid, grid)

        # Identity should have very low complexity (no changes)
        assert 0.0 <= complexity <= 0.5

    def test_pattern_complexity_size_change(self, analyzer):
        """Test pattern complexity with size change."""
        input_grid = np.array([[1, 2]])
        output_grid = np.array([[1, 2], [3, 4]])

        complexity = analyzer.compute_pattern_complexity(input_grid, output_grid)

        # Size change should increase complexity
        assert complexity > 0.3

    def test_pattern_complexity_color_change(self, analyzer):
        """Test pattern complexity with color changes."""
        input_grid = np.array([[1, 1], [1, 1]])
        output_grid = np.array([[2, 3], [4, 5]])

        complexity = analyzer.compute_pattern_complexity(input_grid, output_grid)

        # Many color changes should increase complexity
        assert complexity > 0.3

    def test_compute_difficulty_simple(self, analyzer):
        """Test difficulty computation for simple puzzle."""
        # Small grid, few colors, identity transform
        input_grid = np.array([[0, 1], [1, 0]])
        output_grid = np.array([[0, 1], [1, 0]])

        difficulty = analyzer.compute_difficulty(input_grid, output_grid)

        # Should be low difficulty
        assert 0.0 <= difficulty <= 0.5

    def test_compute_difficulty_complex(self, analyzer):
        """Test difficulty computation for complex puzzle."""
        # Large grid, many colors
        input_grid = np.arange(100).reshape(10, 10)
        output_grid = np.arange(100, 200).reshape(10, 10)

        difficulty = analyzer.compute_difficulty(input_grid, output_grid)

        # Should be high difficulty
        assert 0.5 <= difficulty <= 1.0

    def test_compute_difficulty_from_example(self, analyzer):
        """Test difficulty computation from example dict."""
        example = {
            "input": [[0, 1], [1, 0]],
            "output": [[0, 1], [1, 0]]
        }

        difficulty = analyzer.compute_difficulty_from_example(example)

        assert 0.0 <= difficulty <= 1.0

    def test_difficulty_bounds(self, analyzer):
        """Test that difficulty is always in [0, 1]."""
        # Test various grid sizes and complexities
        test_cases = [
            (np.zeros((1, 1)), np.zeros((1, 1))),
            (np.ones((30, 30)), np.ones((30, 30))),
            (np.random.randint(0, 10, (15, 15)), np.random.randint(0, 10, (15, 15))),
        ]

        for input_grid, output_grid in test_cases:
            difficulty = analyzer.compute_difficulty(input_grid, output_grid)
            assert 0.0 <= difficulty <= 1.0

    def test_custom_weights(self):
        """Test analyzer with custom difficulty weights."""
        config = DifficultyScoreConfig(
            color_weight=0.5,
            grid_size_weight=0.25,
            pattern_complexity_weight=0.25
        )
        analyzer = PuzzleDifficultyAnalyzer(config)

        input_grid = np.array([[0, 1, 2, 3]])
        output_grid = np.array([[0, 1, 2, 3]])

        difficulty = analyzer.compute_difficulty(input_grid, output_grid)
        assert 0.0 <= difficulty <= 1.0


# =============================================================================
# Test CurriculumScheduler
# =============================================================================

class TestCurriculumScheduler:
    """Test curriculum scheduling logic."""

    def test_linear_schedule(self):
        """Test linear threshold progression."""
        config = CurriculumScheduleConfig(
            schedule_type="linear",
            initial_threshold=0.0,
            final_threshold=1.0,
            total_epochs=10
        )
        scheduler = CurriculumScheduler(config)

        # Check progression
        thresholds = [scheduler.get_threshold(epoch) for epoch in range(11)]

        # Should be approximately linear
        assert thresholds[0] == pytest.approx(0.0, abs=0.01)
        assert thresholds[5] == pytest.approx(0.5, abs=0.01)
        assert thresholds[10] == pytest.approx(1.0, abs=0.01)

    def test_exponential_schedule(self):
        """Test exponential threshold progression."""
        config = CurriculumScheduleConfig(
            schedule_type="exponential",
            initial_threshold=0.0,
            final_threshold=1.0,
            total_epochs=10,
            exponential_base=2.0
        )
        scheduler = CurriculumScheduler(config)

        # Check progression
        thresholds = [scheduler.get_threshold(epoch) for epoch in range(11)]

        # Should start slow and accelerate
        assert thresholds[0] == pytest.approx(0.0, abs=0.01)
        assert thresholds[5] < 0.5  # Less than linear midpoint
        assert thresholds[10] == pytest.approx(1.0, abs=0.01)

        # Check monotonic increase
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1]

    def test_step_schedule(self):
        """Test step-based threshold progression."""
        config = CurriculumScheduleConfig(
            schedule_type="step",
            initial_threshold=0.0,
            final_threshold=1.0,
            total_epochs=8
        )
        scheduler = CurriculumScheduler(config)

        # Get thresholds for each epoch
        thresholds = [scheduler.get_threshold(epoch) for epoch in range(9)]

        # Should have discrete steps (4 steps for 8 epochs)
        assert thresholds[0] == pytest.approx(0.0, abs=0.01)
        assert thresholds[2] == pytest.approx(0.25, abs=0.01)
        assert thresholds[4] == pytest.approx(0.5, abs=0.01)
        assert thresholds[6] == pytest.approx(0.75, abs=0.01)
        assert thresholds[8] == pytest.approx(1.0, abs=0.01)

    def test_warmup_epochs(self):
        """Test warmup period maintains initial threshold."""
        config = CurriculumScheduleConfig(
            schedule_type="linear",
            initial_threshold=0.2,
            final_threshold=1.0,
            warmup_epochs=3,
            total_epochs=10
        )
        scheduler = CurriculumScheduler(config)

        # During warmup, threshold should stay at initial
        assert scheduler.get_threshold(0) == pytest.approx(0.2)
        assert scheduler.get_threshold(1) == pytest.approx(0.2)
        assert scheduler.get_threshold(2) == pytest.approx(0.2)

        # After warmup, should start increasing (epoch 4, not 3, since warmup is epochs 0-2)
        assert scheduler.get_threshold(4) > 0.2

    def test_step_method(self):
        """Test scheduler step() method."""
        config = CurriculumScheduleConfig(
            schedule_type="linear",
            total_epochs=10
        )
        scheduler = CurriculumScheduler(config)

        assert scheduler.current_epoch == 0

        threshold1 = scheduler.step()
        assert scheduler.current_epoch == 1

        threshold2 = scheduler.step()
        assert scheduler.current_epoch == 2
        assert threshold2 > threshold1

    def test_reset_method(self):
        """Test scheduler reset."""
        config = CurriculumScheduleConfig(total_epochs=10)
        scheduler = CurriculumScheduler(config)

        scheduler.step()
        scheduler.step()
        assert scheduler.current_epoch == 2

        scheduler.reset()
        assert scheduler.current_epoch == 0

    def test_threshold_bounds(self):
        """Test that thresholds stay within [0, 1]."""
        config = CurriculumScheduleConfig(
            schedule_type="exponential",
            initial_threshold=0.1,
            final_threshold=0.9,
            total_epochs=20
        )
        scheduler = CurriculumScheduler(config)

        for epoch in range(25):  # Test beyond total_epochs
            threshold = scheduler.get_threshold(epoch)
            assert 0.0 <= threshold <= 1.0


# =============================================================================
# Test CurriculumPuzzleDataset
# =============================================================================

class MockPuzzleDataset(Dataset):
    """Mock dataset for testing curriculum wrapper."""

    def __init__(self, num_puzzles: int = 100):
        """Create mock puzzles with varying difficulty."""
        self.puzzles = []

        for i in range(num_puzzles):
            # Create puzzles with predictable difficulty
            # Use grid size to control difficulty
            size = (i % 20) + 1  # 1 to 20

            puzzle = {
                "input": np.random.randint(0, 3, (size, size)).tolist(),
                "output": np.random.randint(0, 3, (size, size)).tolist(),
                "puzzle_id": f"puzzle_{i}"
            }
            self.puzzles.append(puzzle)

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        return self.puzzles[idx]


class TestCurriculumPuzzleDataset:
    """Test curriculum dataset wrapper."""

    @pytest.fixture
    def base_dataset(self):
        """Create mock base dataset."""
        return MockPuzzleDataset(num_puzzles=50)

    @pytest.fixture
    def curriculum_dataset(self, base_dataset):
        """Create curriculum dataset with default config."""
        return CurriculumPuzzleDataset(
            base_dataset=base_dataset,
            enable_curriculum=True,
            initial_threshold=0.5
        )

    def test_initialization(self, base_dataset):
        """Test dataset initialization."""
        dataset = CurriculumPuzzleDataset(base_dataset, initial_threshold=0.5)

        assert len(dataset.difficulty_scores) == len(base_dataset)
        assert all(0.0 <= score <= 1.0 for score in dataset.difficulty_scores)

    def test_threshold_filtering(self, base_dataset):
        """Test that threshold filters puzzles correctly."""
        dataset = CurriculumPuzzleDataset(base_dataset, initial_threshold=0.3)

        # Low threshold should give fewer puzzles
        initial_count = len(dataset)

        dataset.set_threshold(0.7)
        mid_count = len(dataset)

        dataset.set_threshold(1.0)
        final_count = len(dataset)

        # Should be monotonically increasing
        assert initial_count <= mid_count <= final_count
        assert final_count == len(base_dataset)  # Threshold 1.0 includes all

    def test_disabled_curriculum(self, base_dataset):
        """Test that disabled curriculum includes all puzzles."""
        dataset = CurriculumPuzzleDataset(
            base_dataset,
            enable_curriculum=False,
            initial_threshold=0.1  # Very low threshold
        )

        # Should include all puzzles regardless of threshold
        assert len(dataset) == len(base_dataset)

    def test_getitem(self, curriculum_dataset):
        """Test accessing items from filtered dataset."""
        if len(curriculum_dataset) > 0:
            # Should be able to access items
            item = curriculum_dataset[0]
            assert "input" in item
            assert "output" in item

    def test_getitem_bounds(self, curriculum_dataset):
        """Test index bounds checking."""
        with pytest.raises(IndexError):
            _ = curriculum_dataset[len(curriculum_dataset)]

        with pytest.raises(IndexError):
            _ = curriculum_dataset[-100]

    def test_step_epoch(self, base_dataset):
        """Test stepping through epochs."""
        schedule_config = CurriculumScheduleConfig(
            schedule_type="linear",
            initial_threshold=0.0,
            final_threshold=1.0,
            total_epochs=5
        )

        dataset = CurriculumPuzzleDataset(
            base_dataset,
            schedule_config=schedule_config,
            initial_threshold=0.0
        )

        initial_len = len(dataset)

        # Step through epochs
        for _ in range(5):
            dataset.step_epoch()

        final_len = len(dataset)

        # Should have more puzzles available at end
        assert final_len >= initial_len
        assert final_len == len(base_dataset)  # Should reach 100% at end

    def test_set_threshold_validation(self, curriculum_dataset):
        """Test threshold validation."""
        with pytest.raises(ValueError):
            curriculum_dataset.set_threshold(-0.1)

        with pytest.raises(ValueError):
            curriculum_dataset.set_threshold(1.5)

    def test_get_difficulty_stats(self, curriculum_dataset):
        """Test difficulty statistics."""
        stats = curriculum_dataset.get_difficulty_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert "threshold" in stats

        assert 0.0 <= stats["mean"] <= 1.0
        assert 0.0 <= stats["min"] <= 1.0
        assert 0.0 <= stats["max"] <= 1.0
        assert stats["count"] <= len(curriculum_dataset.base_dataset)

    def test_difficulty_stats_empty(self, base_dataset):
        """Test stats when no puzzles are available."""
        dataset = CurriculumPuzzleDataset(base_dataset, initial_threshold=0.0)

        stats = dataset.get_difficulty_stats()

        # Should handle empty case gracefully
        assert stats["count"] >= 0

    def test_tokenized_format_handling(self):
        """Test handling of tokenized dataset format."""
        class TokenizedDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "inputs": torch.randint(0, 100, (512,)),
                    "targets": torch.randint(0, 100, (512,)),
                }

        tokenized = TokenizedDataset()
        dataset = CurriculumPuzzleDataset(tokenized)

        # Should assign default difficulty of 0.5
        assert all(score == pytest.approx(0.5) for score in dataset.difficulty_scores)

    def test_backward_compatibility(self, base_dataset):
        """Test that threshold=1.0 with enabled curriculum matches disabled."""
        enabled_dataset = CurriculumPuzzleDataset(
            base_dataset,
            enable_curriculum=True,
            initial_threshold=1.0
        )

        disabled_dataset = CurriculumPuzzleDataset(
            base_dataset,
            enable_curriculum=False,
            initial_threshold=0.0  # Shouldn't matter
        )

        # Both should have same length
        assert len(enabled_dataset) == len(disabled_dataset) == len(base_dataset)


# =============================================================================
# Test Factory Function
# =============================================================================

class TestCreateCurriculumDataset:
    """Test curriculum dataset factory function."""

    def test_create_with_defaults(self):
        """Test creating dataset with default config."""
        base = MockPuzzleDataset(10)
        dataset = create_curriculum_dataset(base)

        assert isinstance(dataset, CurriculumPuzzleDataset)
        assert len(dataset) > 0

    def test_create_with_custom_config(self):
        """Test creating dataset with custom config."""
        base = MockPuzzleDataset(10)

        config = {
            "enable": True,
            "difficulty": {
                "color_weight": 0.5,
                "grid_size_weight": 0.3,
                "pattern_complexity_weight": 0.2
            },
            "schedule": {
                "schedule_type": "linear",
                "total_epochs": 15
            },
            "initial_threshold": 0.3
        }

        dataset = create_curriculum_dataset(base, config)

        assert isinstance(dataset, CurriculumPuzzleDataset)
        assert dataset.current_threshold == 0.3
        assert dataset.analyzer.config.color_weight == 0.5

    def test_create_disabled(self):
        """Test creating dataset with curriculum disabled."""
        base = MockPuzzleDataset(10)

        config = {"enable": False}

        dataset = create_curriculum_dataset(base, config)

        assert len(dataset) == len(base)


# =============================================================================
# Integration Tests
# =============================================================================

class TestCurriculumIntegration:
    """Test full curriculum learning pipeline."""

    def test_full_training_simulation(self):
        """Simulate full training with curriculum learning."""
        # Create dataset
        base = MockPuzzleDataset(100)

        schedule_config = CurriculumScheduleConfig(
            schedule_type="exponential",
            initial_threshold=0.2,
            final_threshold=1.0,
            total_epochs=10
        )

        dataset = CurriculumPuzzleDataset(
            base,
            schedule_config=schedule_config,
            initial_threshold=0.2
        )

        # Simulate training epochs
        epoch_sizes = []

        for epoch in range(10):
            epoch_sizes.append(len(dataset))
            dataset.step_epoch()

        # Final epoch should have all puzzles
        epoch_sizes.append(len(dataset))

        # Should be monotonically increasing
        for i in range(len(epoch_sizes) - 1):
            assert epoch_sizes[i] <= epoch_sizes[i + 1]

        # Final should be full dataset
        assert epoch_sizes[-1] == len(base)

    def test_dataloader_compatibility(self):
        """Test compatibility with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        base = MockPuzzleDataset(20)
        dataset = CurriculumPuzzleDataset(base, initial_threshold=0.5)

        # Create DataLoader with custom collate function to handle variable-sized grids
        def collate_fn(batch):
            """Simple collate that just returns list of items"""
            return batch

        # Create DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        # Should be able to iterate
        batch_count = 0
        for batch in loader:
            assert isinstance(batch, list)
            assert len(batch) > 0
            batch_count += 1

        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
