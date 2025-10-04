#!/usr/bin/env python3
"""
Curriculum Learning for ARC Puzzle Training
============================================

Implements difficulty-based curriculum learning with:
1. Composite difficulty scoring (colors + grid_size + pattern_complexity)
2. Dynamic threshold scheduling with exponential pacing
3. Configurable via YAML
4. Wrapper class preserving backward compatibility

Author: Ian Shank
Date: October 2025
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DifficultyScoreConfig:
    """Configuration for difficulty scoring weights."""

    color_weight: float = 0.4
    grid_size_weight: float = 0.3
    pattern_complexity_weight: float = 0.3

    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.color_weight + self.grid_size_weight + self.pattern_complexity_weight
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Difficulty weights must sum to 1.0, got {total}")


@dataclass
class CurriculumScheduleConfig:
    """Configuration for curriculum learning schedule."""

    schedule_type: str = "exponential"  # exponential, linear, step
    initial_threshold: float = 0.0
    final_threshold: float = 1.0
    warmup_epochs: int = 0
    total_epochs: int = 20
    exponential_base: float = 2.0

    def __post_init__(self):
        """Validate configuration"""
        if self.schedule_type not in ["exponential", "linear", "step"]:
            raise ValueError(f"Invalid schedule_type: {self.schedule_type}")
        if self.initial_threshold < 0 or self.initial_threshold > 1:
            raise ValueError(f"initial_threshold must be in [0, 1], got {self.initial_threshold}")
        if self.final_threshold < 0 or self.final_threshold > 1:
            raise ValueError(f"final_threshold must be in [0, 1], got {self.final_threshold}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {self.total_epochs}")


class PuzzleDifficultyAnalyzer:
    """
    Analyzes ARC puzzle difficulty using composite scoring.

    Difficulty components:
    1. **Color diversity** (0-1): Normalized number of unique colors
    2. **Grid size** (0-1): Normalized grid area (height × width)
    3. **Pattern complexity** (0-1): Based on output/input ratio and transformations

    Final score = color_weight × colors + grid_size_weight × grid_size + pattern_weight × pattern
    """

    def __init__(
        self,
        config: Optional[DifficultyScoreConfig] = None,
        max_colors: int = 10,
        max_grid_size: int = 900  # 30×30 max for ARC
    ):
        """
        Initialize difficulty analyzer.

        Args:
            config: Difficulty scoring configuration
            max_colors: Maximum number of colors for normalization
            max_grid_size: Maximum grid area for normalization
        """
        self.config = config or DifficultyScoreConfig()
        self.max_colors = max_colors
        self.max_grid_size = max_grid_size

        logger.info(
            f"PuzzleDifficultyAnalyzer initialized: "
            f"weights=[{self.config.color_weight}, {self.config.grid_size_weight}, "
            f"{self.config.pattern_complexity_weight}]"
        )

    def analyze_grid(self, grid: np.ndarray) -> Dict[str, float]:
        """
        Analyze a single grid.

        Args:
            grid: 2D numpy array representing the puzzle grid

        Returns:
            Dictionary with color_count, grid_size, and unique_colors
        """
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)

        if grid.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {grid.shape}")

        height, width = grid.shape
        unique_colors = len(np.unique(grid))
        grid_size = height * width

        return {
            "color_count": unique_colors,
            "grid_size": grid_size,
            "unique_colors": unique_colors,
            "height": height,
            "width": width
        }

    def compute_pattern_complexity(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> float:
        """
        Compute pattern complexity based on input/output relationship.

        Heuristics:
        1. Size change ratio (expansion/contraction)
        2. Color change ratio
        3. Pixel change density

        Args:
            input_grid: Input puzzle grid
            output_grid: Output puzzle grid

        Returns:
            Pattern complexity score in [0, 1]
        """
        input_info = self.analyze_grid(input_grid)
        output_info = self.analyze_grid(output_grid)

        # Size change complexity (0-1)
        size_ratio = output_info["grid_size"] / max(input_info["grid_size"], 1)
        size_complexity = min(abs(math.log2(size_ratio + 1e-6)) / 5.0, 1.0)

        # Color change complexity (0-1)
        color_change = abs(output_info["color_count"] - input_info["color_count"])
        color_complexity = min(color_change / self.max_colors, 1.0)

        # Pixel change density (only if same size)
        if input_info["grid_size"] == output_info["grid_size"]:
            # Resize grids if needed
            if input_grid.shape != output_grid.shape:
                change_density = 1.0  # Different shapes = high complexity
            else:
                changed_pixels = np.sum(input_grid != output_grid)
                total_pixels = input_info["grid_size"]
                change_density = changed_pixels / max(total_pixels, 1)
        else:
            change_density = 1.0  # Size change implies high complexity

        # Combine with equal weights
        pattern_complexity = (size_complexity + color_complexity + change_density) / 3.0

        return float(np.clip(pattern_complexity, 0.0, 1.0))

    def compute_difficulty(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray
    ) -> float:
        """
        Compute overall difficulty score for a puzzle.

        Args:
            input_grid: Input puzzle grid
            output_grid: Output puzzle grid

        Returns:
            Difficulty score in [0, 1]
        """
        # Analyze input grid (use input for color/size metrics)
        input_info = self.analyze_grid(input_grid)

        # Normalize color diversity
        color_score = min(input_info["color_count"] / self.max_colors, 1.0)

        # Normalize grid size
        grid_size_score = min(input_info["grid_size"] / self.max_grid_size, 1.0)

        # Compute pattern complexity
        pattern_score = self.compute_pattern_complexity(input_grid, output_grid)

        # Weighted combination
        difficulty = (
            self.config.color_weight * color_score +
            self.config.grid_size_weight * grid_size_score +
            self.config.pattern_complexity_weight * pattern_score
        )

        return float(np.clip(difficulty, 0.0, 1.0))

    def compute_difficulty_from_example(self, example: Dict[str, Any]) -> float:
        """
        Compute difficulty from a dataset example.

        Args:
            example: Dictionary with "input" and "output" grids

        Returns:
            Difficulty score in [0, 1]
        """
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        return self.compute_difficulty(input_grid, output_grid)


class CurriculumScheduler:
    """
    Manages curriculum learning schedule with dynamic thresholds.

    Supports:
    - Exponential pacing: threshold = initial + (final - initial) * (base^t - 1) / (base - 1)
    - Linear pacing: threshold = initial + (final - initial) * t
    - Step pacing: threshold increases in discrete steps

    where t = normalized epoch progress in [0, 1]
    """

    def __init__(self, config: CurriculumScheduleConfig):
        """
        Initialize curriculum scheduler.

        Args:
            config: Curriculum schedule configuration
        """
        self.config = config
        self.current_epoch = 0

        logger.info(
            f"CurriculumScheduler initialized: "
            f"schedule={self.config.schedule_type}, "
            f"range=[{self.config.initial_threshold}, {self.config.final_threshold}], "
            f"epochs={self.config.total_epochs}"
        )

    def get_threshold(self, epoch: Optional[int] = None) -> float:
        """
        Get difficulty threshold for a given epoch.

        Args:
            epoch: Current epoch (uses self.current_epoch if None)

        Returns:
            Difficulty threshold in [0, 1]
        """
        if epoch is None:
            epoch = self.current_epoch

        # Handle warmup period
        if epoch < self.config.warmup_epochs:
            return self.config.initial_threshold

        # Normalize epoch to [0, 1]
        effective_epoch = epoch - self.config.warmup_epochs
        total_effective = self.config.total_epochs - self.config.warmup_epochs
        t = min(effective_epoch / max(total_effective, 1), 1.0)

        initial = self.config.initial_threshold
        final = self.config.final_threshold

        if self.config.schedule_type == "linear":
            threshold = initial + (final - initial) * t

        elif self.config.schedule_type == "exponential":
            base = self.config.exponential_base
            # Exponential growth: (base^t - 1) / (base - 1)
            if math.isclose(base, 1.0):
                # Degenerate to linear if base = 1
                threshold = initial + (final - initial) * t
            else:
                normalized_growth = (base ** t - 1) / (base - 1)
                threshold = initial + (final - initial) * normalized_growth

        elif self.config.schedule_type == "step":
            # Discrete steps every 25% of training
            step = int(t * 4) / 4.0
            threshold = initial + (final - initial) * step

        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

        return float(np.clip(threshold, 0.0, 1.0))

    def step(self) -> float:
        """
        Advance to next epoch and return new threshold.

        Returns:
            New difficulty threshold
        """
        self.current_epoch += 1
        threshold = self.get_threshold()

        logger.debug(
            f"Curriculum epoch {self.current_epoch}/{self.config.total_epochs}: "
            f"threshold={threshold:.4f}"
        )

        return threshold

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_epoch = 0


class CurriculumPuzzleDataset(Dataset):
    """
    Wrapper dataset that filters puzzles by difficulty threshold.

    Features:
    - Analyzes all puzzles on initialization
    - Caches difficulty scores
    - Filters examples based on current threshold
    - Preserves original dataset interface
    - Fully backward compatible (threshold=1.0 includes all puzzles)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        difficulty_config: Optional[DifficultyScoreConfig] = None,
        schedule_config: Optional[CurriculumScheduleConfig] = None,
        initial_threshold: float = 1.0,
        enable_curriculum: bool = True
    ):
        """
        Initialize curriculum wrapper.

        Args:
            base_dataset: Underlying puzzle dataset
            difficulty_config: Difficulty scoring configuration
            schedule_config: Curriculum schedule configuration
            initial_threshold: Initial difficulty threshold
            enable_curriculum: Whether to enable curriculum filtering
        """
        self.base_dataset = base_dataset
        self.enable_curriculum = enable_curriculum

        # Initialize analyzer and scheduler
        self.analyzer = PuzzleDifficultyAnalyzer(difficulty_config)
        self.scheduler = CurriculumScheduler(
            schedule_config or CurriculumScheduleConfig()
        ) if schedule_config else None

        # Analyze all puzzles and cache difficulty scores
        logger.info("Analyzing puzzle difficulties...")
        self.difficulty_scores: List[float] = []
        self._compute_all_difficulties()

        # Initialize threshold
        self.current_threshold = initial_threshold
        self._update_indices()

        logger.info(
            f"CurriculumPuzzleDataset initialized: "
            f"total={len(self.base_dataset)}, "
            f"available={len(self.available_indices)}, "
            f"threshold={self.current_threshold:.4f}, "
            f"enabled={self.enable_curriculum}"
        )

    def _compute_all_difficulties(self):
        """Compute and cache difficulty scores for all puzzles."""
        for idx in range(len(self.base_dataset)):
            try:
                example = self.base_dataset[idx]

                # Handle different dataset formats
                if isinstance(example, dict):
                    if "input" in example and "output" in example:
                        # Direct input/output format
                        difficulty = self.analyzer.compute_difficulty_from_example(example)
                    elif "inputs" in example and "targets" in example:
                        # Tokenized format - estimate from metadata if available
                        # For now, assign medium difficulty
                        difficulty = 0.5
                        logger.warning(
                            f"Example {idx} has tokenized format, "
                            f"assigning default difficulty=0.5"
                        )
                    else:
                        difficulty = 0.5
                        logger.warning(
                            f"Example {idx} has unknown format, "
                            f"assigning default difficulty=0.5"
                        )
                else:
                    # Unknown format
                    difficulty = 0.5
                    logger.warning(
                        f"Example {idx} has non-dict format, "
                        f"assigning default difficulty=0.5"
                    )

                self.difficulty_scores.append(difficulty)

            except Exception as e:
                logger.warning(f"Failed to compute difficulty for example {idx}: {e}")
                self.difficulty_scores.append(0.5)  # Default to medium difficulty

        # Log difficulty distribution
        scores_array = np.array(self.difficulty_scores)
        logger.info(
            f"Difficulty distribution: "
            f"mean={scores_array.mean():.3f}, "
            f"std={scores_array.std():.3f}, "
            f"min={scores_array.min():.3f}, "
            f"max={scores_array.max():.3f}"
        )

    def _update_indices(self):
        """Update available indices based on current threshold."""
        if not self.enable_curriculum:
            # Include all puzzles
            self.available_indices = list(range(len(self.base_dataset)))
        else:
            # Filter by difficulty threshold
            self.available_indices = [
                idx for idx, score in enumerate(self.difficulty_scores)
                if score <= self.current_threshold
            ]

        logger.debug(
            f"Updated available indices: {len(self.available_indices)} / "
            f"{len(self.base_dataset)} (threshold={self.current_threshold:.4f})"
        )

    def set_threshold(self, threshold: float):
        """
        Set difficulty threshold manually.

        Args:
            threshold: New difficulty threshold in [0, 1]
        """
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")

        self.current_threshold = threshold
        self._update_indices()

        logger.info(
            f"Threshold updated to {threshold:.4f}: "
            f"{len(self.available_indices)} / {len(self.base_dataset)} puzzles available"
        )

    def step_epoch(self) -> float:
        """
        Advance to next epoch using scheduler.

        Returns:
            New difficulty threshold
        """
        if self.scheduler is None:
            logger.warning("No scheduler configured, threshold unchanged")
            return self.current_threshold

        new_threshold = self.scheduler.step()
        self.set_threshold(new_threshold)

        return new_threshold

    def get_difficulty_stats(self) -> Dict[str, float]:
        """
        Get statistics about current difficulty distribution.

        Returns:
            Dictionary with difficulty statistics
        """
        available_scores = [
            self.difficulty_scores[idx] for idx in self.available_indices
        ]

        if not available_scores:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "threshold": self.current_threshold
            }

        scores_array = np.array(available_scores)
        return {
            "mean": float(scores_array.mean()),
            "std": float(scores_array.std()),
            "min": float(scores_array.min()),
            "max": float(scores_array.max()),
            "count": len(available_scores),
            "threshold": self.current_threshold
        }

    def __len__(self) -> int:
        """Return number of available puzzles at current threshold."""
        return len(self.available_indices)

    def __getitem__(self, idx: int):
        """
        Get puzzle by index in filtered dataset.

        Args:
            idx: Index in filtered dataset

        Returns:
            Puzzle example from base dataset
        """
        if idx < 0 or idx >= len(self.available_indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.available_indices)})")

        # Map to base dataset index
        base_idx = self.available_indices[idx]
        return self.base_dataset[base_idx]


def create_curriculum_dataset(
    base_dataset: Dataset,
    curriculum_config: Optional[Dict[str, Any]] = None
) -> CurriculumPuzzleDataset:
    """
    Factory function to create curriculum dataset from config dict.

    Args:
        base_dataset: Underlying puzzle dataset
        curriculum_config: Configuration dictionary with keys:
            - enable: Whether to enable curriculum learning
            - difficulty: DifficultyScoreConfig parameters
            - schedule: CurriculumScheduleConfig parameters
            - initial_threshold: Initial difficulty threshold

    Returns:
        CurriculumPuzzleDataset instance
    """
    if curriculum_config is None:
        curriculum_config = {}

    enable = curriculum_config.get("enable", True)

    # Parse difficulty config
    difficulty_params = curriculum_config.get("difficulty", {})
    difficulty_config = DifficultyScoreConfig(**difficulty_params)

    # Parse schedule config
    schedule_params = curriculum_config.get("schedule", {})
    schedule_config = CurriculumScheduleConfig(**schedule_params)

    initial_threshold = curriculum_config.get("initial_threshold", 1.0)

    return CurriculumPuzzleDataset(
        base_dataset=base_dataset,
        difficulty_config=difficulty_config,
        schedule_config=schedule_config,
        initial_threshold=initial_threshold,
        enable_curriculum=enable
    )
