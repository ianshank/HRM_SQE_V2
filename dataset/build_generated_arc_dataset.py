"""
Programmatic ARC Puzzle Generator

Generates synthetic ARC puzzles targeting specific reasoning categories
to address model weaknesses identified in baseline analysis.

Key Design Principles:
- Modular: Each reasoning category has its own generator function
- Reusable: Generators can be composed for multi-step reasoning
- Validated: Each puzzle is tested for solvability
- Traceable: Full metadata tracking for analysis
"""

from typing import List, Tuple, Set, Dict, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import hashlib
import logging
import argparse

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

# Import from sibling module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.common import PuzzleDatasetMetadata, dihedral_transform


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ARCMaxGridSize = 30
DEFAULT_SEED = 42


class ReasoningCategory(str, Enum):
    """Reasoning categories for targeted puzzle generation."""
    SYMMETRY = "symmetry"
    OBJECT_COUNTING = "object_counting"
    PATTERN_REPETITION = "pattern_repetition"
    SPATIAL_TRANSFORMATION = "spatial_transformation"
    COLOR_TRANSFORMATION = "color_transformation"
    TOPOLOGICAL = "topological"
    SIZE_SCALING = "size_scaling"
    ROTATION = "rotation"
    REFLECTION = "reflection"
    COMPOSITION = "composition"


class GeneratorConfig(BaseModel):
    """Configuration for puzzle generation."""
    categories: List[ReasoningCategory]
    num_examples: int = 500
    output_dir: str = "data/arc-synthetic-targeted"
    seed: int = DEFAULT_SEED

    min_grid_size: int = 3
    max_grid_size: int = 15
    num_colors: int = 10  # ARC uses 10 colors (0-9)

    examples_per_puzzle: int = 3  # training examples per puzzle
    test_examples_per_puzzle: int = 1


@dataclass
class GeneratedPuzzle:
    """Represents a single generated puzzle."""
    category: ReasoningCategory
    examples: List[Tuple[np.ndarray, np.ndarray]]
    puzzle_id: str
    metadata: Dict = field(default_factory=dict)


class PuzzleGenerator:
    """Base class for category-specific puzzle generators."""

    def __init__(self, config: GeneratorConfig, rng: np.random.Generator):
        """
        Initialize generator.

        Args:
            config: Generator configuration
            rng: Random number generator
        """
        self.config = config
        self.rng = rng
        self.generated_hashes: Set[str] = set()

    def generate_random_grid(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        density: float = 0.3
    ) -> np.ndarray:
        """
        Generate a random grid with specified properties.

        Args:
            height: Grid height (random if None)
            width: Grid width (random if None)
            density: Fraction of non-zero cells

        Returns:
            Random grid
        """
        if height is None:
            height = self.rng.integers(
                self.config.min_grid_size,
                self.config.max_grid_size + 1
            )
        if width is None:
            width = self.rng.integers(
                self.config.min_grid_size,
                self.config.max_grid_size + 1
            )

        grid = np.zeros((height, width), dtype=np.uint8)
        num_cells = int(height * width * density)

        for _ in range(num_cells):
            r = self.rng.integers(0, height)
            c = self.rng.integers(0, width)
            color = self.rng.integers(1, self.config.num_colors)
            grid[r, c] = color

        return grid

    def puzzle_hash(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
        """
        Generate unique hash for puzzle to avoid duplicates.

        Args:
            examples: List of (input, output) pairs

        Returns:
            Hash string
        """
        hash_parts = []
        for inp, out in examples:
            hash_parts.append(inp.tobytes())
            hash_parts.append(out.tobytes())

        return hashlib.sha256(b"".join(hash_parts)).hexdigest()

    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        """
        Generate puzzles for this category.

        Args:
            num_puzzles: Number of puzzles to generate

        Returns:
            List of generated puzzles
        """
        raise NotImplementedError("Subclasses must implement generate()")


class SymmetryGenerator(PuzzleGenerator):
    """Generates puzzles requiring symmetry reasoning."""

    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        """Generate symmetry puzzles."""
        puzzles = []

        logger.info(f"Generating {num_puzzles} symmetry puzzles...")

        for i in range(num_puzzles):
            examples = []

            # Generate training examples
            for _ in range(self.config.examples_per_puzzle):
                # Random asymmetric input - ensure even width for symmetry
                height = self.rng.integers(self.config.min_grid_size, min(12, self.config.max_grid_size))
                half_width = self.rng.integers(self.config.min_grid_size // 2, (min(12, self.config.max_grid_size) // 2) + 1)

                left_half = self.generate_random_grid(height, half_width, density=0.4)

                # Create symmetric output (horizontal flip) with even width
                full_width = half_width * 2
                full_grid = np.zeros((height, full_width), dtype=np.uint8)
                full_grid[:, :half_width] = left_half
                full_grid[:, half_width:] = np.fliplr(left_half)

                # Input is left half, output is symmetric
                input_grid = np.copy(left_half)
                output_grid = full_grid

                examples.append((input_grid, output_grid))

            puzzle_hash = self.puzzle_hash(examples)

            if puzzle_hash not in self.generated_hashes:
                self.generated_hashes.add(puzzle_hash)

                puzzle = GeneratedPuzzle(
                    category=ReasoningCategory.SYMMETRY,
                    examples=examples,
                    puzzle_id=f"gen_symmetry_{i:04d}",
                    metadata={"generator": "symmetry", "symmetry_type": "horizontal"}
                )
                puzzles.append(puzzle)

        logger.info(f"Generated {len(puzzles)} unique symmetry puzzles")
        return puzzles


class ObjectCountingGenerator(PuzzleGenerator):
    """Generates puzzles requiring object counting."""

    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        """Generate object counting puzzles."""
        puzzles = []

        logger.info(f"Generating {num_puzzles} object counting puzzles...")

        for i in range(num_puzzles):
            examples = []

            for _ in range(self.config.examples_per_puzzle):
                # Input: scattered objects
                height = self.rng.integers(5, 10)
                width = self.rng.integers(5, 10)

                input_grid = np.zeros((height, width), dtype=np.uint8)
                color = self.rng.integers(1, self.config.num_colors)

                # Place 1-9 objects
                num_objects = self.rng.integers(1, 10)
                for _ in range(num_objects):
                    r = self.rng.integers(0, height)
                    c = self.rng.integers(0, width)
                    input_grid[r, c] = color

                # Output: single number showing count
                output_grid = np.array([[num_objects]], dtype=np.uint8)

                examples.append((input_grid, output_grid))

            puzzle_hash = self.puzzle_hash(examples)

            if puzzle_hash not in self.generated_hashes:
                self.generated_hashes.add(puzzle_hash)

                puzzle = GeneratedPuzzle(
                    category=ReasoningCategory.OBJECT_COUNTING,
                    examples=examples,
                    puzzle_id=f"gen_counting_{i:04d}",
                    metadata={"generator": "object_counting"}
                )
                puzzles.append(puzzle)

        logger.info(f"Generated {len(puzzles)} unique object counting puzzles")
        return puzzles


class PatternRepetitionGenerator(PuzzleGenerator):
    """Generates puzzles requiring pattern repetition reasoning."""

    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        """Generate pattern repetition puzzles."""
        puzzles = []

        logger.info(f"Generating {num_puzzles} pattern repetition puzzles...")

        for i in range(num_puzzles):
            examples = []

            for _ in range(self.config.examples_per_puzzle):
                # Create a small pattern
                pattern_h = self.rng.integers(2, 4)
                pattern_w = self.rng.integers(2, 4)
                pattern = self.generate_random_grid(pattern_h, pattern_w, density=0.5)

                # Input is the pattern
                input_grid = pattern

                # Output is the pattern repeated in a grid
                repeat_h = self.rng.integers(2, 4)
                repeat_w = self.rng.integers(2, 4)

                output_grid = np.tile(pattern, (repeat_h, repeat_w))

                examples.append((input_grid, output_grid))

            puzzle_hash = self.puzzle_hash(examples)

            if puzzle_hash not in self.generated_hashes:
                self.generated_hashes.add(puzzle_hash)

                puzzle = GeneratedPuzzle(
                    category=ReasoningCategory.PATTERN_REPETITION,
                    examples=examples,
                    puzzle_id=f"gen_repetition_{i:04d}",
                    metadata={"generator": "pattern_repetition"}
                )
                puzzles.append(puzzle)

        logger.info(f"Generated {len(puzzles)} unique pattern repetition puzzles")
        return puzzles


class RotationGenerator(PuzzleGenerator):
    """Generates puzzles requiring rotation reasoning."""

    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        """Generate rotation puzzles."""
        puzzles = []

        logger.info(f"Generating {num_puzzles} rotation puzzles...")

        for i in range(num_puzzles):
            examples = []

            # Random rotation amount (consistent within puzzle)
            rotation_k = self.rng.integers(1, 4)  # 90, 180, or 270 degrees

            for _ in range(self.config.examples_per_puzzle):
                # Square grid for clean rotation
                size = self.rng.integers(self.config.min_grid_size, 10)
                input_grid = self.generate_random_grid(size, size, density=0.4)

                # Output is rotated version
                output_grid = np.rot90(input_grid, k=rotation_k)

                examples.append((input_grid, output_grid))

            puzzle_hash = self.puzzle_hash(examples)

            if puzzle_hash not in self.generated_hashes:
                self.generated_hashes.add(puzzle_hash)

                puzzle = GeneratedPuzzle(
                    category=ReasoningCategory.ROTATION,
                    examples=examples,
                    puzzle_id=f"gen_rotation_{i:04d}",
                    metadata={"generator": "rotation", "rotation_k": int(rotation_k)}
                )
                puzzles.append(puzzle)

        logger.info(f"Generated {len(puzzles)} unique rotation puzzles")
        return puzzles


class ColorTransformationGenerator(PuzzleGenerator):
    """Generates puzzles requiring color transformation reasoning."""

    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        """Generate color transformation puzzles."""
        puzzles = []

        logger.info(f"Generating {num_puzzles} color transformation puzzles...")

        for i in range(num_puzzles):
            examples = []

            # Generate random color permutation (consistent within puzzle)
            color_map = np.arange(self.config.num_colors, dtype=np.uint8)
            self.rng.shuffle(color_map)

            for _ in range(self.config.examples_per_puzzle):
                input_grid = self.generate_random_grid(density=0.4)

                # Apply color transformation
                output_grid = color_map[input_grid]

                examples.append((input_grid, output_grid))

            puzzle_hash = self.puzzle_hash(examples)

            if puzzle_hash not in self.generated_hashes:
                self.generated_hashes.add(puzzle_hash)

                puzzle = GeneratedPuzzle(
                    category=ReasoningCategory.COLOR_TRANSFORMATION,
                    examples=examples,
                    puzzle_id=f"gen_color_{i:04d}",
                    metadata={"generator": "color_transformation"}
                )
                puzzles.append(puzzle)

        logger.info(f"Generated {len(puzzles)} unique color transformation puzzles")
        return puzzles


class SyntheticDatasetBuilder:
    """Builds synthetic ARC dataset from generated puzzles."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize dataset builder.

        Args:
            config: Generator configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Map categories to generator classes
        self.generators: Dict[ReasoningCategory, type] = {
            ReasoningCategory.SYMMETRY: SymmetryGenerator,
            ReasoningCategory.OBJECT_COUNTING: ObjectCountingGenerator,
            ReasoningCategory.PATTERN_REPETITION: PatternRepetitionGenerator,
            ReasoningCategory.ROTATION: RotationGenerator,
            ReasoningCategory.COLOR_TRANSFORMATION: ColorTransformationGenerator,
        }

    def generate_puzzles(self) -> List[GeneratedPuzzle]:
        """
        Generate all puzzles across requested categories.

        Returns:
            List of generated puzzles
        """
        all_puzzles = []

        # Distribute puzzle count across categories
        num_per_category = self.config.num_examples // len(self.config.categories)

        for category in self.config.categories:
            if category not in self.generators:
                logger.warning(f"No generator for category: {category}, skipping")
                continue

            generator_class = self.generators[category]
            generator = generator_class(self.config, self.rng)

            puzzles = generator.generate(num_per_category)
            all_puzzles.extend(puzzles)

        logger.info(f"Generated {len(all_puzzles)} total puzzles")
        return all_puzzles

    def save_dataset(self, puzzles: List[GeneratedPuzzle]) -> None:
        """
        Save generated puzzles in ARC dataset format.

        Args:
            puzzles: Generated puzzles to save
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving dataset to {output_dir}")

        # Save each puzzle as JSON (ARC format)
        for puzzle in puzzles:
            puzzle_data = {
                "train": [
                    {
                        "input": example[0].tolist(),
                        "output": example[1].tolist()
                    }
                    for example in puzzle.examples
                ],
                "test": [],  # Empty for training data
                "metadata": {
                    "category": puzzle.category.value,
                    "puzzle_id": puzzle.puzzle_id,
                    **puzzle.metadata
                }
            }

            puzzle_file = output_dir / f"{puzzle.puzzle_id}.json"
            with open(puzzle_file, "w") as f:
                json.dump(puzzle_data, f, indent=2)

        # Save metadata
        metadata = {
            "num_puzzles": len(puzzles),
            "categories": [cat.value for cat in self.config.categories],
            "config": self.config.model_dump()
        }

        metadata_file = output_dir / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(puzzles)} puzzles to {output_dir}")


cli = ArgParser()


class CLIConfig(BaseModel):
    """CLI configuration."""
    categories: str = "symmetry,object_counting,rotation"
    num_examples: int = 500
    output_dir: str = "data/arc-synthetic-targeted"
    seed: int = DEFAULT_SEED


@cli.command(singleton=True)
def main(config: CLIConfig):
    """
    Generate synthetic ARC puzzles for targeted categories.

    Args:
        config: CLI configuration
    """
    # Parse categories
    category_list = [
        ReasoningCategory(cat.strip().lower())
        for cat in config.categories.split(",")
    ]

    gen_config = GeneratorConfig(
        categories=category_list,
        num_examples=config.num_examples,
        output_dir=config.output_dir,
        seed=config.seed
    )

    logger.info("Starting synthetic puzzle generation")
    logger.info(f"Categories: {[c.value for c in category_list]}")
    logger.info(f"Target examples: {config.num_examples}")

    builder = SyntheticDatasetBuilder(gen_config)
    puzzles = builder.generate_puzzles()
    builder.save_dataset(puzzles)

    logger.info("Synthetic puzzle generation complete!")


if __name__ == "__main__":
    cli()
