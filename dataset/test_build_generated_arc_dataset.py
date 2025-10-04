"""
Unit Tests for Synthetic ARC Puzzle Generator

Tests cover:
- Unit tests for each generator category
- Contract tests for generator interface
- Integration tests for full pipeline
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.build_generated_arc_dataset import (
    ReasoningCategory,
    GeneratorConfig,
    PuzzleGenerator,
    SymmetryGenerator,
    ObjectCountingGenerator,
    PatternRepetitionGenerator,
    RotationGenerator,
    ColorTransformationGenerator,
    SyntheticDatasetBuilder,
    GeneratedPuzzle
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return GeneratorConfig(
        categories=[ReasoningCategory.SYMMETRY],
        num_examples=10,
        output_dir="data/test-synthetic",
        seed=42,
        min_grid_size=3,
        max_grid_size=8
    )


@pytest.fixture
def test_rng():
    """Create test random number generator."""
    return np.random.default_rng(42)


class TestPuzzleGeneratorBase:
    """Test base puzzle generator functionality."""

    def test_generate_random_grid(self, test_config, test_rng):
        """Test random grid generation."""
        generator = PuzzleGenerator(test_config, test_rng)

        grid = generator.generate_random_grid(5, 5, density=0.5)

        assert grid.shape == (5, 5)
        assert grid.dtype == np.uint8
        assert np.all(grid >= 0)
        assert np.all(grid < test_config.num_colors)

    def test_generate_random_grid_sizes(self, test_config, test_rng):
        """Test random grid with random sizes."""
        generator = PuzzleGenerator(test_config, test_rng)

        grid = generator.generate_random_grid()

        assert test_config.min_grid_size <= grid.shape[0] <= test_config.max_grid_size
        assert test_config.min_grid_size <= grid.shape[1] <= test_config.max_grid_size

    def test_puzzle_hash_uniqueness(self, test_config, test_rng):
        """Test puzzle hashing produces unique hashes."""
        generator = PuzzleGenerator(test_config, test_rng)

        examples1 = [(np.array([[1, 2]]), np.array([[3, 4]]))]
        examples2 = [(np.array([[1, 2]]), np.array([[3, 5]]))]

        hash1 = generator.puzzle_hash(examples1)
        hash2 = generator.puzzle_hash(examples2)

        assert hash1 != hash2

    def test_puzzle_hash_consistency(self, test_config, test_rng):
        """Test puzzle hashing is consistent."""
        generator = PuzzleGenerator(test_config, test_rng)

        examples = [(np.array([[1, 2]]), np.array([[3, 4]]))]

        hash1 = generator.puzzle_hash(examples)
        hash2 = generator.puzzle_hash(examples)

        assert hash1 == hash2


class TestSymmetryGenerator:
    """Test symmetry puzzle generator."""

    def test_generates_correct_count(self, test_config, test_rng):
        """Test generator produces requested number of puzzles."""
        generator = SymmetryGenerator(test_config, test_rng)

        puzzles = generator.generate(5)

        assert len(puzzles) <= 5  # May be less due to duplicates

    def test_puzzle_structure(self, test_config, test_rng):
        """Test generated puzzles have correct structure."""
        generator = SymmetryGenerator(test_config, test_rng)

        puzzles = generator.generate(1)
        assert len(puzzles) > 0

        puzzle = puzzles[0]

        assert puzzle.category == ReasoningCategory.SYMMETRY
        assert len(puzzle.examples) == test_config.examples_per_puzzle
        assert puzzle.puzzle_id.startswith("gen_symmetry_")

    def test_symmetry_property(self, test_config, test_rng):
        """Test outputs are actually symmetric."""
        generator = SymmetryGenerator(test_config, test_rng)

        puzzles = generator.generate(3)

        for puzzle in puzzles:
            for input_grid, output_grid in puzzle.examples:
                # Output width should be even (for horizontal symmetry)
                assert output_grid.shape[1] % 2 == 0

                # Left and right halves should be mirror images
                width = output_grid.shape[1]
                half_width = width // 2
                left_half = output_grid[:, :half_width]
                right_half = output_grid[:, half_width:]

                assert np.array_equal(left_half, np.fliplr(right_half))

    def test_deterministic_with_seed(self):
        """Test generation is deterministic with same seed."""
        config = GeneratorConfig(
            categories=[ReasoningCategory.SYMMETRY],
            num_examples=5,
            seed=42
        )

        gen1 = SymmetryGenerator(config, np.random.default_rng(42))
        gen2 = SymmetryGenerator(config, np.random.default_rng(42))

        puzzles1 = gen1.generate(3)
        puzzles2 = gen2.generate(3)

        assert len(puzzles1) == len(puzzles2)

        for p1, p2 in zip(puzzles1, puzzles2):
            for (i1, o1), (i2, o2) in zip(p1.examples, p2.examples):
                assert np.array_equal(i1, i2)
                assert np.array_equal(o1, o2)


class TestObjectCountingGenerator:
    """Test object counting puzzle generator."""

    def test_generates_puzzles(self, test_config, test_rng):
        """Test generator produces puzzles."""
        generator = ObjectCountingGenerator(test_config, test_rng)

        puzzles = generator.generate(5)

        assert len(puzzles) > 0

    def test_counting_property(self, test_config, test_rng):
        """Test outputs correctly count objects in inputs."""
        generator = ObjectCountingGenerator(test_config, test_rng)

        puzzles = generator.generate(3)

        for puzzle in puzzles:
            for input_grid, output_grid in puzzle.examples:
                # Count non-zero elements in input
                count = np.count_nonzero(input_grid)

                # Output should be 1x1 grid with count
                assert output_grid.shape == (1, 1)
                assert output_grid[0, 0] == count

    def test_category_tag(self, test_config, test_rng):
        """Test puzzles are tagged with correct category."""
        generator = ObjectCountingGenerator(test_config, test_rng)

        puzzles = generator.generate(2)

        for puzzle in puzzles:
            assert puzzle.category == ReasoningCategory.OBJECT_COUNTING


class TestPatternRepetitionGenerator:
    """Test pattern repetition puzzle generator."""

    def test_generates_puzzles(self, test_config, test_rng):
        """Test generator produces puzzles."""
        generator = PatternRepetitionGenerator(test_config, test_rng)

        puzzles = generator.generate(5)

        assert len(puzzles) > 0

    def test_repetition_property(self, test_config, test_rng):
        """Test outputs contain repeated pattern from input."""
        generator = PatternRepetitionGenerator(test_config, test_rng)

        puzzles = generator.generate(3)

        for puzzle in puzzles:
            for input_grid, output_grid in puzzle.examples:
                # Output should be tileable by input pattern
                h, w = input_grid.shape
                oh, ow = output_grid.shape

                # Check at least first tile matches
                assert np.array_equal(output_grid[:h, :w], input_grid)


class TestRotationGenerator:
    """Test rotation puzzle generator."""

    def test_generates_puzzles(self, test_config, test_rng):
        """Test generator produces puzzles."""
        generator = RotationGenerator(test_config, test_rng)

        puzzles = generator.generate(5)

        assert len(puzzles) > 0

    def test_rotation_property(self, test_config, test_rng):
        """Test outputs are rotated versions of inputs."""
        generator = RotationGenerator(test_config, test_rng)

        puzzles = generator.generate(3)

        for puzzle in puzzles:
            # All examples in a puzzle should use same rotation
            rotation_k = puzzle.metadata.get("rotation_k")
            assert rotation_k is not None
            assert 1 <= rotation_k <= 3

            for input_grid, output_grid in puzzle.examples:
                expected = np.rot90(input_grid, k=rotation_k)
                assert np.array_equal(output_grid, expected)


class TestColorTransformationGenerator:
    """Test color transformation puzzle generator."""

    def test_generates_puzzles(self, test_config, test_rng):
        """Test generator produces puzzles."""
        generator = ColorTransformationGenerator(test_config, test_rng)

        puzzles = generator.generate(5)

        assert len(puzzles) > 0

    def test_color_mapping_property(self, test_config, test_rng):
        """Test outputs have consistent color transformation within a puzzle."""
        generator = ColorTransformationGenerator(test_config, test_rng)

        puzzles = generator.generate(3)

        for puzzle in puzzles:
            # All examples in same puzzle should use same color permutation
            # Build color map from all examples
            observed_mappings = []

            for inp, out in puzzle.examples:
                example_map = {}
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        in_color = int(inp[r, c])
                        out_color = int(out[r, c])

                        if in_color in example_map:
                            # Same input color should map to same output color
                            assert example_map[in_color] == out_color
                        else:
                            example_map[in_color] = out_color

                observed_mappings.append(example_map)

            # All examples should use compatible mappings (same transformation)
            # Note: Different examples may not see all colors, so we check consistency
            for i in range(len(observed_mappings)):
                for j in range(i + 1, len(observed_mappings)):
                    map1 = observed_mappings[i]
                    map2 = observed_mappings[j]

                    # Check colors that appear in both
                    common_colors = set(map1.keys()) & set(map2.keys())
                    for color in common_colors:
                        assert map1[color] == map2[color], \
                            f"Inconsistent mapping for color {color}: {map1[color]} vs {map2[color]}"


class TestSyntheticDatasetBuilder:
    """Test synthetic dataset builder."""

    def test_builder_initialization(self, test_config):
        """Test builder initializes correctly."""
        builder = SyntheticDatasetBuilder(test_config)

        assert builder.config == test_config
        assert builder.rng is not None

    def test_generate_multiple_categories(self):
        """Test generating puzzles for multiple categories."""
        config = GeneratorConfig(
            categories=[
                ReasoningCategory.SYMMETRY,
                ReasoningCategory.ROTATION
            ],
            num_examples=20,
            seed=42
        )

        builder = SyntheticDatasetBuilder(config)
        puzzles = builder.generate_puzzles()

        # Should have puzzles from both categories
        categories = {p.category for p in puzzles}
        assert ReasoningCategory.SYMMETRY in categories
        assert ReasoningCategory.ROTATION in categories

    def test_save_dataset(self, test_config):
        """Test saving dataset to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.output_dir = tmpdir
            test_config.num_examples = 5

            builder = SyntheticDatasetBuilder(test_config)
            puzzles = builder.generate_puzzles()
            builder.save_dataset(puzzles)

            # Check files were created
            output_path = Path(tmpdir)
            assert output_path.exists()

            # Check metadata file
            metadata_file = output_path / "dataset_metadata.json"
            assert metadata_file.exists()

            with open(metadata_file) as f:
                metadata = json.load(f)
                assert "num_puzzles" in metadata
                assert metadata["num_puzzles"] == len(puzzles)

            # Check puzzle files
            puzzle_files = list(output_path.glob("gen_*.json"))
            assert len(puzzle_files) == len(puzzles)

    def test_puzzle_json_format(self, test_config):
        """Test saved puzzles have correct JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config.output_dir = tmpdir
            test_config.num_examples = 2

            builder = SyntheticDatasetBuilder(test_config)
            puzzles = builder.generate_puzzles()
            builder.save_dataset(puzzles)

            # Load and verify first puzzle
            puzzle_files = list(Path(tmpdir).glob("gen_*.json"))
            assert len(puzzle_files) > 0

            with open(puzzle_files[0]) as f:
                puzzle_data = json.load(f)

                assert "train" in puzzle_data
                assert "test" in puzzle_data
                assert "metadata" in puzzle_data

                # Verify train examples
                assert len(puzzle_data["train"]) == test_config.examples_per_puzzle

                for example in puzzle_data["train"]:
                    assert "input" in example
                    assert "output" in example
                    assert isinstance(example["input"], list)
                    assert isinstance(example["output"], list)


class TestIntegration:
    """Integration tests for full generation pipeline."""

    def test_end_to_end_generation(self):
        """Test complete puzzle generation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GeneratorConfig(
                categories=[
                    ReasoningCategory.SYMMETRY,
                    ReasoningCategory.OBJECT_COUNTING,
                    ReasoningCategory.ROTATION
                ],
                num_examples=30,
                output_dir=tmpdir,
                seed=42
            )

            builder = SyntheticDatasetBuilder(config)

            # Generate
            puzzles = builder.generate_puzzles()
            assert len(puzzles) > 0

            # Save
            builder.save_dataset(puzzles)

            # Verify output
            output_path = Path(tmpdir)
            puzzle_files = list(output_path.glob("gen_*.json"))

            assert len(puzzle_files) == len(puzzles)

            # Verify each file is valid
            for puzzle_file in puzzle_files:
                with open(puzzle_file) as f:
                    data = json.load(f)
                    assert "train" in data
                    assert len(data["train"]) > 0


class TestGeneratorContract:
    """Contract tests ensuring all generators follow interface."""

    @pytest.mark.parametrize("generator_class", [
        SymmetryGenerator,
        ObjectCountingGenerator,
        PatternRepetitionGenerator,
        RotationGenerator,
        ColorTransformationGenerator
    ])
    def test_generator_interface(self, generator_class, test_config, test_rng):
        """Test all generators implement required interface."""
        generator = generator_class(test_config, test_rng)

        # Should have generate method
        assert hasattr(generator, "generate")

        # Generate should return list of GeneratedPuzzle
        puzzles = generator.generate(2)
        assert isinstance(puzzles, list)

        if len(puzzles) > 0:
            assert isinstance(puzzles[0], GeneratedPuzzle)

    @pytest.mark.parametrize("generator_class", [
        SymmetryGenerator,
        ObjectCountingGenerator,
        PatternRepetitionGenerator,
        RotationGenerator,
        ColorTransformationGenerator
    ])
    def test_puzzle_validity(self, generator_class, test_config, test_rng):
        """Test all generators produce valid puzzles."""
        generator = generator_class(test_config, test_rng)
        puzzles = generator.generate(3)

        for puzzle in puzzles:
            # Valid category
            assert isinstance(puzzle.category, ReasoningCategory)

            # Has examples
            assert len(puzzle.examples) > 0

            # Valid examples
            for inp, out in puzzle.examples:
                assert isinstance(inp, np.ndarray)
                assert isinstance(out, np.ndarray)
                assert inp.dtype == np.uint8
                assert out.dtype == np.uint8

                # Values in valid range
                assert np.all(inp >= 0)
                assert np.all(inp < test_config.num_colors)
                assert np.all(out >= 0)
                assert np.all(out < test_config.num_colors)

            # Valid puzzle ID
            assert puzzle.puzzle_id
            assert isinstance(puzzle.puzzle_id, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
