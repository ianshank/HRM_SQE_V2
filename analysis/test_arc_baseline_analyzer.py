"""
Comprehensive unit and integration tests for ARC baseline analyzer.

Tests cover:
- Unit tests for each reasoning category detector
- Integration tests for full analysis pipeline
- Contract tests for report generation
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

import numpy as np
import torch

from arc_baseline_analyzer import (
    ARCPuzzleAnalyzer,
    ARCBaselineAnalyzer,
    ReasoningCategory,
    PuzzleFailure,
    CategoryStatistics,
    BaselineReport
)


class TestReasoningCategoryDetection(unittest.TestCase):
    """Unit tests for reasoning category detection heuristics."""
    
    def setUp(self):
        """Initialize analyzer for each test."""
        self.analyzer = ARCPuzzleAnalyzer()
    
    def test_symmetry_vertical(self):
        """Test detection of vertical symmetry."""
        grid = np.array([
            [1, 2, 1],
            [3, 4, 3],
            [1, 2, 1]
        ])
        self.assertTrue(self.analyzer._check_symmetry(grid))
    
    def test_symmetry_horizontal(self):
        """Test detection of horizontal symmetry."""
        grid = np.array([
            [1, 3, 1],
            [2, 4, 2],
            [1, 3, 1]
        ])
        self.assertTrue(self.analyzer._check_symmetry(grid))
    
    def test_symmetry_diagonal(self):
        """Test detection of diagonal symmetry."""
        grid = np.array([
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6]
        ])
        self.assertTrue(self.analyzer._check_symmetry(grid))
    
    def test_no_symmetry(self):
        """Test that asymmetric grids are not detected as symmetric."""
        grid = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.assertFalse(self.analyzer._check_symmetry(grid))
    
    def test_rotation_90(self):
        """Test detection of 90-degree rotation."""
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[3, 1], [4, 2]])
        self.assertTrue(self.analyzer._check_rotation(inp, out))
    
    def test_rotation_180(self):
        """Test detection of 180-degree rotation."""
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[4, 3], [2, 1]])
        self.assertTrue(self.analyzer._check_rotation(inp, out))
    
    def test_reflection_horizontal(self):
        """Test detection of horizontal reflection."""
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        out = np.array([[3, 2, 1], [6, 5, 4]])
        self.assertTrue(self.analyzer._check_reflection(inp, out))
    
    def test_reflection_vertical(self):
        """Test detection of vertical reflection."""
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        out = np.array([[4, 5, 6], [1, 2, 3]])
        self.assertTrue(self.analyzer._check_reflection(inp, out))
    
    def test_size_scaling_up(self):
        """Test detection of upscaling."""
        inp = np.array([[1, 2], [3, 4]])
        out = np.zeros((6, 6))  # 3x scale
        self.assertTrue(self.analyzer._check_size_scaling(inp, out))
    
    def test_size_scaling_down(self):
        """Test detection of downscaling."""
        inp = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        out = np.array([[1], [2]])  # Much smaller
        self.assertTrue(self.analyzer._check_size_scaling(inp, out))
    
    def test_color_transformation(self):
        """Test detection of color transformation."""
        inp = np.array([[1, 2, 3], [4, 5, 6]])
        out = np.array([[7, 8, 9], [1, 2, 3]])
        self.assertTrue(self.analyzer._check_color_transformation(inp, out))
    
    def test_no_color_transformation(self):
        """Test that same colors are not detected as transformation."""
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[1, 2], [3, 4]])
        self.assertFalse(self.analyzer._check_color_transformation(inp, out))
    
    def test_object_counting(self):
        """Test detection of counting-based puzzles."""
        inp = np.array([[1, 1, 2, 2, 3]])
        out = np.array([[3]])  # Counting unique objects
        self.assertTrue(self.analyzer._check_object_counting(inp, out))
    
    def test_pattern_repetition(self):
        """Test detection of pattern repetition."""
        inp = np.array([[1, 2]])
        out = np.array([[1, 2], [1, 2], [1, 2]])  # Repeated rows
        self.assertTrue(self.analyzer._check_pattern_repetition(inp, out))
    
    def test_spatial_transformation(self):
        """Test detection of spatial shifting."""
        inp = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        out = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])  # Shifted to corner
        self.assertTrue(self.analyzer._check_spatial_transformation(inp, out))
    
    def test_topological_fill(self):
        """Test detection of topological operations."""
        # Create two separate components
        inp = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4]
        ])
        # Merge into one component
        out = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        self.assertTrue(self.analyzer._check_topological(inp, out))
    
    def test_component_counting(self):
        """Test connected component counting."""
        grid = np.array([
            [1, 0, 2],
            [1, 0, 2],
            [0, 0, 0]
        ])
        count = self.analyzer._count_components(grid)
        self.assertEqual(count, 2)
    
    def test_analyze_puzzle_pair_composition(self):
        """Test that complex puzzles get composition category."""
        # Create a puzzle with rotation, reflection, and color change
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[8, 7], [6, 5]])
        
        categories = self.analyzer.analyze_puzzle_pair(inp, out)
        
        # Should detect multiple categories and composition
        self.assertGreater(len(categories), 0)
        if len(categories) > 2:
            self.assertIn(ReasoningCategory.COMPOSITION, categories)
    
    def test_analyze_puzzle_pair_unknown(self):
        """Test that unrecognized patterns get unknown category."""
        # Identical grids - no clear transformation
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[1, 2], [3, 4]])
        
        categories = self.analyzer.analyze_puzzle_pair(inp, out)
        
        # Should either be empty or contain unknown
        self.assertTrue(
            len(categories) == 0 or ReasoningCategory.UNKNOWN in categories
        )


class TestARCBaselineAnalyzer(unittest.TestCase):
    """Integration tests for the complete baseline analyzer."""
    
    def setUp(self):
        """Set up test fixtures with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "dataset"
        self.checkpoint_path = Path(self.temp_dir) / "checkpoint" / "model"
        
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_grid_hash_deterministic(self):
        """Test that grid hashing is deterministic."""
        grid = np.array([[1, 2], [3, 4]])
        hash1 = ARCBaselineAnalyzer.grid_hash(grid)
        hash2 = ARCBaselineAnalyzer.grid_hash(grid)
        self.assertEqual(hash1, hash2)
    
    def test_grid_hash_different_grids(self):
        """Test that different grids produce different hashes."""
        grid1 = np.array([[1, 2], [3, 4]])
        grid2 = np.array([[1, 2], [3, 5]])
        hash1 = ARCBaselineAnalyzer.grid_hash(grid1)
        hash2 = ARCBaselineAnalyzer.grid_hash(grid2)
        self.assertNotEqual(hash1, hash2)
    
    def test_crop_grid_removes_padding(self):
        """Test that crop_grid correctly removes padding."""
        # Create 30x30 grid with content in top-left 2x2
        grid = np.zeros((900,), dtype=np.int64)
        grid = grid.reshape(30, 30)
        grid[0, 0] = 2  # Value 0 (2-2)
        grid[0, 1] = 3  # Value 1 (3-2)
        grid[1, 0] = 4  # Value 2 (4-2)
        grid[1, 1] = 5  # Value 3 (5-2)
        # Add EOS markers
        grid[2, 0:2] = 1
        grid[0:2, 2] = 1
        
        grid = grid.flatten()
        cropped = ARCBaselineAnalyzer.crop_grid(grid)
        
        expected = np.array([[0, 1], [2, 3]])
        np.testing.assert_array_equal(cropped, expected)
    
    def test_inverse_aug_identity(self):
        """Test inverse_aug on non-augmented name."""
        grid = np.array([[1, 2], [3, 4]])
        result = ARCBaselineAnalyzer.inverse_aug("puzzle_name", grid)
        np.testing.assert_array_equal(result, grid)
    
    def test_category_statistics_compute_success_rate(self):
        """Test success rate computation."""
        stats = CategoryStatistics(category=ReasoningCategory.SYMMETRY)
        stats.total_puzzles = 100
        stats.failed_puzzles = 25
        stats.compute_success_rate()
        self.assertAlmostEqual(stats.success_rate, 0.75)
    
    def test_category_statistics_zero_puzzles(self):
        """Test success rate with zero puzzles."""
        stats = CategoryStatistics(category=ReasoningCategory.SYMMETRY)
        stats.total_puzzles = 0
        stats.failed_puzzles = 0
        stats.compute_success_rate()
        self.assertEqual(stats.success_rate, 0.0)
    
    @patch('arc_baseline_analyzer.torch.load')
    @patch('builtins.open', create=True)
    def test_load_predictions_removes_padding(self, mock_open, mock_torch_load):
        """Test that load_predictions correctly removes padding."""
        # Create mock data
        identifier_map = ["<blank>", "puzzle1", "puzzle2"]
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(identifier_map)
        
        # Create mock predictions with padding
        mock_preds = {
            "puzzle_identifiers": torch.tensor([0, 1, 2, 0]),  # 0 is padding
            "inputs": torch.zeros((4, 900)),
            "labels": torch.zeros((4, 900)),
            "logits": torch.zeros((4, 900, 12)),
            "q_halt_logits": torch.zeros((4,))
        }
        mock_torch_load.return_value = mock_preds
        
        # Mock glob to return one file
        with patch.object(Path, 'glob', return_value=[Path("pred.pt")]):
            analyzer = ARCBaselineAnalyzer(str(self.dataset_path), str(self.checkpoint_path))
            _, all_preds = analyzer.load_predictions()
        
        # Should remove padding entries
        self.assertEqual(all_preds["puzzle_identifiers"].shape[0], 2)
        self.assertTrue(torch.all(all_preds["puzzle_identifiers"] != 0))
    
    def test_baseline_report_creation(self):
        """Test creation of baseline report object."""
        category_stats = {
            cat: CategoryStatistics(category=cat, total_puzzles=10, failed_puzzles=2)
            for cat in ReasoningCategory
        }
        
        for stats in category_stats.values():
            stats.compute_success_rate()
        
        report = BaselineReport(
            total_puzzles=100,
            total_failures=20,
            overall_accuracy=0.80,
            category_statistics=category_stats,
            top_weak_categories=[ReasoningCategory.SYMMETRY, ReasoningCategory.ROTATION],
            failure_details=[]
        )
        
        self.assertEqual(report.total_puzzles, 100)
        self.assertEqual(report.total_failures, 20)
        self.assertAlmostEqual(report.overall_accuracy, 0.80)
        self.assertEqual(len(report.top_weak_categories), 2)


class TestReportGeneration(unittest.TestCase):
    """Contract tests for report generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir) / "dataset"
        self.checkpoint_path = Path(self.temp_dir) / "checkpoint" / "model"
        
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_report_creates_file(self):
        """Test that report generation creates a file."""
        analyzer = ARCBaselineAnalyzer(str(self.dataset_path), str(self.checkpoint_path))
        
        # Create minimal report
        category_stats = {
            cat: CategoryStatistics(
                category=cat,
                total_puzzles=10,
                failed_puzzles=2,
                success_rate=0.8
            )
            for cat in ReasoningCategory
        }
        
        report = BaselineReport(
            total_puzzles=100,
            total_failures=20,
            overall_accuracy=0.80,
            category_statistics=category_stats,
            top_weak_categories=[ReasoningCategory.SYMMETRY],
            failure_details=[]
        )
        
        output_path = Path(self.temp_dir) / "report.md"
        analyzer.generate_report(report, output_path)
        
        self.assertTrue(output_path.exists())
    
    def test_generate_report_contains_sections(self):
        """Test that generated report contains required sections."""
        analyzer = ARCBaselineAnalyzer(str(self.dataset_path), str(self.checkpoint_path))
        
        category_stats = {
            cat: CategoryStatistics(
                category=cat,
                total_puzzles=10,
                failed_puzzles=2,
                success_rate=0.8
            )
            for cat in ReasoningCategory
        }
        
        report = BaselineReport(
            total_puzzles=100,
            total_failures=20,
            overall_accuracy=0.80,
            category_statistics=category_stats,
            top_weak_categories=[ReasoningCategory.SYMMETRY, ReasoningCategory.ROTATION],
            failure_details=[]
        )
        
        output_path = Path(self.temp_dir) / "report.md"
        analyzer.generate_report(report, output_path)
        
        with open(output_path, "r") as f:
            content = f.read()
        
        # Check for required sections
        self.assertIn("# ARC Failure Analysis Baseline Report", content)
        self.assertIn("## Overall Performance", content)
        self.assertIn("## Top 5 Weakest Reasoning Categories", content)
        self.assertIn("## All Category Statistics", content)
        self.assertIn("## Recommendations", content)
        
        # Check for key metrics
        self.assertIn("Total Puzzles", content)
        self.assertIn("Total Failures", content)
        self.assertIn("Overall Accuracy", content)
    
    def test_generate_report_formats_percentages(self):
        """Test that report correctly formats percentage values."""
        analyzer = ARCBaselineAnalyzer(str(self.dataset_path), str(self.checkpoint_path))
        
        category_stats = {
            ReasoningCategory.SYMMETRY: CategoryStatistics(
                category=ReasoningCategory.SYMMETRY,
                total_puzzles=100,
                failed_puzzles=25,
                success_rate=0.75
            )
        }
        
        report = BaselineReport(
            total_puzzles=100,
            total_failures=25,
            overall_accuracy=0.8523,
            category_statistics=category_stats,
            top_weak_categories=[ReasoningCategory.SYMMETRY],
            failure_details=[]
        )
        
        output_path = Path(self.temp_dir) / "report.md"
        analyzer.generate_report(report, output_path)
        
        with open(output_path, "r") as f:
            content = f.read()
        
        # Check percentage formatting
        self.assertIn("85.23%", content)  # Overall accuracy
        self.assertIn("75.00%", content)  # Category success rate


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""
    
    def setUp(self):
        """Initialize analyzer for each test."""
        self.analyzer = ARCPuzzleAnalyzer()
    
    def test_empty_grid(self):
        """Test handling of empty grids."""
        empty = np.zeros((0, 0))
        categories = self.analyzer.analyze_puzzle_pair(empty, empty)
        self.assertIsInstance(categories, set)
    
    def test_single_pixel(self):
        """Test handling of single-pixel grids."""
        grid = np.array([[1]])
        categories = self.analyzer.analyze_puzzle_pair(grid, grid)
        self.assertIsInstance(categories, set)
    
    def test_large_grid(self):
        """Test handling of large grids."""
        large = np.random.randint(0, 10, (30, 30))
        categories = self.analyzer.analyze_puzzle_pair(large, large)
        self.assertIsInstance(categories, set)
    
    def test_all_zeros(self):
        """Test handling of grids with all zeros."""
        zeros = np.zeros((5, 5), dtype=np.int32)
        categories = self.analyzer.analyze_puzzle_pair(zeros, zeros)
        self.assertIsInstance(categories, set)
    
    def test_dfs_mark_boundary_conditions(self):
        """Test DFS marking with boundary conditions."""
        grid = np.array([[1, 1], [1, 1]])
        visited = np.zeros_like(grid, dtype=bool)
        
        # Should not raise exception
        self.analyzer._dfs_mark(grid, visited, 0, 0, 1)
        
        # All should be visited
        self.assertTrue(np.all(visited))


if __name__ == "__main__":
    unittest.main(verbosity=2)

