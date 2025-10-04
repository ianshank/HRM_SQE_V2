"""
ARC Baseline Error Analysis System.

This module provides comprehensive analysis of model failures on the ARC dataset,
categorizing errors by reasoning type and generating actionable insights.
"""

from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
import logging
import hashlib
import sys

import numpy as np
import torch
from numba import njit

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.common import inverse_dihedral_transform


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReasoningCategory(Enum):
    """Enumeration of reasoning categories for ARC puzzles."""
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
    UNKNOWN = "unknown"


@dataclass
class PuzzleFailure:
    """Represents a single puzzle failure with categorization."""
    puzzle_name: str
    input_hash: int
    predicted_hash: int
    ground_truth_hash: int
    reasoning_categories: Set[ReasoningCategory] = field(default_factory=set)
    confidence_score: float = 0.0
    num_steps: int = 0
    

@dataclass
class CategoryStatistics:
    """Statistics for a specific reasoning category."""
    category: ReasoningCategory
    total_puzzles: int = 0
    failed_puzzles: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    failure_examples: List[str] = field(default_factory=list)
    
    def compute_success_rate(self) -> None:
        """Compute success rate based on failures."""
        if self.total_puzzles > 0:
            self.success_rate = 1.0 - (self.failed_puzzles / self.total_puzzles)
        else:
            self.success_rate = 0.0


@dataclass
class BaselineReport:
    """Complete baseline analysis report."""
    total_puzzles: int
    total_failures: int
    overall_accuracy: float
    category_statistics: Dict[ReasoningCategory, CategoryStatistics]
    top_weak_categories: List[ReasoningCategory]
    failure_details: List[PuzzleFailure]
    

class ARCPuzzleAnalyzer:
    """Analyzes ARC puzzle characteristics and categorizes by reasoning type."""
    
    def __init__(self):
        """Initialize the analyzer with pattern detection heuristics."""
        self.analysis_cache: Dict[int, Set[ReasoningCategory]] = {}
    
    def analyze_puzzle_pair(
        self, 
        input_grid: np.ndarray, 
        output_grid: np.ndarray
    ) -> Set[ReasoningCategory]:
        """
        Analyze an input-output pair and determine reasoning categories.
        
        Args:
            input_grid: Input puzzle grid
            output_grid: Expected output grid
            
        Returns:
            Set of reasoning categories required for this puzzle
        """
        categories = set()
        
        # Check symmetry operations
        if self._check_symmetry(output_grid):
            categories.add(ReasoningCategory.SYMMETRY)
        
        # Check rotation
        if self._check_rotation(input_grid, output_grid):
            categories.add(ReasoningCategory.ROTATION)
        
        # Check reflection
        if self._check_reflection(input_grid, output_grid):
            categories.add(ReasoningCategory.REFLECTION)
        
        # Check size scaling
        if self._check_size_scaling(input_grid, output_grid):
            categories.add(ReasoningCategory.SIZE_SCALING)
        
        # Check color transformations
        if self._check_color_transformation(input_grid, output_grid):
            categories.add(ReasoningCategory.COLOR_TRANSFORMATION)
        
        # Check object counting
        if self._check_object_counting(input_grid, output_grid):
            categories.add(ReasoningCategory.OBJECT_COUNTING)
        
        # Check pattern repetition
        if self._check_pattern_repetition(input_grid, output_grid):
            categories.add(ReasoningCategory.PATTERN_REPETITION)
        
        # Check spatial transformations
        if self._check_spatial_transformation(input_grid, output_grid):
            categories.add(ReasoningCategory.SPATIAL_TRANSFORMATION)
        
        # Check topological operations
        if self._check_topological(input_grid, output_grid):
            categories.add(ReasoningCategory.TOPOLOGICAL)
        
        # Check composition (multiple operations)
        if len(categories) > 2:
            categories.add(ReasoningCategory.COMPOSITION)
        
        # Default to unknown if no categories detected
        if not categories:
            categories.add(ReasoningCategory.UNKNOWN)
        
        return categories
    
    def _check_symmetry(self, grid: np.ndarray) -> bool:
        """Check if output exhibits symmetry."""
        # Vertical symmetry
        if np.array_equal(grid, np.flipud(grid)):
            return True
        # Horizontal symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            return True
        # Diagonal symmetry
        if grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T):
            return True
        return False
    
    def _check_rotation(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is a rotation of input."""
        for k in [1, 2, 3]:
            if np.array_equal(out, np.rot90(inp, k)):
                return True
        return False
    
    def _check_reflection(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is a reflection of input."""
        if np.array_equal(out, np.fliplr(inp)) or np.array_equal(out, np.flipud(inp)):
            return True
        return False
    
    def _check_size_scaling(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output is scaled version of input."""
        h_ratio = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 0
        w_ratio = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 0
        return abs(h_ratio - w_ratio) < 0.1 and (h_ratio > 1.5 or h_ratio < 0.5)
    
    def _check_color_transformation(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output involves color permutation or transformation."""
        inp_colors = set(np.unique(inp))
        out_colors = set(np.unique(out))
        # Color set changed but same cardinality suggests permutation
        return len(inp_colors.symmetric_difference(out_colors)) > 0
    
    def _check_object_counting(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if puzzle involves counting objects."""
        # Heuristic: small output with limited colors suggests counting
        return out.size < 20 and len(np.unique(out)) < 4
    
    def _check_pattern_repetition(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if output involves repeating patterns."""
        # Check if output is larger and contains repeated blocks
        if out.size <= inp.size:
            return False
        
        # Simple check for repeating rows or columns
        h, w = out.shape
        if h >= 2 and np.array_equal(out[0], out[1]):
            return True
        if w >= 2 and np.array_equal(out[:, 0], out[:, 1]):
            return True
        return False
    
    def _check_spatial_transformation(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if puzzle involves spatial transformations like shifting."""
        # Check if non-zero elements shifted position
        inp_nonzero = np.argwhere(inp != 0)
        out_nonzero = np.argwhere(out != 0)
        
        if len(inp_nonzero) == 0 or len(out_nonzero) == 0:
            return False
        
        # Check if centroids shifted
        inp_centroid = np.mean(inp_nonzero, axis=0)
        out_centroid = np.mean(out_nonzero, axis=0)
        return np.linalg.norm(inp_centroid - out_centroid) > 2.0
    
    def _check_topological(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if puzzle involves topological operations (fill, connect, etc)."""
        # Heuristic: number of connected components changed
        inp_components = self._count_components(inp)
        out_components = self._count_components(out)
        return inp_components != out_components
    
    def _count_components(self, grid: np.ndarray) -> int:
        """Count connected components in grid (simple 4-connectivity)."""
        visited = np.zeros_like(grid, dtype=bool)
        count = 0
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    self._dfs_mark(grid, visited, i, j, grid[i, j])
                    count += 1
        
        return count
    
    def _dfs_mark(self, grid: np.ndarray, visited: np.ndarray, i: int, j: int, color: int) -> None:
        """Mark connected component via DFS."""
        if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
            return
        if visited[i, j] or grid[i, j] != color:
            return
        
        visited[i, j] = True
        self._dfs_mark(grid, visited, i + 1, j, color)
        self._dfs_mark(grid, visited, i - 1, j, color)
        self._dfs_mark(grid, visited, i, j + 1, color)
        self._dfs_mark(grid, visited, i, j - 1, color)


class ARCBaselineAnalyzer:
    """Main baseline analyzer for ARC evaluation."""
    
    def __init__(
        self,
        dataset_path: str,
        checkpoint_path: str,
        pad_puzzle_identifier: int = 0
    ):
        """
        Initialize baseline analyzer.
        
        Args:
            dataset_path: Path to ARC dataset
            checkpoint_path: Path to model checkpoint predictions
            pad_puzzle_identifier: ID used for padding
        """
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.pad_puzzle_identifier = pad_puzzle_identifier
        self.puzzle_analyzer = ARCPuzzleAnalyzer()
        
        logger.info(f"Initialized analyzer for {dataset_path}")
    
    @staticmethod
    def grid_hash(grid: np.ndarray) -> int:
        """Generate hash for a grid."""
        return hash((grid.tobytes(), grid.shape))
    
    @staticmethod
    @njit
    def crop_grid(grid: np.ndarray) -> np.ndarray:
        """
        Crop grid to remove padding and EOS tokens.
        
        Args:
            grid: 900-element flat grid (30x30)
            
        Returns:
            Cropped grid with values shifted by -2
        """
        grid = grid.reshape(30, 30)
        
        max_area = 0
        max_size = (0, 0)
        nr, nc = grid.shape
        
        num_c = nc
        for num_r in range(1, nr + 1):
            for c in range(1, num_c + 1):
                x = grid[num_r - 1, c - 1]
                if (x < 2) | (x > 11):
                    num_c = c - 1
                    break
            
            area = num_r * num_c
            if area > max_area:
                max_area = area
                max_size = (num_r, num_c)
        
        return grid[:max_size[0], :max_size[1]] - 2
    
    @staticmethod
    def inverse_aug(name: str, grid: np.ndarray) -> np.ndarray:
        """Invert augmentation transformation."""
        if "_" not in name:
            return grid
        
        trans_id, perm = name.split("_")[-2:]
        trans_id = int(trans_id[1:])  # Remove "t" prefix
        inv_perm = np.argsort(list(perm))
        
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]
    
    def load_predictions(self) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """
        Load puzzle identifiers and model predictions.
        
        Returns:
            Tuple of (identifier_map, all_predictions)
        """
        logger.info("Loading puzzle identifiers...")
        with open(self.dataset_path / "identifiers.json", "r") as f:
            identifier_map = json.load(f)
        
        logger.info("Loading model predictions...")
        all_preds = {}
        
        # Load all prediction shards
        for filename in self.checkpoint_path.parent.glob(f"{self.checkpoint_path.name}_all_preds.*"):
            logger.info(f"Loading {filename}...")
            preds = torch.load(filename, map_location="cpu")
            
            for k, v in preds.items():
                all_preds.setdefault(k, [])
                all_preds[k].append(v)
            
            del preds
        
        # Concatenate all shards
        all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
        
        # Remove padding
        mask = all_preds["puzzle_identifiers"] != self.pad_puzzle_identifier
        all_preds = {k: v[mask] for k, v in all_preds.items()}
        
        logger.info(f"Loaded {mask.sum().item()} non-padded predictions")
        
        return identifier_map, all_preds
    
    def analyze_failures(
        self,
        identifier_map: List[str],
        all_preds: Dict[str, torch.Tensor],
        top_k: int = 1
    ) -> BaselineReport:
        """
        Analyze all failures and generate comprehensive report.
        
        Args:
            identifier_map: List mapping IDs to puzzle names
            all_preds: Dictionary of predictions
            top_k: Consider top-k predictions
            
        Returns:
            Complete baseline report with categorized failures
        """
        logger.info("Building puzzle ground truth mappings...")
        
        # Map puzzle names to their input/output pairs
        puzzle_labels = {}
        global_grids = {}  # Hash to grid mapping
        
        for identifier, input_tensor, label_tensor in zip(
            all_preds["puzzle_identifiers"],
            all_preds["inputs"],
            all_preds["labels"]
        ):
            name = identifier_map[identifier.item()]
            
            # Only process non-augmented examples for ground truth
            if "_" not in name:
                puzzle_labels.setdefault(name, {})
                
                input_grid = self.crop_grid(input_tensor.numpy())
                label_grid = self.crop_grid(label_tensor.numpy())
                
                input_hash = self.grid_hash(input_grid)
                label_hash = self.grid_hash(label_grid)
                
                global_grids[input_hash] = input_grid
                global_grids[label_hash] = label_grid
                
                puzzle_labels[name][input_hash] = label_hash
        
        logger.info(f"Found {len(puzzle_labels)} unique puzzles")
        
        # Collect predictions
        logger.info("Collating model predictions...")
        pred_answers = {}
        preds = all_preds["logits"].argmax(-1)
        
        for identifier, input_tensor, pred, q in zip(
            all_preds["puzzle_identifiers"],
            all_preds["inputs"],
            preds,
            all_preds["q_halt_logits"].sigmoid()
        ):
            name = identifier_map[identifier.item()]
            orig_name = name.split("_")[0]
            
            input_grid = input_tensor.numpy()
            input_hash = self.grid_hash(
                self.inverse_aug(name, self.crop_grid(input_grid))
            )
            
            pred_grid = self.inverse_aug(name, self.crop_grid(pred.numpy()))
            pred_hash = self.grid_hash(pred_grid)
            global_grids[pred_hash] = pred_grid
            
            pred_answers.setdefault(orig_name, {})
            pred_answers[orig_name].setdefault(input_hash, [])
            pred_answers[orig_name][input_hash].append((pred_hash, q.item()))
        
        # Analyze failures
        logger.info("Analyzing failures by reasoning category...")
        
        category_stats = {cat: CategoryStatistics(category=cat) for cat in ReasoningCategory}
        failures = []
        correct_count = 0
        total_count = 0
        
        for puzzle_name, tests in puzzle_labels.items():
            total_count += 1
            puzzle_correct = True
            
            for input_hash, label_hash in tests.items():
                predictions = pred_answers[puzzle_name][input_hash]
                
                # Aggregate predictions by voting
                pred_map = {}
                for pred_hash, confidence in predictions:
                    pred_map.setdefault(pred_hash, [0, 0.0])
                    pred_map[pred_hash][0] += 1
                    pred_map[pred_hash][1] += confidence
                
                # Average confidence
                for pred_hash, stats in pred_map.items():
                    stats[1] /= stats[0]
                
                # Sort by vote count and confidence
                sorted_preds = sorted(
                    pred_map.items(),
                    key=lambda x: (x[1][0], x[1][1]),
                    reverse=True
                )
                
                # Check if correct
                top_pred_hash = sorted_preds[0][0]
                is_correct = top_pred_hash == label_hash
                
                if not is_correct:
                    puzzle_correct = False
                    
                    # Categorize failure
                    input_grid = global_grids[input_hash]
                    output_grid = global_grids[label_hash]
                    
                    categories = self.puzzle_analyzer.analyze_puzzle_pair(
                        input_grid, output_grid
                    )
                    
                    failure = PuzzleFailure(
                        puzzle_name=puzzle_name,
                        input_hash=input_hash,
                        predicted_hash=top_pred_hash,
                        ground_truth_hash=label_hash,
                        reasoning_categories=categories,
                        confidence_score=sorted_preds[0][1][1],
                        num_steps=sorted_preds[0][1][0]
                    )
                    
                    failures.append(failure)
                    
                    # Update category statistics
                    for category in categories:
                        category_stats[category].failed_puzzles += 1
                        category_stats[category].failure_examples.append(puzzle_name)
                        category_stats[category].average_confidence += failure.confidence_score
                
                # Update total counts for categories
                input_grid = global_grids[input_hash]
                output_grid = global_grids[label_hash]
                categories = self.puzzle_analyzer.analyze_puzzle_pair(input_grid, output_grid)
                
                for category in categories:
                    category_stats[category].total_puzzles += 1
            
            if puzzle_correct:
                correct_count += 1
        
        # Compute statistics
        for stats in category_stats.values():
            stats.compute_success_rate()
            if stats.failed_puzzles > 0:
                stats.average_confidence /= stats.failed_puzzles
        
        # Identify top weak categories
        weak_categories = sorted(
            [s for s in category_stats.values() if s.total_puzzles > 0],
            key=lambda s: (s.success_rate, -s.total_puzzles)
        )[:5]
        
        overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        report = BaselineReport(
            total_puzzles=total_count,
            total_failures=len(failures),
            overall_accuracy=overall_accuracy,
            category_statistics=category_stats,
            top_weak_categories=[s.category for s in weak_categories],
            failure_details=failures
        )
        
        logger.info(f"Analysis complete: {overall_accuracy:.2%} accuracy")
        
        return report
    
    def generate_report(self, report: BaselineReport, output_path: Path) -> None:
        """
        Generate markdown report from analysis.
        
        Args:
            report: Baseline report to write
            output_path: Path to output markdown file
        """
        logger.info(f"Generating report at {output_path}...")
        
        with open(output_path, "w") as f:
            f.write("# ARC Failure Analysis Baseline Report\n\n")
            f.write(f"**Generated:** {Path(__file__).parent.parent}\n\n")
            f.write(f"**Dataset:** {self.dataset_path}\n\n")
            f.write(f"**Checkpoint:** {self.checkpoint_path}\n\n")
            f.write("---\n\n")
            
            f.write("## Overall Performance\n\n")
            f.write(f"- **Total Puzzles:** {report.total_puzzles}\n")
            f.write(f"- **Total Failures:** {report.total_failures}\n")
            f.write(f"- **Overall Accuracy:** {report.overall_accuracy:.2%}\n\n")
            
            f.write("---\n\n")
            f.write("## Top 5 Weakest Reasoning Categories\n\n")
            
            for idx, category in enumerate(report.top_weak_categories, 1):
                stats = report.category_statistics[category]
                f.write(f"### {idx}. {category.value.replace('_', ' ').title()}\n\n")
                f.write(f"- **Success Rate:** {stats.success_rate:.2%}\n")
                f.write(f"- **Total Puzzles:** {stats.total_puzzles}\n")
                f.write(f"- **Failed Puzzles:** {stats.failed_puzzles}\n")
                f.write(f"- **Average Confidence:** {stats.average_confidence:.3f}\n\n")
                
                if stats.failure_examples:
                    f.write("**Example Failures:**\n")
                    for example in stats.failure_examples[:5]:
                        f.write(f"- `{example}`\n")
                    f.write("\n")
            
            f.write("---\n\n")
            f.write("## All Category Statistics\n\n")
            f.write("| Category | Success Rate | Total | Failed | Avg Confidence |\n")
            f.write("|----------|--------------|-------|--------|----------------|\n")
            
            for category in sorted(ReasoningCategory, key=lambda c: c.value):
                stats = report.category_statistics[category]
                if stats.total_puzzles > 0:
                    f.write(
                        f"| {category.value.replace('_', ' ').title()} "
                        f"| {stats.success_rate:.2%} "
                        f"| {stats.total_puzzles} "
                        f"| {stats.failed_puzzles} "
                        f"| {stats.average_confidence:.3f} |\n"
                    )
            
            f.write("\n---\n\n")
            f.write("## Recommendations\n\n")
            f.write("Based on this analysis, the following actions are recommended:\n\n")
            
            for idx, category in enumerate(report.top_weak_categories[:3], 1):
                stats = report.category_statistics[category]
                f.write(
                    f"{idx}. **Target {category.value.replace('_', ' ').title()}:** "
                    f"Generate ~{min(200, stats.failed_puzzles * 3)} additional training examples\n"
                )
            
            f.write("\n")
        
        logger.info(f"Report written to {output_path}")


def main(
    dataset_path: str = "data/arc-aug-1000",
    checkpoint_path: str = "checkpoints/Arc-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 amphibian-turaco/step_414456",
    output_path: str = "docs/ARC_Failure_Analysis_Baseline.md"
):
    """
    Main entry point for baseline analysis.
    
    Args:
        dataset_path: Path to ARC dataset
        checkpoint_path: Path to model checkpoint
        output_path: Path for output report
    """
    analyzer = ARCBaselineAnalyzer(dataset_path, checkpoint_path)
    identifier_map, predictions = analyzer.load_predictions()
    report = analyzer.analyze_failures(identifier_map, predictions)
    analyzer.generate_report(report, Path(output_path))
    
    logger.info("Baseline analysis complete!")
    

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        main(
            dataset_path=sys.argv[1],
            checkpoint_path=sys.argv[2] if len(sys.argv) > 2 else None,
            output_path=sys.argv[3] if len(sys.argv) > 3 else "docs/ARC_Failure_Analysis_Baseline.md"
        )
    else:
        main()

