"""
Phase 1 Workflow: Foundational Improvements - Data Generation & Baseline

This script automates the complete Phase 1 workflow:
1. Train baseline model
2. Generate predictions
3. Analyze failures
4. Generate targeted synthetic data
5. Retrain and evaluate

Adapted for single GPU (GTX 1660 Ti 6GB).
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse

import torch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase1_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for Phase 1 workflow."""
    dataset_path: str = "data/arc-aug-100"
    checkpoint_dir: str = "checkpoints"
    docs_dir: str = "docs"

    # Training config
    config_file: str = "cfg_pretrain_single_gpu"
    training_steps: int = 50000  # Reduced for single GPU
    eval_interval: int = 5000

    # Synthetic data generation
    num_synthetic_examples: int = 500
    target_improvement_pct: float = 5.0

    # Paths
    baseline_analysis_path: str = "docs/ARC_Failure_Analysis_Baseline.md"
    synthetic_dataset_path: str = "data/arc-synthetic-targeted"
    combined_dataset_path: str = "data/arc-combined-100"


class Phase1Workflow:
    """Manages the complete Phase 1 workflow execution."""

    def __init__(self, config: WorkflowConfig, skip_training: bool = False):
        """
        Initialize workflow manager.

        Args:
            config: Workflow configuration
            skip_training: If True, skip training steps (for testing data pipeline)
        """
        self.config = config
        self.skip_training = skip_training
        self.checkpoint_path: Optional[Path] = None
        self.weak_categories: List[str] = []

        os.makedirs("logs", exist_ok=True)
        os.makedirs(self.config.docs_dir, exist_ok=True)

        logger.info("Phase 1 Workflow initialized")
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Skip training: {self.skip_training}")

    def step_1_train_baseline(self) -> bool:
        """
        Step 1: Train baseline HRM model.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Training Baseline HRM Model")
        logger.info("=" * 80)

        if self.skip_training:
            logger.warning("Skipping training (skip_training=True)")
            logger.info("Looking for existing checkpoints...")

            checkpoint_dir = Path(self.config.checkpoint_dir)
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.rglob("step_*"))
                if checkpoints:
                    self.checkpoint_path = checkpoints[-1].parent
                    logger.info(f"Found checkpoint: {self.checkpoint_path}")
                    return True

            logger.error("No existing checkpoints found and training skipped")
            return False

        try:
            cmd = [
                "python", "pretrain.py",
                f"--config-name={self.config.config_file}",
                f"epochs={self.config.training_steps}",
                f"eval_interval={self.config.eval_interval}"
            ]

            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info("This will take several hours on single GPU...")
            logger.info("Monitor with: tail -f logs/phase1_workflow.log")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info("Training completed successfully")

            # Find the latest checkpoint
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoints = list(checkpoint_dir.rglob("step_*"))

            if not checkpoints:
                logger.error("No checkpoints found after training")
                return False

            self.checkpoint_path = checkpoints[-1].parent
            logger.info(f"Checkpoint saved at: {self.checkpoint_path}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during training: {e}")
            return False

    def step_2_generate_predictions(self) -> bool:
        """
        Step 2: Generate predictions on validation set.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Generating Predictions on Validation Set")
        logger.info("=" * 80)

        if self.checkpoint_path is None:
            logger.error("No checkpoint path set")
            return False

        try:
            # Find the latest checkpoint step file
            step_files = list(self.checkpoint_path.glob("step_*"))
            step_files = [f for f in step_files if f.is_file() and not f.name.endswith(".log")]

            if not step_files:
                logger.error(f"No checkpoint step files found in {self.checkpoint_path}")
                return False

            latest_checkpoint = max(step_files, key=lambda p: int(p.stem.split('_')[1]))

            logger.info(f"Using checkpoint: {latest_checkpoint}")

            cmd = [
                "python", "evaluate.py",
                f"checkpoint={latest_checkpoint}"
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info("Predictions generated successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Prediction generation failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            return False

    def step_3_analyze_failures(self) -> bool:
        """
        Step 3: Analyze failures and identify weak categories.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Analyzing Failures by Reasoning Category")
        logger.info("=" * 80)

        if self.checkpoint_path is None:
            logger.error("No checkpoint path set")
            return False

        try:
            # Import and run analyzer
            sys.path.insert(0, str(Path(__file__).parent))
            from analysis.arc_baseline_analyzer import ARCBaselineAnalyzer

            # Find checkpoint with predictions
            checkpoint_files = list(self.checkpoint_path.glob("step_*"))
            checkpoint_files = [f for f in checkpoint_files if f.is_file() and not f.name.endswith("_all_preds.*")]

            if not checkpoint_files:
                logger.error("No checkpoint files found")
                return False

            latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[1]))

            logger.info(f"Analyzing predictions from: {latest_checkpoint}")

            analyzer = ARCBaselineAnalyzer(
                dataset_path=self.config.dataset_path,
                checkpoint_path=str(latest_checkpoint)
            )

            identifier_map, predictions = analyzer.load_predictions()
            report = analyzer.analyze_failures(identifier_map, predictions)
            analyzer.generate_report(report, Path(self.config.baseline_analysis_path))

            # Extract weak categories
            self.weak_categories = [cat.value for cat in report.top_weak_categories[:3]]

            logger.info(f"Analysis complete. Top 3 weak categories: {self.weak_categories}")
            logger.info(f"Report saved to: {self.config.baseline_analysis_path}")
            logger.info(f"Overall accuracy: {report.overall_accuracy:.2%}")

            return True

        except Exception as e:
            logger.error(f"Failure analysis failed: {e}")
            logger.exception("Full traceback:")
            return False

    def step_4_generate_synthetic_data(self) -> bool:
        """
        Step 4: Generate targeted synthetic puzzles for weak categories.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Generating Targeted Synthetic Data")
        logger.info("=" * 80)

        if not self.weak_categories:
            logger.error("No weak categories identified")
            return False

        try:
            logger.info(f"Targeting categories: {self.weak_categories}")
            logger.info(f"Generating {self.config.num_synthetic_examples} examples")

            cmd = [
                "python", "dataset/build_generated_arc_dataset.py",
                f"--categories={','.join(self.weak_categories)}",
                f"--num-examples={self.config.num_synthetic_examples}",
                f"--output-dir={self.config.synthetic_dataset_path}"
            ]

            logger.info(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info("Synthetic data generated successfully")
            logger.info(f"Dataset saved to: {self.config.synthetic_dataset_path}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Synthetic data generation failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during synthetic generation: {e}")
            return False

    def step_5_merge_datasets(self) -> bool:
        """
        Step 5: Merge original and synthetic datasets.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Merging Original and Synthetic Datasets")
        logger.info("=" * 80)

        try:
            # TODO: Implement dataset merging logic
            logger.warning("Dataset merging not yet implemented - placeholder")
            logger.info(f"Would merge {self.config.dataset_path} + {self.config.synthetic_dataset_path}")
            logger.info(f"Output: {self.config.combined_dataset_path}")

            return True

        except Exception as e:
            logger.error(f"Dataset merging failed: {e}")
            return False

    def step_6_retrain_model(self) -> bool:
        """
        Step 6: Retrain model on combined dataset.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Retraining Model on Combined Dataset")
        logger.info("=" * 80)

        if self.skip_training:
            logger.warning("Skipping retraining (skip_training=True)")
            return True

        try:
            cmd = [
                "python", "pretrain.py",
                f"--config-name={self.config.config_file}",
                f"data_path={self.config.combined_dataset_path}",
                f"epochs={self.config.training_steps}",
                f"eval_interval={self.config.eval_interval}"
            ]

            logger.info(f"Running command: {' '.join(cmd)}")
            logger.info("This will take several hours...")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            logger.info("Retraining completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Retraining failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during retraining: {e}")
            return False

    def step_7_final_evaluation(self) -> bool:
        """
        Step 7: Final evaluation and performance comparison.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 7: Final Evaluation and Performance Comparison")
        logger.info("=" * 80)

        try:
            # TODO: Implement final evaluation logic
            logger.warning("Final evaluation not yet fully implemented - placeholder")
            logger.info(f"Target improvement: >{self.config.target_improvement_pct}%")

            return True

        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            return False

    def run_full_workflow(self) -> bool:
        """
        Execute the complete Phase 1 workflow.

        Returns:
            True if all steps succeeded, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STARTING PHASE 1 WORKFLOW")
        logger.info("=" * 80)

        steps = [
            ("Train Baseline Model", self.step_1_train_baseline),
            ("Generate Predictions", self.step_2_generate_predictions),
            ("Analyze Failures", self.step_3_analyze_failures),
            ("Generate Synthetic Data", self.step_4_generate_synthetic_data),
            ("Merge Datasets", self.step_5_merge_datasets),
            ("Retrain Model", self.step_6_retrain_model),
            ("Final Evaluation", self.step_7_final_evaluation)
        ]

        for step_name, step_func in steps:
            logger.info(f"\nExecuting: {step_name}")

            if not step_func():
                logger.error(f"Step failed: {step_name}")
                logger.error("Workflow terminated")
                return False

            logger.info(f"Step completed: {step_name}")

        logger.info("=" * 80)
        logger.info("PHASE 1 WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return True


def main():
    """Main entry point for Phase 1 workflow."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Workflow: Foundational Improvements"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training steps (for testing data pipeline)"
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=50000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--synthetic-examples",
        type=int,
        default=500,
        help="Number of synthetic examples to generate"
    )

    args = parser.parse_args()

    config = WorkflowConfig(
        training_steps=args.training_steps,
        num_synthetic_examples=args.synthetic_examples
    )

    workflow = Phase1Workflow(config, skip_training=args.skip_training)
    success = workflow.run_full_workflow()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
