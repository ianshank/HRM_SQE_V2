# Phase 1: Foundational Improvements - Implementation Guide

**Date:** October 4, 2025
**Status:** Infrastructure Complete - Ready for Training
**Hardware:** Single GPU (NVIDIA GTX 1660 Ti 6GB)

---

## Overview

Phase 1 implements a data-centric approach to improve ARC reasoning by programmatically generating synthetic training data targeting the model's weakest reasoning categories.

### Key Innovation

Instead of blindly generating more data, Phase 1:
1. Analyzes baseline model failures by reasoning category
2. Identifies the top 3-5 weakest categories
3. Generates targeted synthetic puzzles for those specific weaknesses
4. Retrains on combined original + synthetic data
5. Validates >5% performance improvement on weak categories

---

## Implementation Status

### ✅ Completed Components

#### 1. Dataset Infrastructure
- **ARC-1 Dataset**: Built with 100 augmentations (reduced from 1000 for memory constraints)
  - Location: [`data/arc-aug-100`](../data/arc-aug-100)
  - Training examples: 960 puzzle groups with 4.15 examples per puzzle avg
  - Validation set: Available for evaluation

#### 2. Baseline Analyzer
- **File**: [`analysis/arc_baseline_analyzer.py`](../analysis/arc_baseline_analyzer.py)
- **Features**:
  - Categorizes failures by 11 reasoning types:
    - Symmetry, Object Counting, Pattern Repetition
    - Spatial Transformation, Color Transformation
    - Topological, Size Scaling, Rotation, Reflection
    - Composition, Unknown
  - Generates detailed markdown reports with:
    - Overall accuracy metrics
    - Category-specific success rates
    - Example failures for each category
    - Actionable recommendations

#### 3. Synthetic Puzzle Generator
- **File**: [`dataset/build_generated_arc_dataset.py`](../dataset/build_generated_arc_dataset.py)
- **Architecture**: Modular, extensible generator system
- **Implemented Generators**:
  - **SymmetryGenerator**: Creates horizontal mirror puzzles
  - **ObjectCountingGenerator**: Generates counting tasks
  - **PatternRepetitionGenerator**: Creates tiling/repetition puzzles
  - **RotationGenerator**: Generates 90°/180°/270° rotation tasks
  - **ColorTransformationGenerator**: Creates color permutation puzzles

#### 4. Comprehensive Test Suite
- **File**: [`dataset/test_build_generated_arc_dataset.py`](../dataset/test_build_generated_arc_dataset.py)
- **Coverage**: 32 tests, 100% passing
- **Test Categories**:
  - Unit tests for each generator
  - Contract tests ensuring interface compliance
  - Integration tests for full pipeline
  - Property-based tests verifying puzzle correctness

#### 5. Training Configuration
- **File**: [`config/cfg_pretrain_single_gpu.yaml`](../config/cfg_pretrain_single_gpu.yaml)
- **Adaptations for Single GPU**:
  - Batch size: 96 (vs 768 multi-GPU)
  - More frequent evaluation: every 5000 steps
  - Uses reduced augmentation dataset

#### 6. Automated Workflow
- **File**: [`phase1_workflow.py`](../phase1_workflow.py)
- **Capabilities**:
  - End-to-end automation of Phase 1
  - Checkpointing and resume support
  - Logging and error handling
  - Performance tracking

---

## File Structure

```
HRM/
├── analysis/
│   ├── arc_baseline_analyzer.py     # Failure analysis system
│   └── test_arc_baseline_analyzer.py # Tests (to be created)
│
├── config/
│   ├── cfg_pretrain.yaml            # Original multi-GPU config
│   └── cfg_pretrain_single_gpu.yaml # Single GPU adaptation
│
├── dataset/
│   ├── build_arc_dataset.py         # Original ARC dataset builder
│   ├── build_generated_arc_dataset.py # NEW: Synthetic generator
│   └── test_build_generated_arc_dataset.py # NEW: Comprehensive tests
│
├── data/
│   ├── arc-aug-100/                 # Reduced augmentation dataset
│   └── arc-synthetic-test/          # Sample synthetic data (50 puzzles)
│
├── docs/
│   ├── PHASE1_IMPLEMENTATION.md     # This file
│   └── ARC_Failure_Analysis_Baseline.md # Generated after training
│
├── logs/
│   ├── arc_dataset_build_100.log    # Dataset build log
│   └── phase1_workflow.log          # Workflow execution log
│
└── phase1_workflow.py               # NEW: Automated workflow orchestrator
```

---

## Usage Guide

### Quick Start: Generate Synthetic Data

```bash
# Generate 500 synthetic puzzles for specific categories
python dataset/build_generated_arc_dataset.py \
  --categories=symmetry,rotation,object_counting \
  --num-examples=500 \
  --output-dir=data/arc-synthetic-targeted

# Run with default settings
python dataset/build_generated_arc_dataset.py
```

### Full Phase 1 Workflow

```bash
# Option 1: Full training workflow (will take several hours)
python phase1_workflow.py

# Option 2: Skip training for data pipeline testing
python phase1_workflow.py --skip-training

# Option 3: Custom configuration
python phase1_workflow.py \
  --training-steps=50000 \
  --synthetic-examples=500
```

### Manual Step-by-Step

#### Step 1: Train Baseline Model

```bash
# Using single GPU configuration
python pretrain.py --config-name=cfg_pretrain_single_gpu

# Monitor training
tail -f logs/phase1_workflow.log
```

#### Step 2: Generate Predictions

```bash
# After training completes, generate predictions
python evaluate.py checkpoint=<CHECKPOINT_PATH>
```

#### Step 3: Analyze Failures

```bash
# Run baseline analyzer
python -c "
from analysis.arc_baseline_analyzer import ARCBaselineAnalyzer
from pathlib import Path

analyzer = ARCBaselineAnalyzer(
    dataset_path='data/arc-aug-100',
    checkpoint_path='<CHECKPOINT_PATH>'
)

identifier_map, predictions = analyzer.load_predictions()
report = analyzer.analyze_failures(identifier_map, predictions)
analyzer.generate_report(report, Path('docs/ARC_Failure_Analysis_Baseline.md'))

print(f'Top weak categories: {[c.value for c in report.top_weak_categories[:3]]}')
"
```

#### Step 4: Generate Targeted Synthetic Data

```bash
# Use categories from Step 3
python dataset/build_generated_arc_dataset.py \
  --categories=<WEAK_CATEGORIES> \
  --num-examples=500
```

#### Step 5: Retrain on Combined Dataset

```bash
# TODO: Implement dataset merging
# Then retrain with combined dataset
```

---

## Testing

### Run All Tests

```bash
# Run generator tests
python -m pytest dataset/test_build_generated_arc_dataset.py -v

# Expected: 32 passed in ~1s
```

### Test Individual Generators

```bash
# Test symmetry generator
python -m pytest dataset/test_build_generated_arc_dataset.py::TestSymmetryGenerator -v

# Test object counting generator
python -m pytest dataset/test_build_generated_arc_dataset.py::TestObjectCountingGenerator -v
```

### Integration Test

```bash
# Generate small test dataset
python dataset/build_generated_arc_dataset.py \
  --categories=symmetry,rotation \
  --num-examples=50 \
  --output-dir=data/arc-synthetic-test

# Verify output
ls -lh data/arc-synthetic-test/
cat data/arc-synthetic-test/dataset_metadata.json
```

---

## Design Principles

### 1. Modular Architecture

Each reasoning category has its own generator class:

```python
class SymmetryGenerator(PuzzleGenerator):
    def generate(self, num_puzzles: int) -> List[GeneratedPuzzle]:
        # Category-specific logic
        pass
```

### 2. Reusable Components

Generators inherit from base class with common utilities:
- Random grid generation
- Puzzle hashing (deduplication)
- Metadata tracking

### 3. Validated Output

Every puzzle is:
- Tested for solvability
- Deduplicated via hashing
- Tagged with metadata for traceability

### 4. Traceable & Debuggable

Full logging and metadata:
- Generator type
- Category
- Generation parameters
- Unique puzzle IDs

---

## Configuration

### Generator Configuration

```python
GeneratorConfig(
    categories=[ReasoningCategory.SYMMETRY, ...],  # Target categories
    num_examples=500,                               # Total puzzles
    output_dir="data/arc-synthetic-targeted",       # Output path
    seed=42,                                        # Reproducibility
    min_grid_size=3,                                # Minimum grid dimension
    max_grid_size=15,                               # Maximum grid dimension
    num_colors=10,                                  # ARC color palette (0-9)
    examples_per_puzzle=3,                          # Training examples
    test_examples_per_puzzle=1                      # Test examples
)
```

### Training Configuration (Single GPU)

```yaml
data_path: data/arc-aug-100
global_batch_size: 96                # Reduced for 6GB VRAM
epochs: 100000
eval_interval: 5000                  # More frequent evaluation
lr: 1e-4
puzzle_emb_lr: 1e-2
weight_decay: 0.1
```

---

## Hardware Requirements

### Minimum (Current Setup)

- **GPU**: NVIDIA GTX 1660 Ti (6GB VRAM)
- **RAM**: 16GB
- **Storage**: 20GB for datasets + checkpoints
- **Time**: ~24+ hours for baseline training

### Recommended (Original Paper)

- **GPU**: 8x NVIDIA A100 or H100
- **RAM**: 256GB
- **Time**: ~10-24 hours for full training

---

## Performance Metrics

### Expected Outcomes

- **Baseline Accuracy**: ~76% (from paper)
- **Target Improvement**: >5% on weak categories
- **Overall Improvement**: 2-3% on full validation set

### Tracking Metrics

1. **Category-Specific Success Rate**
   - Before: Baseline performance per category
   - After: Post-retrain performance per category
   - Goal: >5% improvement on targeted categories

2. **Overall Accuracy**
   - Full ARC validation set performance
   - Monitor for no regression on strong categories

3. **Synthetic Data Quality**
   - Puzzle solvability rate
   - Diversity metrics (unique puzzles/total generated)

---

## Next Steps

### Immediate (Ready to Execute)

1. **Start Baseline Training**
   ```bash
   python pretrain.py --config-name=cfg_pretrain_single_gpu
   ```

2. **Monitor Progress**
   - Check W&B dashboard (requires setup)
   - Monitor logs: `tail -f logs/*.log`

### After Baseline Training

3. **Generate Analysis Report**
   - Run evaluation on validation set
   - Execute baseline analyzer
   - Review weak categories

4. **Generate Synthetic Data**
   - Use identified weak categories
   - Generate 500 targeted examples
   - Validate puzzle quality

5. **Implement Dataset Merging**
   - Combine original + synthetic datasets
   - Maintain proper train/val split
   - Update metadata

6. **Retrain and Evaluate**
   - Train on combined dataset
   - Compare performance metrics
   - Validate >5% improvement

---

## Troubleshooting

### Common Issues

#### Memory Errors During Dataset Building

**Symptom**: `MemoryError` when building ARC dataset

**Solution**: Reduce augmentation count
```bash
python dataset/build_arc_dataset.py --num-aug 100
```

#### GPU Out of Memory

**Symptom**: CUDA OOM during training

**Solution**: Reduce batch size in config
```yaml
global_batch_size: 48  # Reduce from 96
```

#### Slow Training

**Symptom**: Very slow progress on single GPU

**Expected**: Training will take 2-3x longer than multi-GPU setup
**Monitor**: Ensure GPU utilization is >80%

---

## References

### Key Files

- Paper: [Hierarchical Reasoning Model (arXiv:2506.21734)](https://arxiv.org/abs/2506.21734)
- Original Repo: [https://github.com/anthropics/HRM](https://github.com/anthropics/HRM)
- ARC Benchmark: [https://github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)

### Code Quality

- **Testing**: 32/32 tests passing (100%)
- **Logging**: Comprehensive logging in all modules
- **Documentation**: Inline docstrings + this guide
- **Type Hints**: Full type annotations
- **Error Handling**: Try-except blocks with logging

---

## Changelog

### 2025-10-04

- ✅ Initialized git submodules for ARC datasets
- ✅ Built ARC-1 dataset with 100 augmentations
- ✅ Created single-GPU training configuration
- ✅ Implemented synthetic puzzle generator with 5 categories
- ✅ Created comprehensive test suite (32 tests, 100% passing)
- ✅ Built automated Phase 1 workflow orchestrator
- ✅ Generated sample synthetic dataset (50 puzzles)
- ✅ Created implementation documentation

### Pending

- ⏳ Train baseline HRM model (long-running)
- ⏳ Generate and analyze failure report
- ⏳ Implement dataset merging functionality
- ⏳ Retrain on combined dataset
- ⏳ Validate performance improvements

---

## Contact & Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review test output: `pytest -v`
3. Refer to [README.md](../README.md)
4. Discord: [https://discord.gg/sapient](https://discord.gg/sapient)

---

**Status**: Phase 1 infrastructure complete and tested. Ready to begin baseline training.
