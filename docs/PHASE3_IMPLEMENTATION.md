# Phase 3: Curriculum Learning Implementation

**Status**: ✅ Complete
**Date**: October 2025
**Author**: Ian Shank

---

## Overview

Phase 3 implements curriculum learning for the HRM-SQE V2 model, enabling training on puzzles of progressively increasing difficulty. This approach improves learning efficiency by starting with simpler examples and gradually introducing more complex ones.

### Key Features

- **Composite Difficulty Scoring**: Multi-factor assessment combining color diversity, grid size, and pattern complexity
- **Dynamic Threshold Scheduling**: Exponential/linear/step-based progression of difficulty thresholds
- **YAML Configuration**: Fully configurable parameters without code changes
- **Backward Compatible**: Can be disabled via flag, preserves all existing functionality
- **Wrapper Architecture**: Non-invasive design using dataset wrapper pattern

---

## Architecture

### Components

1. **PuzzleDifficultyAnalyzer**: Analyzes and scores puzzle difficulty
2. **CurriculumScheduler**: Manages difficulty threshold progression over epochs
3. **CurriculumPuzzleDataset**: Wrapper that filters puzzles based on current threshold
4. **Training Integration**: Seamless integration with existing training pipeline

### Difficulty Scoring Formula

```
difficulty = 0.4 × color_score + 0.3 × grid_size_score + 0.3 × pattern_score

where:
  color_score = num_unique_colors / max_colors
  grid_size_score = (height × width) / max_grid_size
  pattern_score = f(input_grid, output_grid)  # transformation complexity
```

All scores are normalized to [0, 1].

### Scheduling Strategies

**Exponential Pacing (Default)**:
```
threshold(epoch) = initial + (final - initial) × (base^t - 1) / (base - 1)
where t = (epoch - warmup) / (total - warmup), normalized to [0, 1]
```

Starts slow with easy puzzles, accelerates exposure to harder puzzles.

**Linear Pacing**:
```
threshold(epoch) = initial + (final - initial) × t
```

Steady, uniform increase in difficulty.

**Step Pacing**:
```
threshold(epoch) = initial + (final - initial) × floor(4t) / 4
```

Discrete jumps at 0%, 25%, 50%, 75%, 100% progress.

---

## Usage

### Basic Training with Curriculum Learning

```bash
# Enable curriculum learning with exponential schedule
python run_sqe_training.py \
  --data-dir data/processed \
  --config config/sqe_enhanced.yaml \
  --epochs 20 \
  --curriculum

# Standard training (without curriculum)
python run_sqe_training.py \
  --data-dir data/processed \
  --config config/sqe_enhanced.yaml \
  --epochs 20
```

### Configuration

Edit `config/sqe_enhanced.yaml`:

```yaml
curriculum_learning:
  enable: true

  # Difficulty weights (must sum to 1.0)
  difficulty:
    color_weight: 0.4
    grid_size_weight: 0.3
    pattern_complexity_weight: 0.3

  schedule:
    schedule_type: "exponential"  # exponential | linear | step
    initial_threshold: 0.2         # Start at 20% difficulty
    final_threshold: 1.0           # End at 100% difficulty
    warmup_epochs: 2               # Keep initial threshold for 2 epochs
    total_epochs: 20               # Total training epochs
    exponential_base: 2.0          # Growth rate (for exponential)

  initial_threshold: 0.2
```

### Programmatic Usage

```python
from dataset.curriculum_learning import (
    PuzzleDifficultyAnalyzer,
    CurriculumScheduler,
    CurriculumPuzzleDataset,
    DifficultyScoreConfig,
    CurriculumScheduleConfig
)
from torch.utils.data import DataLoader

# Analyze difficulty
analyzer = PuzzleDifficultyAnalyzer()
difficulty = analyzer.compute_difficulty(input_grid, output_grid)

# Create curriculum dataset
difficulty_config = DifficultyScoreConfig(
    color_weight=0.4,
    grid_size_weight=0.3,
    pattern_complexity_weight=0.3
)

schedule_config = CurriculumScheduleConfig(
    schedule_type="exponential",
    initial_threshold=0.2,
    final_threshold=1.0,
    total_epochs=20
)

curriculum_dataset = CurriculumPuzzleDataset(
    base_dataset=my_dataset,
    difficulty_config=difficulty_config,
    schedule_config=schedule_config,
    initial_threshold=0.2
)

# Training loop
for epoch in range(20):
    # Update threshold
    new_threshold = curriculum_dataset.step_epoch()

    # Get difficulty stats
    stats = curriculum_dataset.get_difficulty_stats()
    print(f"Epoch {epoch}: {stats['count']} puzzles, "
          f"difficulty [{stats['min']:.2f}, {stats['max']:.2f}]")

    # Create DataLoader (will use filtered puzzles)
    loader = DataLoader(curriculum_dataset, batch_size=8, shuffle=True)

    # Train...
    for batch in loader:
        # Your training code
        pass
```

---

## Implementation Details

### File Structure

```
HRM/
├── dataset/
│   └── curriculum_learning.py      # Core curriculum learning module (616 lines)
├── tests/
│   └── test_curriculum_learning.py # Comprehensive test suite (717 lines, 44 tests)
├── config/
│   └── sqe_enhanced.yaml           # Updated with curriculum config
├── run_sqe_training.py             # Updated training script with --curriculum flag
└── docs/
    └── PHASE3_IMPLEMENTATION.md    # This document
```

### Core Classes

#### `DifficultyScoreConfig`

Configuration dataclass for difficulty scoring weights.

```python
@dataclass
class DifficultyScoreConfig:
    color_weight: float = 0.4
    grid_size_weight: float = 0.3
    pattern_complexity_weight: float = 0.3
```

Validates that weights sum to 1.0.

#### `CurriculumScheduleConfig`

Configuration dataclass for curriculum scheduling.

```python
@dataclass
class CurriculumScheduleConfig:
    schedule_type: str = "exponential"
    initial_threshold: float = 0.0
    final_threshold: float = 1.0
    warmup_epochs: int = 0
    total_epochs: int = 20
    exponential_base: float = 2.0
```

Validates schedule type and threshold ranges.

#### `PuzzleDifficultyAnalyzer`

Analyzes ARC puzzle difficulty using composite scoring.

**Key Methods**:
- `analyze_grid(grid)`: Extract grid statistics (colors, size)
- `compute_pattern_complexity(input_grid, output_grid)`: Analyze transformation complexity
- `compute_difficulty(input_grid, output_grid)`: Compute overall difficulty score
- `compute_difficulty_from_example(example)`: Convenience wrapper for dict inputs

#### `CurriculumScheduler`

Manages difficulty threshold progression.

**Key Methods**:
- `get_threshold(epoch)`: Get threshold for specific epoch
- `step()`: Advance to next epoch and return new threshold
- `reset()`: Reset to initial state

Supports warmup periods, exponential/linear/step pacing.

#### `CurriculumPuzzleDataset`

Dataset wrapper that filters puzzles by difficulty threshold.

**Key Methods**:
- `set_threshold(threshold)`: Manually set difficulty threshold
- `step_epoch()`: Advance to next epoch (updates threshold via scheduler)
- `get_difficulty_stats()`: Get statistics about current difficulty distribution
- `__len__()`: Returns count of available puzzles at current threshold
- `__getitem__(idx)`: Access puzzle by index in filtered dataset

**Features**:
- Caches difficulty scores on initialization
- Dynamically filters based on current threshold
- Maintains mapping to base dataset indices
- Fully compatible with PyTorch DataLoader

---

## Testing

### Test Suite

Comprehensive test suite with **44 tests**, all passing:

```bash
pytest tests/test_curriculum_learning.py -v
```

**Test Coverage**:

1. **Configuration Tests** (9 tests)
   - Default configs
   - Custom weights
   - Validation (weights sum, threshold ranges, schedule types)

2. **Difficulty Analyzer Tests** (12 tests)
   - Grid analysis (simple, empty, complex)
   - Pattern complexity (identity, size change, color change)
   - Difficulty computation (simple, complex, from examples)
   - Bounds checking
   - Custom weights

3. **Scheduler Tests** (7 tests)
   - Linear/exponential/step schedules
   - Warmup epochs
   - Step/reset methods
   - Threshold bounds

4. **Dataset Wrapper Tests** (11 tests)
   - Initialization
   - Threshold filtering
   - Disabled curriculum
   - Item access and bounds
   - Epoch stepping
   - Validation
   - Stats
   - Tokenized format handling
   - Backward compatibility

5. **Factory Tests** (3 tests)
   - Default creation
   - Custom config
   - Disabled mode

6. **Integration Tests** (2 tests)
   - Full training simulation
   - PyTorch DataLoader compatibility

### Running Tests

```bash
# Run all curriculum tests
pytest tests/test_curriculum_learning.py -v

# Run specific test class
pytest tests/test_curriculum_learning.py::TestPuzzleDifficultyAnalyzer -v

# Run with coverage
pytest tests/test_curriculum_learning.py --cov=dataset.curriculum_learning --cov-report=html
```

---

## Training Workflow

### Standard Training (No Curriculum)

```bash
python run_sqe_training.py \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 8
```

Output:
```
📂 Loading datasets from data/processed...
  ✓ Train: 1000 examples
  ✓ Val:   200 examples
  ✓ Test:  200 examples

🔧 Creating data loaders (batch_size=8)...
🚀 Starting training for 20 epochs...
```

### Curriculum Training

```bash
python run_sqe_training.py \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 8 \
  --curriculum
```

Output:
```
📂 Loading datasets from data/processed...
  ✓ Train: 1000 examples
  ✓ Val:   200 examples
  ✓ Test:  200 examples

📚 Enabling curriculum learning...
  ✓ Curriculum dataset: 312 / 1000 examples available at current threshold
  ✓ Difficulty stats: mean=0.156, std=0.042, range=[0.050, 0.200]

🔧 Creating data loaders (batch_size=8)...
🚀 Starting training for 20 epochs...

📚 Curriculum Update (Epoch 1):
  Threshold: 0.2143
  Available puzzles: 347 / 1000
  Difficulty range: [0.050, 0.214], mean=0.168

🔄 Epoch 1:
  Batch 0/43, Loss: 3.2145
  ...
```

### Monitoring with Weights & Biases

```bash
python run_sqe_training.py \
  --data-dir data/processed \
  --epochs 20 \
  --curriculum \
  --use-wandb
```

Additional metrics logged:
- `curriculum/threshold`: Current difficulty threshold
- `curriculum/available_count`: Number of available puzzles
- `curriculum/difficulty_mean`: Mean difficulty of available puzzles
- `curriculum/difficulty_std`: Standard deviation of difficulty

---

## Design Decisions

### 1. Composite Difficulty Score

**Decision**: Use weighted combination of 3 metrics (colors, grid size, pattern complexity)

**Rationale**:
- Single metric (e.g., grid size) insufficient for ARC puzzles
- Color diversity correlates with reasoning complexity
- Pattern complexity captures transformation difficulty
- Weights (0.4, 0.3, 0.3) prioritize color reasoning (key ARC skill)

**Alternatives Considered**:
- Grid size only: Too simplistic, misses reasoning complexity
- Manual labeling: Too time-consuming, subjective
- Model-based (loss/accuracy): Requires pre-training, chicken-and-egg problem

### 2. Exponential Schedule (Default)

**Decision**: Default to exponential pacing with base=2.0

**Rationale**:
- Proven effective in curriculum learning literature
- Spends more time on easier examples early
- Gradual exposure to harder examples prevents overwhelming model
- Configurable (can use linear or step if needed)

**Evidence**:
- Bengio et al. (2009): "Curriculum Learning"
- Graves et al. (2017): "Automated Curriculum Learning for Neural Networks"

### 3. Wrapper Class Pattern

**Decision**: Use dataset wrapper instead of modifying base dataset

**Rationale**:
- Non-invasive: Preserves existing code
- Composable: Can wrap any PyTorch Dataset
- Backward compatible: Easy to disable
- Testable: Clear interface boundaries

**Alternatives Considered**:
- Modify base dataset: Breaks existing code, harder to test
- External sorting: Less flexible, requires full dataset in memory
- Sampler-based: Incompatible with shuffle, harder to track progress

### 4. YAML Configuration

**Decision**: All parameters configurable via YAML, overridable by CLI

**Rationale**:
- Reproducibility: Configs can be version-controlled
- Experimentation: Easy to test different schedules
- No code changes: Non-technical users can adjust parameters
- CLI override: Flexibility for quick experiments

---

## Performance Considerations

### Memory Usage

- **Difficulty Score Caching**: All difficulty scores computed once at initialization
- **Memory Overhead**: O(N) for N puzzles, ~8 bytes per float
- **For 10,000 puzzles**: ~80 KB additional memory (negligible)

### Computational Cost

- **Initialization**: O(N) difficulty analysis, typically <10 seconds for 10K puzzles
- **Per-Epoch Overhead**: O(N) filtering, ~1-5ms for 10K puzzles
- **DataLoader Recreation**: Minimal cost, uses filtered indices

### Training Efficiency

**Expected Benefits**:
- Faster convergence (fewer epochs to reach target accuracy)
- Better generalization (gradual exposure to complexity)
- Reduced training time (smaller effective dataset early on)

**Potential Drawbacks**:
- Slower initial epochs (smaller dataset)
- Risk of overfitting to easy examples (mitigated by warmup + exponential schedule)

---

## Troubleshooting

### Issue: "Difficulty weights must sum to 1.0"

**Cause**: Invalid weight configuration in YAML

**Fix**: Ensure weights sum to 1.0
```yaml
difficulty:
  color_weight: 0.4
  grid_size_weight: 0.3
  pattern_complexity_weight: 0.3  # 0.4 + 0.3 + 0.3 = 1.0
```

### Issue: "No puzzles available at current threshold"

**Cause**: Initial threshold too low, or all puzzles too difficult

**Fix**:
1. Check difficulty distribution: `dataset.get_difficulty_stats()`
2. Increase `initial_threshold` in config
3. Verify puzzles have reasonable difficulty (use analyzer on samples)

### Issue: Training loop crashes with "Index out of range"

**Cause**: DataLoader not updated after epoch step

**Fix**: Recreate DataLoader after calling `step_epoch()`:
```python
for epoch in range(epochs):
    dataset.step_epoch()
    loader = DataLoader(dataset, ...)  # Recreate loader
    # Train...
```

### Issue: Curriculum seems to have no effect

**Cause**: Curriculum might be disabled or threshold already at 1.0

**Check**:
1. Verify `--curriculum` flag is set
2. Check `curriculum_learning.enable: true` in YAML
3. Print difficulty stats each epoch to monitor threshold

---

## Future Enhancements

### Potential Improvements

1. **Adaptive Scheduling**
   - Adjust threshold based on validation loss
   - Slow down if loss increases
   - Speed up if loss plateaus

2. **Category-Specific Curriculum**
   - Separate thresholds for each reasoning category
   - Focus on weak categories first
   - Interleave categories strategically

3. **Difficulty Prediction Model**
   - Train model to predict difficulty from puzzle
   - Use model-predicted difficulty instead of heuristics
   - Active learning: focus on puzzles near decision boundary

4. **Multi-Stage Curriculum**
   - Define discrete stages (beginner, intermediate, advanced)
   - Different mixing ratios at each stage
   - Explicit checkpoints between stages

5. **Reverse Curriculum**
   - Start with hard examples (for robust features)
   - Then show easy examples (for fine-tuning)
   - Useful if model has good initialization

---

## References

### Curriculum Learning Literature

1. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). **Curriculum learning**. *ICML*.
   - Original curriculum learning paper
   - Showed benefits for various tasks

2. Graves, A., Bellemare, M. G., Menick, J., Munos, R., & Kavukcuoglu, K. (2017). **Automated curriculum learning for neural networks**. *ICML*.
   - Automated difficulty assessment
   - Used for RL tasks

3. Soviany, P., Ionescu, R. T., Rota, P., & Sebe, N. (2022). **Curriculum learning: A survey**. *IJCV*.
   - Comprehensive survey of curriculum learning methods
   - Taxonomy of approaches

### ARC Challenge

4. Chollet, F. (2019). **On the Measure of Intelligence**. *arXiv:1911.01547*.
   - Defines ARC benchmark
   - Discusses reasoning complexity

---

## Changelog

### Version 1.0 (October 2025)

**Features**:
- ✅ Composite difficulty scoring (colors + grid size + pattern complexity)
- ✅ Dynamic threshold scheduling (exponential/linear/step)
- ✅ YAML configuration
- ✅ Wrapper class architecture
- ✅ Full backward compatibility
- ✅ Comprehensive test suite (44 tests)
- ✅ Training script integration
- ✅ W&B logging support

**Tests**: 44/44 passing

**Files**:
- `dataset/curriculum_learning.py` (616 lines)
- `tests/test_curriculum_learning.py` (717 lines)
- `config/sqe_enhanced.yaml` (updated)
- `run_sqe_training.py` (updated)

---

## Contact

**Author**: Ian Shank
**Date**: October 2025
**Project**: HRM-SQE V2

For questions or issues, please refer to project documentation or create an issue in the repository.
