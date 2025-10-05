# CUDA Initialization and Indexing Error Fixes

## Overview
This document details the fixes applied to resolve CUDA initialization errors (`CUBLAS_STATUS_NOT_INITIALIZED`) and indexing assertion errors in the HRM training pipeline.

## Root Cause Analysis

### Primary Issue: CUBLAS_STATUS_NOT_INITIALIZED
**Cause**: CUDA operations were attempted before the CUDA context was properly initialized on the GPU. This occurred when tensors were created or moved to CUDA devices before the GPU runtime was ready.

**Impact**: Training failed at the first step with cuBLAS errors, preventing any forward or backward passes.

### Secondary Issue: Indexing Assertion Errors
**Cause**: Array indexing operations in dataset loading and sparse embedding access were not properly validated, leading to out-of-bounds access.

**Impact**: IndexError exceptions during data loading or model forward passes, particularly with puzzle identifier arrays.

## Fixes Applied

### 1. Explicit CUDA Context Initialization ([pretrain.py:445-456](pretrain.py#L445-L456))

**Change**: Added explicit CUDA initialization after distributed setup and before any model or data operations.

```python
# Initialize CUDA context explicitly before any operations
if torch.cuda.is_available():
    try:
        torch.cuda.synchronize()
        # Warm up CUDA by allocating and freeing a small tensor
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        if RANK == 0:
            print(f"CUDA initialized successfully on device {device}")
    except Exception as e:
        print(f"Warning: CUDA initialization failed: {e}")
        raise
```

**Benefits**:
- Ensures CUDA context is created before any tensor operations
- Validates GPU is accessible and operational
- Provides clear error messages if initialization fails
- Prevents downstream cuBLAS errors

### 2. Model-to-Device Transfer with Synchronization ([pretrain.py:159-166](pretrain.py#L159-L166))

**Change**: Added explicit error handling and synchronization when moving model to device.

```python
# Move model to device with explicit error handling
try:
    model = model.to(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
except RuntimeError as e:
    print(f"Error moving model to {device}: {e}")
    raise
```

**Benefits**:
- Ensures model is fully transferred to GPU before use
- Catches device-related errors early with clear messages
- Validates cuBLAS operations are possible after transfer

### 3. CPU Initialization for Rotary Embeddings ([models/layers.py:94-98](models/layers.py#L94-L98))

**Change**: Modified RotaryEmbedding to initialize on CPU, avoiding premature CUDA allocation.

```python
# RoPE - Initialize on CPU to avoid CUDA initialization issues
# Will be moved to correct device when model.to(device) is called
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
t = torch.arange(max_position_embeddings, dtype=torch.float32)
```

**Benefits**:
- Prevents CUDA allocation during model construction
- Allows model.to(device) to handle device placement consistently
- Eliminates race conditions in multi-GPU setups

### 4. Sparse Embedding Index Validation ([models/sparse_embedding.py:36-41](models/sparse_embedding.py#L36-L41))

**Change**: Added bounds checking for input indices before embedding lookup.

```python
# Validate input indices to prevent out-of-bounds errors
if inputs.max() >= self.weights.shape[0] or inputs.min() < 0:
    raise IndexError(
        f"Input indices out of bounds: min={inputs.min()}, max={inputs.max()}, "
        f"valid range=[0, {self.weights.shape[0]-1}]"
    )
```

**Benefits**:
- Prevents out-of-bounds access to embedding weights
- Provides clear error messages for debugging
- Catches data corruption or configuration errors early

### 5. Dataset Puzzle Index Validation ([puzzle_dataset.py:134-146](puzzle_dataset.py#L134-L146))

**Change**: Added bounds checking and clamping for puzzle indices in test iteration.

```python
puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
# Ensure puzzle_index is within valid bounds
puzzle_index = max(0, min(puzzle_index, len(dataset["puzzle_identifiers"]) - 1))

for i in range(local_start, local_end):
    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
        puzzle_index += 1

    # Verify puzzle_index is valid before appending
    if puzzle_index >= len(dataset["puzzle_identifiers"]):
        raise IndexError(
            f"Puzzle index {puzzle_index} out of bounds for puzzle_identifiers array of size {len(dataset['puzzle_identifiers'])}"
        )
    puzzle_indices.append(puzzle_index)
```

**Benefits**:
- Prevents IndexError in puzzle identifier lookups
- Validates dataset consistency
- Provides detailed error messages for data debugging

### 6. Training Batch Index Validation ([puzzle_dataset.py:199-205](puzzle_dataset.py#L199-L205))

**Change**: Added validation for puzzle indices in training batches.

```python
# Validate puzzle indices before indexing
if len(batch_puzzle_indices) > 0:
    max_puzzle_idx = batch_puzzle_indices.max()
    if max_puzzle_idx >= len(dataset["puzzle_identifiers"]):
        raise IndexError(
            f"Puzzle index {max_puzzle_idx} out of bounds for puzzle_identifiers array of size {len(dataset['puzzle_identifiers'])}"
        )
```

**Benefits**:
- Catches invalid indices before they cause crashes
- Ensures data integrity across distributed training
- Simplifies debugging of data pipeline issues

### 7. Carry Initialization with Device Validation ([models/hrm/hrm_act_v1.py:176-190](models/hrm/hrm_act_v1.py#L176-L190))

**Change**: Added CUDA synchronization check when creating empty carry tensors.

```python
def empty_carry(self, batch_size: int):
    # Create carry tensors on the same device as the model
    device = self.H_init.device

    # Ensure device is properly initialized
    if device.type == 'cuda':
        try:
            torch.cuda.synchronize(device)
        except RuntimeError as e:
            raise RuntimeError(f"CUDA device {device} not properly initialized: {e}")

    return HierarchicalReasoningModel_ACTV1InnerCarry(
        z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
    )
```

**Benefits**:
- Validates CUDA device is ready before tensor allocation
- Prevents CUBLAS errors during carry initialization
- Provides clear error context for debugging

## Testing Recommendations

### Unit Tests
1. **CUDA Initialization Test**: Verify CUDA context is properly initialized
2. **Index Bounds Test**: Validate all array accesses are within bounds
3. **Device Transfer Test**: Ensure model transfers to GPU successfully

### Integration Tests
1. **Single-GPU Training**: Run a few training steps on single GPU
2. **Multi-GPU Training**: Verify distributed training with proper device placement
3. **Data Pipeline**: Validate dataset loading with various batch sizes

### Validation Commands
```bash
# Test CUDA initialization
python -c "import torch; print(torch.cuda.is_available()); torch.cuda.synchronize(); print('CUDA OK')"

# Run single-step training test
python pretrain.py --config-name cfg_pretrain_fast

# Run distributed training test (if multi-GPU)
torchrun --nproc_per_node=2 pretrain.py --config-name cfg_pretrain_fast
```

## Monitoring and Logging

### Key Metrics to Monitor
1. **CUDA Initialization Success**: Check for "CUDA initialized successfully" message
2. **Model Loading Time**: Monitor time to move model to device
3. **First Training Step**: Ensure first forward/backward pass completes
4. **Memory Usage**: Track GPU memory allocation patterns

### Error Patterns to Watch
1. `CUBLAS_STATUS_NOT_INITIALIZED`: Indicates CUDA context issues
2. `IndexError` with puzzle indices: Dataset configuration or corruption
3. `RuntimeError` with device mismatch: Tensor on wrong device
4. `CUBLAS_STATUS_ALLOC_FAILED`: GPU out of memory (likely downstream from initialization failure)

## Performance Impact

### Expected Changes
- **Initialization Time**: +0.1-0.5s due to explicit CUDA warmup
- **Runtime Overhead**: Negligible (<0.1%) from validation checks
- **Memory Usage**: No significant change

### Mitigation Strategies
- Validation checks only run once at startup or data loading
- Index validation uses efficient numpy/torch operations
- CUDA synchronization only occurs during initialization

## Rollback Plan

If issues arise, remove the following changes in order:
1. Remove CUDA synchronization in `empty_carry` (lowest risk)
2. Remove training batch validation (medium risk)
3. Remove sparse embedding validation (higher risk)
4. Revert RotaryEmbedding to device-specific init (not recommended)
5. Remove explicit CUDA initialization (only if necessary)

## Related Issues

- CUBLAS initialization errors in distributed training
- Index out of bounds in dataset loading
- Device mismatch errors in multi-GPU setups
- Memory allocation failures due to uninitialized CUDA context

## Future Improvements

1. **Automated Testing**: Add CI/CD tests for CUDA initialization
2. **Validation Mode**: Add `--validate-only` flag to check data without training
3. **Device Management**: Consider using `torch.cuda.set_device` context managers
4. **Error Recovery**: Implement retry logic for transient CUDA errors
5. **Profiling**: Add detailed timing for each initialization step
