# Weights & Biases (W&B) Setup for HRM Training

## Current Status

**v9 Job** (Currently Running):
- ✅ Running without W&B (API key not configured)
- ✅ All CUDA and embedding fixes applied
- ✅ Optimized logging with `eval_interval: 100`
- Status: Keep running as-is

**v10 Deployment** (Ready to Deploy):
- ✅ W&B integration configured
- ✅ Deployment script ready: `deploy_v10_with_wandb.ps1`
- ⏳ Deploy when ready for W&B monitoring

---

## W&B Configuration

### API Key
```
26a08535e80a6d5d6f4a941f6a742264b25f4819
```
**Security**: Stored in deployment script, passed as container argument, set as environment variable

### Project Details
- **Project Name**: `hrm-vertex-training`
- **Dashboard**: https://wandb.ai/hrm-vertex-training
- **Run Naming**: Auto-generated with timestamp (e.g., `v10-cuda-fixes-wandb-20251005-152000`)

---

## What W&B Will Track

### Metrics Logged Every Step
1. **Loss Values**
   - Total loss
   - Task loss
   - Ponder loss (ACT)
   - Per-step losses

2. **Learning Rate**
   - Main model LR
   - Puzzle embedding LR
   - Warmup progress

3. **Training Progress**
   - Step number
   - Epoch number
   - Examples processed
   - Time per step

4. **Model Metrics**
   - Gradient norms
   - Parameter norms
   - ACT ponder cost
   - Halting probabilities

5. **Evaluation Metrics** (Every 100 epochs)
   - Test set accuracy
   - Test set loss
   - Per-puzzle performance
   - Generalization metrics

---

## Deployment Instructions

### When to Deploy v10

Deploy v10 with W&B when:
1. You want detailed metrics visualization
2. You need to compare multiple runs
3. You're ready for production monitoring
4. You want to track hyperparameter effects

### How to Deploy

```powershell
# From HRM directory
cd c:\Users\iansh\Documents\HRM

# Run deployment script
.\deploy_v10_with_wandb.ps1
```

This will:
1. Build Docker image with current code
2. Push to GCP Container Registry
3. Submit training job with W&B enabled
4. Start logging to W&B dashboard

**Note**: v9 will continue running. You can run both jobs in parallel or cancel v9 first.

---

## Cost Comparison

### v9 (Current - No W&B)
- **Logging**: Cloud Logging only
- **Visibility**: Basic logs every ~2 hours
- **Cost**: $0.73/hour (compute only)

### v10 (With W&B)
- **Logging**: Cloud Logging + W&B
- **Visibility**: Real-time metrics, charts, comparisons
- **Cost**: $0.73/hour (compute) + Free W&B tier
- **Benefits**:
  - Detailed loss curves
  - Learning rate schedules
  - Performance comparisons
  - Experiment tracking

---

## W&B Integration Details

### Code Changes Made

The existing code already supports W&B:
- `pretrain.py` has W&B initialization (lines 490-497)
- `train_vertex.py` accepts `--wandb-api-key` argument (line 188)
- No code changes needed

### Configuration Flow

```
deploy_v10_with_wandb.ps1
    ↓
Sets W&B environment variables
    ↓
Passes to train_vertex.py via args
    ↓
Sets WANDB_API_KEY environment variable
    ↓
pretrain.py initializes wandb.init()
    ↓
Metrics logged throughout training
```

### Logged Automatically

In `pretrain.py:493`, the code logs:
```python
wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
```

In `pretrain.py:513`, metrics are logged each step:
```python
if use_wandb:
    wandb.log(metrics, step=train_state.step)
```

---

## Monitoring v9 vs v10

### v9 (Current)
```bash
# View logs
gcloud logging read "resource.labels.job_id=\"7091755114737696768\"" --limit 20 --project hrmtrain

# Check for eval results (every ~2 hours)
gcloud logging read "resource.labels.job_id=\"7091755114737696768\" AND textPayload=~\"eval\"" --project hrmtrain
```

### v10 (With W&B)
```bash
# View cloud logs (same as v9)
gcloud logging read "resource.type=ml_job" --limit 20 --project hrmtrain

# PLUS W&B dashboard
# Visit: https://wandb.ai/hrm-vertex-training
# See real-time metrics, charts, and comparisons
```

---

## Troubleshooting

### If W&B Fails to Initialize

The code gracefully handles W&B failures:
```python
try:
    wandb.init(...)
except Exception as e:
    print(f"Warning: Failed to initialize wandb: {e}")
    print("Continuing training without wandb logging...")
```

Training will continue even if W&B fails.

### Common Issues

1. **API Key Invalid**
   - Check key is correct in deployment script
   - Verify key hasn't expired at wandb.ai

2. **Network Issues**
   - W&B requires internet connectivity
   - Vertex AI containers have internet access by default

3. **Project Not Found**
   - W&B will auto-create project on first run
   - No manual setup needed

---

## Next Steps

### Immediate
- ✅ v9 is running successfully (keep as-is)
- ✅ v10 deployment script ready
- ⏳ Deploy v10 when you want W&B monitoring

### Future Enhancements
- Add custom W&B charts for ACT metrics
- Track model architecture variations
- Log example predictions to W&B
- Add W&B sweeps for hyperparameter tuning

---

## Summary

**Current Setup**: v9 running without W&B
**Ready to Deploy**: v10 with W&B integration
**Command**: `.\deploy_v10_with_wandb.ps1`
**W&B Dashboard**: https://wandb.ai/hrm-vertex-training
**API Key**: Configured in deployment script

The v10 deployment is ready to use whenever you want enhanced monitoring with W&B!
