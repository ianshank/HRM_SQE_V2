# Deploy v10 with W&B Logging Enabled
# This deployment includes all CUDA/embedding fixes plus W&B monitoring

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "HRM Training - v10 Deployment with W&B Logging" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PROJECT_ID = "hrmtrain"
$REGION = "us-central1"
$BUCKET = "hrm_train_us_central1"
$IMAGE_NAME = "hrm-training:v10-wandb-enabled"
$IMAGE_URI = "us-central1-docker.pkg.dev/$PROJECT_ID/vertex-ai-training/$IMAGE_NAME"

# W&B Configuration
$WANDB_API_KEY = "26a08535e80a6d5d6f4a941f6a742264b25f4819"
$WANDB_PROJECT = "hrm-vertex-training"
$WANDB_RUN_NAME = "v10-cuda-fixes-wandb-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

# Training Configuration
$ARCH_CONFIG = "hrm_v1_small"
$BATCH_SIZE = 16
$EPOCHS = 100000
$EVAL_INTERVAL = 100  # Evaluate every 100 epochs (~2 hours)
$LEARNING_RATE = 0.0001

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$JOB_NAME = "hrm-v10-wandb-$TIMESTAMP"

Write-Host "[1/4] Building Docker image..." -ForegroundColor Yellow
Write-Host "Image: $IMAGE_URI" -ForegroundColor Gray
Write-Host ""

docker build -t $IMAGE_URI -f vertex_ai/Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Docker image built successfully" -ForegroundColor Green
Write-Host ""

Write-Host "[2/4] Pushing Docker image to GCP..." -ForegroundColor Yellow
docker push $IMAGE_URI

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Docker image pushed successfully" -ForegroundColor Green
Write-Host ""

Write-Host "[3/4] Submitting training job to Vertex AI..." -ForegroundColor Yellow
Write-Host "Job Name: $JOB_NAME" -ForegroundColor Gray
Write-Host "W&B Project: $WANDB_PROJECT" -ForegroundColor Gray
Write-Host "W&B Run: $WANDB_RUN_NAME" -ForegroundColor Gray
Write-Host ""

# Submit the job with W&B credentials
python -c @"
from google.cloud import aiplatform
from datetime import datetime

aiplatform.init(
    project='$PROJECT_ID',
    location='$REGION',
    staging_bucket='gs://$BUCKET'
)

job = aiplatform.CustomContainerTrainingJob(
    display_name='$JOB_NAME',
    container_uri='$IMAGE_URI',
)

job.run(
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    boot_disk_type='pd-ssd',
    boot_disk_size_gb=100,
    args=[
        '--project-id=$PROJECT_ID',
        '--gcs-bucket=$BUCKET',
        '--gcs-data-path=training-data/arc-aug-100',
        '--gcs-checkpoint-path=checkpoints/hrm_v10_wandb',
        '--arch-config=$ARCH_CONFIG',
        '--batch-size=$BATCH_SIZE',
        '--epochs=$EPOCHS',
        '--eval-interval=$EVAL_INTERVAL',
        '--learning-rate=$LEARNING_RATE',
        '--wandb-project=$WANDB_PROJECT',
        '--wandb-run-name=$WANDB_RUN_NAME',
        '--wandb-api-key=$WANDB_API_KEY',
    ],
    base_output_dir='gs://$BUCKET/vertex_outputs',
    sync=False,
)

print(f'Training job submitted successfully!')
print(f'Job name: $JOB_NAME')
print(f'View at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID')
print(f'W&B Dashboard: https://wandb.ai/$WANDB_PROJECT/runs')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Job submission failed" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Training job submitted successfully" -ForegroundColor Green
Write-Host ""

Write-Host "[4/4] Deployment Summary" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Deployment: v10-wandb-enabled" -ForegroundColor White
Write-Host "Job Name: $JOB_NAME" -ForegroundColor White
Write-Host "Docker Image: $IMAGE_URI" -ForegroundColor White
Write-Host ""
Write-Host "Configuration:" -ForegroundColor White
Write-Host "  - Architecture: $ARCH_CONFIG" -ForegroundColor Gray
Write-Host "  - Batch Size: $BATCH_SIZE" -ForegroundColor Gray
Write-Host "  - Eval Interval: $EVAL_INTERVAL epochs (~2 hours)" -ForegroundColor Gray
Write-Host "  - Learning Rate: $LEARNING_RATE" -ForegroundColor Gray
Write-Host ""
Write-Host "W&B Monitoring:" -ForegroundColor White
Write-Host "  - Project: $WANDB_PROJECT" -ForegroundColor Gray
Write-Host "  - Run: $WANDB_RUN_NAME" -ForegroundColor Gray
Write-Host "  - Dashboard: https://wandb.ai/$WANDB_PROJECT" -ForegroundColor Cyan
Write-Host ""
Write-Host "Fixes Included:" -ForegroundColor White
Write-Host "  [x] CUDA context initialization" -ForegroundColor Green
Write-Host "  [x] Embedding index validation" -ForegroundColor Green
Write-Host "  [x] Dataset bounds checking" -ForegroundColor Green
Write-Host "  [x] Optimized logging (eval_interval=100)" -ForegroundColor Green
Write-Host "  [x] W&B metrics tracking" -ForegroundColor Green
Write-Host ""
Write-Host "Monitoring Commands:" -ForegroundColor White
Write-Host "  View logs:" -ForegroundColor Gray
Write-Host "    gcloud logging read 'resource.type=ml_job' --limit 20 --project $PROJECT_ID" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  List jobs:" -ForegroundColor Gray
Write-Host "    gcloud ai custom-jobs list --region $REGION --project $PROJECT_ID" -ForegroundColor DarkGray
Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Deployment Complete! Current v9 job will continue running." -ForegroundColor Green
Write-Host "Run this script when ready to deploy v10 with W&B logging." -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
