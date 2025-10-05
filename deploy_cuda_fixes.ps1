# Deploy CUDA fixes: Build Docker, push, and submit training job
# This script builds the Docker image with all CUDA initialization and indexing fixes
# Run this in your authenticated PowerShell terminal

$ErrorActionPreference = "Stop"

# Configuration
$PROJECT_ID = "hrmtrain"
$REGION = "us-central1"
$REPOSITORY = "vertex-ai-training"
$IMAGE_NAME = "hrm-training"
$VERSION = "v7-cuda-fixes"
$IMAGE_URI = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/${IMAGE_NAME}:$VERSION"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "DEPLOYING CUDA FIXES TO GOOGLE CLOUD" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Run tests
Write-Host "`n[1/5] Running tests..." -ForegroundColor Yellow
python run_tests.py --type unit --verbose 0

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some unit tests failed. Continue anyway? (y/n)" -ForegroundColor Yellow
    $response = Read-Host
    if ($response -ne "y") {
        Write-Host "Deployment cancelled." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Tests completed!" -ForegroundColor Green

# Step 2: Build Docker image
Write-Host "`n[2/5] Building Docker image..." -ForegroundColor Yellow
docker build -t $IMAGE_URI -f vertex_ai/Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "Docker image built successfully!" -ForegroundColor Green

# Step 3: Push Docker image
Write-Host "`n[3/5] Pushing Docker image to GCP..." -ForegroundColor Yellow
docker push $IMAGE_URI

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker push failed" -ForegroundColor Red
    exit 1
}

Write-Host "Docker image pushed successfully!" -ForegroundColor Green

# Step 4: Display fixes summary
Write-Host "`n[4/5] CUDA Fixes Summary:" -ForegroundColor Cyan
Write-Host "  - Explicit CUDA context initialization" -ForegroundColor White
Write-Host "  - Model-to-device transfer with synchronization" -ForegroundColor White
Write-Host "  - Sparse embedding index validation" -ForegroundColor White
Write-Host "  - Dataset puzzle index bounds checking" -ForegroundColor White
Write-Host "  - RoPE CPU initialization" -ForegroundColor White
Write-Host "  - Carry initialization with device validation" -ForegroundColor White

# Step 5: Submit training job
Write-Host "`n[5/5] Submitting training job..." -ForegroundColor Yellow

# Get user confirmation
Write-Host "`nJob configuration:" -ForegroundColor Cyan
Write-Host "  Container: $IMAGE_URI" -ForegroundColor White
Write-Host "  Machine: n1-standard-8" -ForegroundColor White
Write-Host "  GPU: NVIDIA_TESLA_T4 x1" -ForegroundColor White
Write-Host "  Arch: hrm_v1_small" -ForegroundColor White
Write-Host "  Batch size: 16" -ForegroundColor White
Write-Host "  Epochs: 10000" -ForegroundColor White
Write-Host "`nSubmit job? (y/n)" -ForegroundColor Yellow
$submit = Read-Host

if ($submit -eq "y") {
    python vertex_ai/submit_job.py `
        --project-id $PROJECT_ID `
        --region $REGION `
        --staging-bucket hrm_train_us_central1 `
        --display-name "hrm-cuda-fixes-$(Get-Date -Format 'yyyyMMdd-HHmmss')" `
        --container-uri $IMAGE_URI `
        --gcs-bucket hrm_train_us_central1 `
        --gcs-data-path training-data/arc-aug-100 `
        --gcs-checkpoint-path checkpoints/hrm_cuda_fixes `
        --machine-type n1-standard-8 `
        --accelerator-type NVIDIA_TESLA_T4 `
        --accelerator-count 1 `
        --arch-config hrm_v1_small `
        --batch-size 16 `
        --epochs 10000 `
        --learning-rate 1e-4 `
        --estimated-hours 24

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n==========================================" -ForegroundColor Green
        Write-Host "DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
        Write-Host "==========================================" -ForegroundColor Green
        Write-Host "`nTraining job submitted with CUDA fixes" -ForegroundColor Green
        Write-Host "Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID" -ForegroundColor Cyan
        Write-Host "`nExpected improvements:" -ForegroundColor Cyan
        Write-Host "  - No CUBLAS_STATUS_NOT_INITIALIZED errors" -ForegroundColor White
        Write-Host "  - No IndexError in data loading" -ForegroundColor White
        Write-Host "  - Training proceeds past first step" -ForegroundColor White
    } else {
        Write-Host "ERROR: Job submission failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`nJob submission skipped. Docker image is ready at:" -ForegroundColor Yellow
    Write-Host $IMAGE_URI -ForegroundColor White
}
