# AtelierAI: Complete Model Training Pipeline
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         AtelierAI: Complete Local Training Pipeline      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Train Auxiliary Fashion-MNIST CNN Classifier
Write-Host "`n[1/4] Training Auxiliary Fashion-MNIST CNN Classifier..." -ForegroundColor Yellow
python -m ml.training.train_classifier --epochs 8
if ($LASTEXITCODE -ne 0) { Write-Error "Classifier training failed"; exit 1 }

# 2. Train Conditional GAN (cGAN)
Write-Host "`n[2/4] Training Conditional GAN (cGAN)..." -ForegroundColor Yellow
python -m ml.training.train_cgan --epochs 25 --batch-size 64
if ($LASTEXITCODE -ne 0) { Write-Error "cGAN training failed"; exit 1 }

# 3. Train Conditional VAE (C-VAE)
Write-Host "`n[3/4] Training Conditional VAE (C-VAE)..." -ForegroundColor Yellow
python -m ml.training.train_vae --epochs 25 --batch-size 64
if ($LASTEXITCODE -ne 0) { Write-Error "VAE training failed"; exit 1 }

# 4. Fine-Tune DistilBERT on Natural Language Clothing Prompts
Write-Host "`n[4/4] Fine-tuning DistilBERT Clothing Classifier..." -ForegroundColor Yellow
python -m ml.training.train_distilbert --epochs 4 --batch-size 16
if ($LASTEXITCODE -ne 0) { Write-Error "DistilBERT fine-tuning failed"; exit 1 }

# 5. Run Complete Benchmark Suite
Write-Host "`n[+] Executing Complete Benchmark Suite..." -ForegroundColor Green
python -m scripts.run_benchmarks

Write-Host "`nAll models trained and verified successfully!" -ForegroundColor Green
