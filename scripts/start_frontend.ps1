# AtelierAI: Start Local Vite Frontend Server
Write-Host "Starting AtelierAI React/Vite Frontend on http://localhost:5173..." -ForegroundColor Cyan
Set-Location -Path "$PSScriptRoot\..\frontend"
npm run dev
