# Contributing to AtelierAI

Thank you for your interest in contributing to AtelierAI!

## Development Guidelines

1. **Local-First Verification**: Always ensure models, tests, and builds run locally and cleanly before submitting changes.
2. **Architecture Decoupling**: Keep machine learning business logic separate from API routing handlers. Services in `backend/app/services` or `ml/` must be importable without starting FastAPI.
3. **No Binary Weights in Git**: Never commit `.pt`, `.pth`, `.safetensors`, or `.bin` weight files to git. Use scripts to train or download artifacts.
4. **Code Quality**:
   - Python: Format with `ruff format` and lint with `ruff check`. Run unit tests with `pytest`.
   - Frontend: Format with prettier, lint with eslint, and check types with `npm run build` or `tsc --noEmit`.
5. **Responsible AI**: When modifying prompt templates or parsers, run the hallucination and prompt-injection benchmark suites.

## Getting Started

1. Set up your Python environment with PyTorch and dependencies (`pip install -r backend/requirements.txt`).
2. Train or verify models using the provided CLI scripts in `ml/training/`.
3. Start the backend: `python -m uvicorn backend.app.main:app --reload --port 8000`.
4. Start the frontend: `cd frontend && npm install && npm run dev`.
