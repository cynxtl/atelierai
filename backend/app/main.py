"""Main FastAPI application entry point for AtelierAI."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.endpoints import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("atelier-ai")

app = FastAPI(
    title="AtelierAI API",
    description="Text-to-Conditional-GAN Synthesis Studio API for Fashion-MNIST",
    version="1.0.0",
)

# CORS Middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",  # Allow local Vite dev preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to AtelierAI API. Open /docs for Swagger API specification.",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
