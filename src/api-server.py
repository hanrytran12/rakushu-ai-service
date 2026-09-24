import os
import sys
import importlib
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

logger = logging.getLogger("RakushuApiServer")

app = FastAPI(
    title="Rakushu AI Service",
    description="AI Service powering Video Ingestion & Language Learning Pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Video Processing Pipeline Router
_pipeline_router = importlib.import_module(".pipeline-router", package="src").router
app.include_router(_pipeline_router)


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint for service monitoring."""
    return {
        "status": "UP",
        "service": "rakushu-ai-service",
        "supported_features": ["Video Import Pipeline"]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("src.api-server:app", host=host, port=port, reload=False)
