import os
import sys
import importlib
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

try:
    _models = importlib.import_module(".contextual-models", package="src")
    ContextualExplainRequest = _models.ContextualExplainRequest
    ContextualExplainResponse = _models.ContextualExplainResponse
    _explainer = importlib.import_module(".contextual-explainer", package="src")
    ContextualExplainerService = _explainer.ContextualExplainerService
    explainer_service = ContextualExplainerService()
except Exception:
    explainer_service = None

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
        "supported_features": ["Video Import Pipeline", "Interactive Learner Queries"] if explainer_service else ["Video Import Pipeline"]
    }


@app.post("/api/v1/contextual-explain")
@app.post("/api/v1/interactive-ask")
def contextual_explain_endpoint(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Flow 2: Answers learner custom queries based on video frame, knowledge, and user context."""
    if not explainer_service:
        raise HTTPException(status_code=501, detail="Interactive contextual explainer is not loaded.")
    try:
        req = ContextualExplainRequest.from_dict(payload)
        resp = explainer_service.explain(req)
        return resp.to_dict()
    except Exception as exc:
        logger.error(f"Error processing contextual explanation: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI Service explanation error: {str(exc)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("src.api-server:app", host=host, port=port, reload=False)
