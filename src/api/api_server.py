import os
import sys
import importlib
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RakushuApiServer")

app = FastAPI(
    title="Rakushu Engine",
    description="Language processing engine powering video ingestion and learning pipeline",
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
_pipeline_router = importlib.import_module(".pipeline_router", package="src.api").router
app.include_router(_pipeline_router)

# Dictionary & Internal Sync Router
_dictionary_router = importlib.import_module(".dictionary_router", package="src.api").router
app.include_router(_dictionary_router)

# Read-only NLP output routers for completed video pipelines
_nlp_router = importlib.import_module(".nlp_router", package="src.api").router
app.include_router(_nlp_router)


@app.get("/health", include_in_schema=False)
def health_check() -> Dict[str, Any]:
    """Health check endpoint for service monitoring."""
    return {
        "status": "UP",
        "service": "rakushu-engine",
        "supported_features": ["Streaming Video Import Pipeline"]
    }


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def stream_demo_page() -> str:
    """Interactive Realtime SSE Stream Inspector UI."""
    html_file = os.path.join(os.path.dirname(__file__), "..", "demo-page.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Rakushu Engine</h1><p>Visit <a href='/docs'>/docs</a> for API.</p>"


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("src.api.api_server:app", host=host, port=port, reload=False)
