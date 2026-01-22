"""
LegalGPT Research Website - FastAPI Application.

Features:
- Static research content pages
- Live agent dashboard with SSE
- Embedded Gradio demo
- REST API for data
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Status file path
STATUS_FILE = RESULTS_DIR / "pipeline_status.json"


def load_status():
    """Load pipeline status from file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"pipeline": {"status": "idle"}, "agents": {}}


def load_data_stats():
    """Load data statistics."""
    stats_file = RESULTS_DIR / "data_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    return {}


def load_citation_stats():
    """Load citation graph statistics."""
    stats_file = DATA_DIR / "citations" / "stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    return {}


def load_research_content():
    """Load research context content."""
    md_file = DOCS_DIR / "RESEARCH_CONTEXT.md"
    if md_file.exists():
        return md_file.read_text()
    return "Research content not found."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("LegalGPT Research Website starting...")
    yield
    # Shutdown
    print("Website shutting down...")


# Create FastAPI app
app = FastAPI(
    title="LegalGPT Research",
    description="Graph-Augmented Legal Outcome Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))


# ============================================================================
# Page Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home/Overview page."""
    data_stats = load_data_stats()
    citation_stats = load_citation_stats()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "LegalGPT - Graph-Augmented Legal Outcome Prediction",
        "data_stats": data_stats,
        "citation_stats": citation_stats,
    })


@app.get("/methodology", response_class=HTMLResponse)
async def methodology(request: Request):
    """Methodology page."""
    return templates.TemplateResponse("methodology.html", {
        "request": request,
        "title": "Methodology - LegalGPT",
    })


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """Data page with SCDB stats and graph viz."""
    data_stats = load_data_stats()
    citation_stats = load_citation_stats()

    return templates.TemplateResponse("data.html", {
        "request": request,
        "title": "Data - LegalGPT",
        "data_stats": data_stats,
        "citation_stats": citation_stats,
    })


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Results page."""
    # Load any existing results
    metrics_file = RESULTS_DIR / "test_metrics.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "title": "Results - LegalGPT",
        "metrics": metrics,
    })


@app.get("/agents", response_class=HTMLResponse)
async def agents_dashboard(request: Request):
    """Live agent dashboard page."""
    status = load_status()

    return templates.TemplateResponse("agents.html", {
        "request": request,
        "title": "Agent Dashboard - LegalGPT",
        "status": status,
    })


@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    """Demo page with embedded Gradio."""
    return templates.TemplateResponse("demo.html", {
        "request": request,
        "title": "Demo - LegalGPT",
        "gradio_url": "http://localhost:7860",
    })


# ============================================================================
# API Routes
# ============================================================================

@app.get("/api/status")
async def api_status():
    """Get current pipeline status."""
    return load_status()


@app.get("/api/data-stats")
async def api_data_stats():
    """Get data statistics."""
    return load_data_stats()


@app.get("/api/citation-stats")
async def api_citation_stats():
    """Get citation graph statistics."""
    return load_citation_stats()


@app.get("/api/results")
async def api_results():
    """Get evaluation results."""
    metrics_file = RESULTS_DIR / "test_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return {"status": "no_results"}


# ============================================================================
# SSE (Server-Sent Events) for Real-time Updates
# ============================================================================

@app.get("/sse/status")
async def sse_status(request: Request):
    """SSE endpoint for real-time pipeline status updates."""

    async def event_generator():
        last_status = None

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            try:
                current_status = load_status()

                # Only send if status changed
                if current_status != last_status:
                    data = json.dumps(current_status)
                    yield f"event: status\ndata: {data}\n\n"
                    last_status = current_status.copy()
                else:
                    # Send keepalive
                    yield ": keepalive\n\n"

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                yield f"event: error\ndata: {str(e)}\n\n"
                await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/sse/agents/{agent_id}")
async def sse_agent(request: Request, agent_id: str):
    """SSE endpoint for specific agent updates."""

    async def event_generator():
        last_agent_status = None

        while True:
            if await request.is_disconnected():
                break

            try:
                status = load_status()
                agent_status = status.get("agents", {}).get(agent_id)

                if agent_status and agent_status != last_agent_status:
                    data = json.dumps(agent_status)
                    yield f"event: agent\ndata: {data}\n\n"
                    last_agent_status = agent_status.copy() if agent_status else None
                else:
                    yield ": keepalive\n\n"

                await asyncio.sleep(1)

            except Exception as e:
                yield f"event: error\ndata: {str(e)}\n\n"
                await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
