"""
Vercel Serverless Entry Point for LegalGPT Website.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import json

# Create app
app = FastAPI(title="LegalGPT Research")

# Paths - for Vercel, static files are served from /public
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR.parent / "data"
RESULTS_DIR = BASE_DIR.parent / "results"

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def load_json_safe(path: Path, default=None):
    """Safely load JSON file."""
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except:
        pass
    return default


def load_status():
    return load_json_safe(RESULTS_DIR / "pipeline_status.json", {"pipeline": {"status": "idle"}, "agents": {}})


def load_data_stats():
    return load_json_safe(RESULTS_DIR / "data_stats.json")


def load_citation_stats():
    return load_json_safe(DATA_DIR / "citations" / "stats.json")


# ============================================================================
# Page Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "LegalGPT - Graph-Augmented Legal Outcome Prediction",
        "data_stats": load_data_stats(),
        "citation_stats": load_citation_stats(),
    })


@app.get("/methodology", response_class=HTMLResponse)
async def methodology(request: Request):
    return templates.TemplateResponse("methodology.html", {
        "request": request,
        "title": "Methodology - LegalGPT",
    })


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    return templates.TemplateResponse("data.html", {
        "request": request,
        "title": "Data - LegalGPT",
        "data_stats": load_data_stats(),
        "citation_stats": load_citation_stats(),
    })


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    metrics = load_json_safe(RESULTS_DIR / "test_metrics.json")
    return templates.TemplateResponse("results.html", {
        "request": request,
        "title": "Results - LegalGPT",
        "metrics": metrics,
    })


@app.get("/agents", response_class=HTMLResponse)
async def agents_dashboard(request: Request):
    return templates.TemplateResponse("agents.html", {
        "request": request,
        "title": "Agent Dashboard - LegalGPT",
        "status": load_status(),
    })


@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    return templates.TemplateResponse("demo.html", {
        "request": request,
        "title": "Demo - LegalGPT",
        "gradio_url": "https://your-gradio-space.hf.space",  # Update with HuggingFace Space URL
    })


# ============================================================================
# API Routes (for polling instead of SSE on Vercel)
# ============================================================================

@app.get("/api/status")
async def api_status():
    return JSONResponse(load_status())


@app.get("/api/data-stats")
async def api_data_stats():
    return JSONResponse(load_data_stats())


@app.get("/api/citation-stats")
async def api_citation_stats():
    return JSONResponse(load_citation_stats())


@app.get("/api/results")
async def api_results():
    return JSONResponse(load_json_safe(RESULTS_DIR / "test_metrics.json", {"status": "no_results"}))


# Handler for Vercel
handler = app
