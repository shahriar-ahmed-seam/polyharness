"""FastAPI application for PolyHarness Web Studio and API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from polyharness import __version__
from polyharness.server.routes import router

app = FastAPI(
    title="PolyHarness API & Studio",
    description="Neutral Trajectory Interlingua & Multi-Harness Evaluation Suite",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Static files and Web Studio mounting
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "PolyHarness API is active. Web Studio static files loading.",
        "docs": "/docs",
        "api": "/api/health",
    }
