"""Serve the AegisOps dashboard and its cutaway preview."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse


DEMO_DIR = Path(__file__).resolve().parent / "ion-mill-demo"
app = FastAPI(title="AegisOps Dashboard")


@app.get("/", response_class=FileResponse)
@app.get("/aegisops_theme.html", response_class=FileResponse, include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DEMO_DIR / "aegisops_theme.html", media_type="text/html")


@app.get("/ion-mill.html", response_class=FileResponse)
def cutaway() -> FileResponse:
    return FileResponse(DEMO_DIR / "ion-mill.html", media_type="text/html")
