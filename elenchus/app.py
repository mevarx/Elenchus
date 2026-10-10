"""FastAPI app — the frontend's backend.

One process serves the page and the API:

  GET  /            the single-page UI
  POST /api/audit   upload a bill -> extraction + adjudication
  GET  /api/demo    replay all cached fixtures (instant, offline)

Uploads are copied into fixtures/ so the extraction cache persists and a
re-upload of the same bill is instant — the live model call is a bonus,
never a dependency.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .adjudicator import adjudicate
from .extract import extract_with_cache
from .schemas import Extraction

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
FIXTURES = ROOT.parent / "fixtures"
FIXTURES.mkdir(exist_ok=True)

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"}

app = FastAPI(title="Elenchus", version="0.1.0")


def _verdict_payload(ext: Extraction, label: str | None = None) -> dict:
    v = adjudicate(ext)
    return {
        "label": label,
        "extraction": ext.to_dict(),
        "verdict": {
            "reconciled": v.reconciled,
            "summary": v.summary(),
            "stated_total": v.stated_total,
            "naive_total": v.naive_total,
            "naive_delta": v.naive_delta,
            "computed_total": v.computed_total,
            "delta": v.delta,
            "findings": [
                {"code": f.code, "severity": f.severity,
                 "detail": f.detail, "numbers": f.numbers}
                for f in v.findings
            ],
        },
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/audit")
async def audit(file: UploadFile):
    suffix = Path(file.filename or "bill.png").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return JSONResponse(status_code=400,
                            content={"error": f"unsupported file type: {suffix}"})
    # copy into fixtures so the extraction cache persists beside the bill
    dest = FIXTURES / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        ext = extract_with_cache(str(dest))
    except Exception as e:  # model down / timeout — still adjudicate what we have
        cached = dest.with_suffix("").with_suffix(".extraction.json")
        if not cached.exists():
            return JSONResponse(status_code=502,
                                content={"error": f"extraction failed: {e}"})
    payload = _verdict_payload(ext, label=dest.name)
    return payload


@app.get("/api/demo")
def demo():
    """Instant replay of every cached fixture — the offline demo path."""
    results = []
    for p in sorted(FIXTURES.glob("*.extraction.json")):
        try:
            ext = Extraction.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
        results.append(_verdict_payload(ext, label=p.stem.replace(".extraction", "")))
    return {"results": results}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
