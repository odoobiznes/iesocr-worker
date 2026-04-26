"""IES-OCR worker — FastAPI HTTP service.

Contract (matches C# PaddleOcrClient in PohodaDigi):
  POST /extract
    { file_base64, file_name, mime, template?, fields_requested? }
    → { ok, fields, layout, elapsed_ms, engine, confidence, error? }

Engines are dispatched by file type + explicit override.  All heavy work
(docling PDF parsing, tesseract/easyocr/paddleocr rasters) runs in-process.

The service is stateless per request; workers can be scaled via
uvicorn --workers.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

import engines as registry
from template_runner import apply_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("iesocr")

app = FastAPI(title="IES-OCR worker", version="0.1.1", default_response_class=ORJSONResponse)

# ============================================================ models


class ExtractRequest(BaseModel):
    file_base64: str = Field(..., min_length=4, description="document bytes as base64")
    file_name: str
    mime: Optional[str] = None
    template: Optional[Dict[str, Any]] = None
    fields_requested: Optional[List[str]] = None
    engine_override: Optional[str] = Field(
        None, description="Force a specific engine: docling|tesseract|paddleocr|easyocr"
    )


class ExtractResponse(BaseModel):
    ok: bool
    fields: Dict[str, Any] = {}
    layout: Optional[Dict[str, Any]] = None
    elapsed_ms: int = 0
    engine: str = ""
    confidence: Optional[float] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = {}


# ============================================================ endpoints


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "iesocr-worker",
        "version": "0.1",
        "engines": registry.available_engines(),
    }


@app.get("/engines")
async def engines():
    return {"engines": registry.engine_details()}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    t0 = time.perf_counter()
    try:
        blob = base64.b64decode(req.file_base64)
    except Exception as e:
        return ExtractResponse(ok=False, engine="-", error=f"bad base64: {e}")
    if not blob:
        return ExtractResponse(ok=False, engine="-", error="empty body")

    engine_name = req.engine_override or _pick_engine(req.mime, req.file_name)
    engine = registry.get(engine_name)
    if engine is None:
        return ExtractResponse(ok=False, engine=engine_name or "?",
                               error=f"engine not available: {engine_name}")

    try:
        result = engine.run(
            content=blob, file_name=req.file_name, mime=req.mime,
            fields_requested=req.fields_requested or [],
        )
    except Exception as e:
        log.exception("engine %s crashed", engine_name)
        return ExtractResponse(ok=False, engine=engine_name, error=f"engine crash: {e}",
                               elapsed_ms=int((time.perf_counter() - t0) * 1000))

    fields: Dict[str, Any] = {}
    if req.template:
        try:
            fields = apply_template(req.template, result)
        except Exception as e:
            log.exception("template apply failed")
            return ExtractResponse(ok=False, engine=engine_name,
                                   error=f"template apply: {e}",
                                   elapsed_ms=int((time.perf_counter() - t0) * 1000),
                                   meta={"raw_text_len": len(result.get("text", ""))})

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return ExtractResponse(
        ok=True,
        fields=fields,
        layout={"blocks": result.get("blocks", [])} if result.get("blocks") else None,
        elapsed_ms=elapsed_ms,
        engine=engine_name,
        confidence=result.get("confidence"),
        meta={
            "raw_text_len": len(result.get("text", "")),
            "page_count": result.get("page_count", 1),
        },
    )


# ============================================================ helpers


def _pick_engine(mime: Optional[str], file_name: str) -> str:
    """Pick the best-fit engine for the input."""
    mt = (mime or "").lower()
    ext = os.path.splitext(file_name)[1].lower()

    # Office formats → docling handles them natively (incl. xlsx as markdown tables)
    if ext in (".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
               ".html", ".htm", ".md"):
        return "docling"
    if "pdf" in mt or "spreadsheet" in mt or "wordprocessing" in mt or \
       "presentation" in mt or "html" in mt:
        return "docling"

    # Images: tesseract is fastest baseline; paddleocr better on scans
    # Default to tesseract; callers override via engine_override for paddle/easy.
    return "tesseract"


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("unhandled")
    return ORJSONResponse(
        status_code=500,
        content={"ok": False, "engine": "-", "error": f"unhandled: {exc}"},
    )
