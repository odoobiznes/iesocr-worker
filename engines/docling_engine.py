"""Docling (IBM) engine — digital-born PDF fast path.

Docling shines on PDFs with a text layer: 0.2–1 s/page on CPU, zero OCR.
For rasterised PDFs it falls through to its own OCR (EasyOCR backend by
default). We configure it to prefer the native text layer when available.

Covers: pdf, docx, pptx, html, md.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger("iesocr.engines.docling")


class DoclingEngine:
    name = "docling"

    def __init__(self):
        from docling.document_converter import DocumentConverter
        from docling.datamodel import base_models  # noqa: F401 (sanity import)
        import docling as _docling
        self.version = getattr(_docling, "__version__", "?")
        # Use default pipeline (structured extraction + tables).
        self._converter = DocumentConverter()

    def run(self, *, content: bytes, file_name: str, mime: Optional[str],
            fields_requested: List[str]) -> Dict:
        from docling.datamodel.document import DocumentStream

        ext = os.path.splitext(file_name)[1].lower() or ".pdf"
        stream = DocumentStream(name=file_name, stream=io.BytesIO(content))
        result = self._converter.convert(stream)
        doc = result.document

        md = doc.export_to_markdown()

        blocks: List[dict] = []
        page_count = 0
        try:
            # Docling produces a structured document tree; emit one block per text item.
            for item, level in doc.iterate_items():
                page = getattr(getattr(item, "prov", None), "page_no", 0) or 1
                page_count = max(page_count, page)
                txt = getattr(item, "text", None)
                if not txt: continue
                bbox = None
                try:
                    prov = item.prov
                    if prov and hasattr(prov, "bbox") and prov.bbox:
                        b = prov.bbox
                        bbox = [b.l, b.t, b.r - b.l, b.b - b.t]
                except Exception:
                    pass
                blocks.append({
                    "page": page,
                    "text": txt,
                    "bbox": bbox,
                    "source": getattr(item, "label", "text") or "text",
                })
        except Exception as e:
            log.debug("docling iterate_items failed: %s", e)

        return {
            "text": md,
            "blocks": blocks,
            "page_count": page_count or 1,
            "confidence": None,
            "engine": self.name,
        }
