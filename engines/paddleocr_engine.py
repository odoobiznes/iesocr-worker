"""PaddleOCR 3.0 engine (CPU build).

PP-OCRv5 multilingual pipeline: det + rec + cls. Slower than Tesseract on
clean text but noticeably better on scans, forms, and multi-column layouts.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional

from PIL import Image
import numpy as np

log = logging.getLogger("iesocr.engines.paddleocr")


class PaddleOcrEngine:
    name = "paddleocr"

    def __init__(self):
        import paddleocr as _p
        self.version = getattr(_p, "__version__", "?")
        # PP-OCRv5 models, Czech/Slovak/Ukrainian covered by "latin" + "cyrillic"
        # on recognition side.  Language mix: we pick 'ch' for detection (handles
        # mixed scripts well) and fall through rec models for each language.
        from paddleocr import PaddleOCR
        lang = os.environ.get("IESOCR_PADDLE_LANG", "ch")
        self._ocr = PaddleOCR(
            use_textline_orientation=True,
            lang=lang,
        )

    def run(self, *, content: bytes, file_name: str, mime: Optional[str],
            fields_requested: List[str]) -> Dict:
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf" or (mime and "pdf" in mime.lower()):
            return self._run_pdf(content)
        return self._run_image(content)

    def _run_image(self, content: bytes) -> Dict:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        arr = np.array(img)
        return self._process_pages([arr])

    def _run_pdf(self, content: bytes) -> Dict:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(content, dpi=300)
        arrs = [np.array(p.convert("RGB")) for p in pages]
        return self._process_pages(arrs)

    def _process_pages(self, page_arrays: list) -> Dict:
        all_text: list[str] = []
        all_blocks: list[dict] = []
        confs: list[float] = []
        for page_idx, arr in enumerate(page_arrays, start=1):
            try:
                result = self._ocr.predict(arr)
            except AttributeError:
                # older API
                result = self._ocr.ocr(arr, cls=True)

            page_text: list[str] = []
            for item in _iter_paddle_items(result):
                text = item.get("text")
                if not text: continue
                page_text.append(text)
                bbox = item.get("bbox")
                all_blocks.append({
                    "page": page_idx,
                    "text": text,
                    "bbox": bbox,
                    "source": "line",
                })
                c = item.get("score")
                if isinstance(c, (int, float)): confs.append(float(c))
            all_text.append("\n".join(page_text))
        avg_conf = round(sum(confs) / len(confs), 4) if confs else None
        return {"text": "\n\n".join(all_text), "blocks": all_blocks,
                "page_count": len(page_arrays), "confidence": avg_conf,
                "engine": self.name}


def _iter_paddle_items(result):
    """Normalise across PaddleOCR 2.x vs 3.x output shapes."""
    if result is None: return
    # Paddle 3.x: list of PaddleOCRResult objects with .json
    for page in result if isinstance(result, list) else [result]:
        if hasattr(page, "json"):
            page = page.json
        if isinstance(page, dict) and "rec_texts" in page:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            polys = page.get("rec_polys") or []
            for i, t in enumerate(texts):
                poly = polys[i] if i < len(polys) else None
                bbox = _poly_to_bbox(poly) if poly is not None else None
                yield {"text": t, "bbox": bbox,
                       "score": scores[i] if i < len(scores) else None}
            continue
        # Paddle 2.x: list of [poly, (text, score)]
        if isinstance(page, list):
            for it in page:
                if not it or len(it) < 2: continue
                poly = it[0]
                tscore = it[1]
                if isinstance(tscore, (list, tuple)) and len(tscore) >= 2:
                    text, score = tscore[0], tscore[1]
                else:
                    text, score = str(tscore), None
                yield {"text": text, "bbox": _poly_to_bbox(poly), "score": score}


def _poly_to_bbox(poly):
    try:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return [int(min(xs)), int(min(ys)),
                int(max(xs) - min(xs)), int(max(ys) - min(ys))]
    except Exception:
        return None
