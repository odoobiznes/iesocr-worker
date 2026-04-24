"""EasyOCR engine — alternative for hard-to-read scans / handwriting.

Slower than Tesseract on clean text but more robust on noisy input.
Multilang: Czech / Slovak / Ukrainian / Russian / English.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional

from PIL import Image
import numpy as np

log = logging.getLogger("iesocr.engines.easyocr")


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self):
        import easyocr as _e
        self.version = getattr(_e, "__version__", "?")
        # EasyOCR pairing rule: only one Asian + many Latin-script, or multi Latin/Cyrillic.
        langs = os.environ.get("IESOCR_EASY_LANGS", "cs,sk,uk,ru,en,de").split(",")
        langs = [x.strip() for x in langs if x.strip()]
        self._reader = _e.Reader(langs, gpu=False, verbose=False)

    def run(self, *, content: bytes, file_name: str, mime: Optional[str],
            fields_requested: List[str]) -> Dict:
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf" or (mime and "pdf" in mime.lower()):
            return self._run_pdf(content)
        return self._run_image(content)

    def _run_image(self, content: bytes) -> Dict:
        arr = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
        return self._process([arr])

    def _run_pdf(self, content: bytes) -> Dict:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(content, dpi=300)
        arrs = [np.array(p.convert("RGB")) for p in pages]
        return self._process(arrs)

    def _process(self, pages: list) -> Dict:
        text_parts: list[str] = []
        all_blocks: list[dict] = []
        confs: list[float] = []
        for page_idx, arr in enumerate(pages, start=1):
            rows = self._reader.readtext(arr, paragraph=False)
            page_text: list[str] = []
            for poly, text, score in rows:
                if not text: continue
                page_text.append(text)
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                bbox = [int(min(xs)), int(min(ys)),
                        int(max(xs) - min(xs)), int(max(ys) - min(ys))]
                all_blocks.append({
                    "page": page_idx, "text": text, "bbox": bbox,
                    "source": "line",
                })
                confs.append(float(score))
            text_parts.append("\n".join(page_text))
        avg_conf = round(sum(confs) / len(confs), 4) if confs else None
        return {"text": "\n\n".join(text_parts), "blocks": all_blocks,
                "page_count": len(pages), "confidence": avg_conf,
                "engine": self.name}
