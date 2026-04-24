"""Tesseract 5.5 engine — CPU-fast baseline for clean scans."""
from __future__ import annotations

import io
import logging
import os
from typing import Dict, List, Optional

from PIL import Image

log = logging.getLogger("iesocr.engines.tesseract")


class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        self._pt = pytesseract
        try:
            self.version = str(pytesseract.get_tesseract_version())
        except Exception:
            self.version = "?"
        # Our install ships eng + ces + slk + ukr + rus + deu + pol + hun
        self._langs = os.environ.get("IESOCR_TESS_LANGS", "ces+slk+eng+ukr+rus+deu")

    def run(self, *, content: bytes, file_name: str, mime: Optional[str],
            fields_requested: List[str]) -> Dict:
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf" or (mime and "pdf" in mime.lower()):
            return self._run_pdf(content)
        return self._run_image(content)

    def _run_image(self, content: bytes) -> Dict:
        img = Image.open(io.BytesIO(content))
        text = self._pt.image_to_string(img, lang=self._langs)
        data = self._pt.image_to_data(img, lang=self._langs,
                                      output_type=self._pt.Output.DICT)
        blocks = _blocks_from_tesseract_data(data, page=1)
        conf = _mean_conf(data)
        return {"text": text, "blocks": blocks, "page_count": 1,
                "confidence": conf, "engine": self.name}

    def _run_pdf(self, content: bytes) -> Dict:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(content, dpi=300)
        all_text: list[str] = []
        all_blocks: list[dict] = []
        confs: list[float] = []
        for i, img in enumerate(pages, start=1):
            t = self._pt.image_to_string(img, lang=self._langs)
            d = self._pt.image_to_data(img, lang=self._langs,
                                       output_type=self._pt.Output.DICT)
            all_text.append(t)
            all_blocks.extend(_blocks_from_tesseract_data(d, page=i))
            c = _mean_conf(d)
            if c is not None: confs.append(c)
        avg_conf = sum(confs) / len(confs) if confs else None
        return {"text": "\n\n".join(all_text), "blocks": all_blocks,
                "page_count": len(pages), "confidence": avg_conf,
                "engine": self.name}


def _blocks_from_tesseract_data(data: dict, page: int) -> List[dict]:
    """Group tesseract word-level output into line-level blocks."""
    n = len(data.get("text", []))
    by_line: dict[tuple, list[int]] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt: continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        by_line.setdefault(key, []).append(i)
    blocks: List[dict] = []
    for key, idxs in by_line.items():
        words = [data["text"][i] for i in idxs]
        lefts = [data["left"][i] for i in idxs]
        tops = [data["top"][i] for i in idxs]
        widths = [data["width"][i] for i in idxs]
        heights = [data["height"][i] for i in idxs]
        x = min(lefts)
        y = min(tops)
        w = max(l + ww for l, ww in zip(lefts, widths)) - x
        h = max(t + hh for t, hh in zip(tops, heights)) - y
        blocks.append({
            "page": page,
            "text": " ".join(words),
            "bbox": [x, y, w, h],
            "source": "line",
        })
    return blocks


def _mean_conf(data: dict) -> Optional[float]:
    cs = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c >= 0]
    if not cs: return None
    # tesseract returns 0-100
    return round(sum(cs) / len(cs) / 100.0, 4)
