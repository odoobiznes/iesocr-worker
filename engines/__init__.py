"""OCR engine registry.

Each engine implements a minimal contract:

    class Engine:
        name: str
        version: str

        def run(self, *, content: bytes, file_name: str,
                mime: str | None, fields_requested: list[str]) -> dict:
            '''Return:
                {
                  "text": str,                 # plain text reading
                  "blocks": [                  # optional, for template fitting
                      {"page": int, "text": str,
                       "bbox": [x,y,w,h],     # mm or px depending on engine
                       "source": "paragraph"|"cell"|"line"|"word"}
                  ],
                  "page_count": int,
                  "confidence": float | None,  # 0..1
                  "engine": str,
                }
            '''
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

log = logging.getLogger("iesocr.engines")

_REGISTRY: Dict[str, "Engine"] = {}
_FAILED_INIT: Dict[str, str] = {}


class Engine:
    name: str = "?"
    version: str = "?"

    def run(self, *, content: bytes, file_name: str, mime: Optional[str],
            fields_requested: List[str]) -> Dict:
        raise NotImplementedError


def register(engine_cls):
    try:
        inst = engine_cls()
    except Exception as e:
        _FAILED_INIT[engine_cls.__name__] = str(e)
        log.warning("engine init failed: %s → %s", engine_cls.__name__, e)
        return
    _REGISTRY[inst.name] = inst
    log.info("registered engine: %s v%s", inst.name, inst.version)


def get(name: str) -> Optional[Engine]:
    return _REGISTRY.get(name)


def available_engines() -> list:
    return sorted(_REGISTRY.keys())


def engine_details() -> list:
    return (
        [{"name": e.name, "version": e.version, "status": "ok"}
         for e in _REGISTRY.values()]
        + [{"name": k, "status": "init_failed", "error": v}
           for k, v in _FAILED_INIT.items()]
    )


# -- auto-register engines at import time
from .tesseract_engine import TesseractEngine   # noqa: E402
register(TesseractEngine)

try:
    from .docling_engine import DoclingEngine    # noqa: E402
    register(DoclingEngine)
except Exception as e:
    _FAILED_INIT["DoclingEngine"] = f"import: {e}"
    log.warning("docling import failed: %s", e)

try:
    from .easyocr_engine import EasyOcrEngine    # noqa: E402
    register(EasyOcrEngine)
except Exception as e:
    _FAILED_INIT["EasyOcrEngine"] = f"import: {e}"
    log.warning("easyocr import failed: %s", e)

try:
    from .paddleocr_engine import PaddleOcrEngine  # noqa: E402
    register(PaddleOcrEngine)
except Exception as e:
    _FAILED_INIT["PaddleOcrEngine"] = f"import: {e}"
    log.warning("paddleocr import failed: %s", e)
