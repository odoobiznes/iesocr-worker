"""Apply a template (from the C# TemplateField[] structure) to an OCR result.

Template field shapes (see IES_OCR_DESIGN.md §4):

  { "Name": ..., "Type": "String|Number|Money|Date|Bool|Enum",
    "Required": bool, "LlmPinned": bool,
    "Extractor": {
      "Kind": "regex_after|after_label|cell_to_right_of|cell_below|
               bbox_absolute|bbox_relative_to_anchor|field_in_surya_table",
      "AnchorLabel": str?, "Regex": str?, "Capture": {field: groupIndex}?,
      "Post": [{"Fn":"parse_date|strip_whitespace|clip_length|split_words|
                       enum_map", "Args": {...}}]?,
      "Bbox": {"Page":..., "XMm":..., "YMm":..., "WidthMm":..., "HeightMm":...}?
    }
  }
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger("iesocr.template_runner")


def apply_template(template: Dict, ocr_result: Dict) -> Dict[str, Any]:
    """Apply each non-LLM-pinned field extractor to the OCR text / blocks."""
    text: str = ocr_result.get("text") or ""
    blocks: list = ocr_result.get("blocks") or []
    fields_spec = template.get("Fields") or template.get("fields") or []

    out: Dict[str, Any] = {}
    for f in fields_spec:
        name = f.get("Name") or f.get("name")
        if not name: continue
        if f.get("LlmPinned") or f.get("llmPinned"):
            continue
        ex = f.get("Extractor") or f.get("extractor") or {}
        try:
            val = _run_extractor(ex, text, blocks)
        except Exception as e:
            log.debug("extractor %s failed: %s", name, e)
            val = None
        if val is None:
            continue
        val = _post_process(val, ex.get("Post") or ex.get("post") or [])
        if val is None or val == "":
            continue
        # Capture-map (single-kind extractors that return a dict)
        if isinstance(val, dict):
            for k, v in val.items():
                out[k] = v
        else:
            out[name] = val
    return out


# ------------------------- extractor kinds -------------------------


def _run_extractor(ex: Dict, text: str, blocks: List[Dict]) -> Any:
    kind = (ex.get("Kind") or ex.get("kind") or "").lower()
    anchor = ex.get("AnchorLabel") or ex.get("anchorLabel")
    regex = ex.get("Regex") or ex.get("regex")
    capture = ex.get("Capture") or ex.get("capture")

    if kind in ("regex_after",):
        return _regex_after(text, anchor, regex, capture)
    if kind in ("after_label", "after_colon"):
        return _after_label(text, anchor)
    if kind in ("cell_to_right_of",):
        return _cell_to_right_of(blocks, anchor)
    if kind in ("cell_below",):
        return _cell_below(blocks, anchor)
    if kind in ("bbox_absolute", "bbox_relative_to_anchor"):
        # Phase 1: not implemented (needs pt↔mm conversion with DPI). Fall through.
        return None
    if kind in ("llm_pinned",):
        return None
    return None


def _regex_after(text: str, anchor: Optional[str], regex: Optional[str],
                 capture: Optional[Dict[str, int]]) -> Any:
    if not regex: return None
    # If anchor given, restrict search region to 500 chars after first hit.
    region = text
    if anchor:
        at = _find_anchor(text, anchor)
        if at >= 0: region = text[at:at + 800]
    m = re.search(regex, region, flags=re.IGNORECASE | re.UNICODE)
    if not m: return None
    if capture:
        return {name: m.group(idx) for name, idx in capture.items()
                if idx <= m.lastindex}
    return m.group(1) if m.lastindex else m.group(0)


def _after_label(text: str, anchor: Optional[str]) -> Optional[str]:
    """Return value following ``anchor``. Handles plain rows + markdown tables."""
    if not anchor: return None
    extent = _find_anchor_extent(text, anchor)
    if not extent: return None
    tail = text[extent[1]:]
    nl = tail.find(chr(10))
    line_tail = tail[:nl] if nl >= 0 else tail[:500]
    # Markdown-table layout (docling output for xlsx/docx) — pick first non-empty
    # non-separator non-anchor cell.
    if "|" in line_tail:
        for c in (cc.strip() for cc in line_tail.split("|")):
            if not c: continue
            if re.fullmatch(r"[-:\s]+", c): continue
            if _normalize(c).startswith(_normalize(anchor)): continue
            return c[:200]
        return None
    val = line_tail.lstrip(" 	:| ").strip()
    return (val[:200] or None)


def _cell_to_right_of(blocks: List[Dict], anchor: Optional[str]) -> Optional[str]:
    if not anchor or not blocks: return None
    anchor_norm = _normalize(anchor)
    # find a block matching anchor; return the closest block to its right
    # on the same (±8 px) row.
    anchor_block = None
    for b in blocks:
        if _normalize(b.get("text", "")).find(anchor_norm) >= 0:
            anchor_block = b
            break
    if not anchor_block or not anchor_block.get("bbox"):
        return None
    ax, ay, aw, ah = anchor_block["bbox"]
    best = None; best_dx = 1e9
    for b in blocks:
        if b is anchor_block: continue
        bx = (b.get("bbox") or [0,0,0,0])[0]
        by = (b.get("bbox") or [0,0,0,0])[1]
        if bx <= ax + aw: continue            # not to the right
        if abs(by - ay) > max(ah, 20): continue  # not same row
        dx = bx - (ax + aw)
        if dx < best_dx:
            best_dx = dx; best = b
    return best.get("text") if best else None


def _cell_below(blocks: List[Dict], anchor: Optional[str]) -> Optional[str]:
    if not anchor or not blocks: return None
    anchor_norm = _normalize(anchor)
    anchor_block = None
    for b in blocks:
        if _normalize(b.get("text", "")).find(anchor_norm) >= 0:
            anchor_block = b
            break
    if not anchor_block or not anchor_block.get("bbox"):
        return None
    ax, ay, aw, ah = anchor_block["bbox"]
    best = None; best_dy = 1e9
    for b in blocks:
        if b is anchor_block: continue
        bx = (b.get("bbox") or [0,0,0,0])[0]
        by = (b.get("bbox") or [0,0,0,0])[1]
        bw = (b.get("bbox") or [0,0,0,0])[2]
        if by <= ay + ah: continue              # not below
        if bx + bw < ax or bx > ax + aw: continue  # not overlapping column
        dy = by - (ay + ah)
        if dy < best_dy:
            best_dy = dy; best = b
    return best.get("text") if best else None


# ------------------------- post-processors -------------------------


def _post_process(value: Any, post: List[Dict]) -> Any:
    if value is None: return None
    if isinstance(value, dict):
        return {k: _apply_chain(v, post) for k, v in value.items()}
    return _apply_chain(value, post)


def _apply_chain(v: Any, post: List[Dict]) -> Any:
    for p in post or []:
        fn = (p.get("Fn") or p.get("fn") or "").lower()
        args = p.get("Args") or p.get("args") or {}
        if not isinstance(v, str): continue
        v = _apply_one(v, fn, args)
        if v is None: break
    return v


def _apply_one(v: str, fn: str, args: Dict) -> Any:
    if fn == "strip_whitespace":
        return re.sub(r"\s+", " ", v).strip()
    if fn == "clip_length":
        m = args.get("max") or args.get("Max") or 0
        try: m = int(m) if not isinstance(m, int) else m
        except Exception: m = 0
        return v[:m] if m and len(v) > m else v
    if fn == "parse_date":
        formats = args.get("formats") or args.get("Formats") or \
            ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"]
        for f in formats:
            # accept both .NET and strftime tokens
            pyfmt = _dotnet_to_strftime(f)
            try: return datetime.strptime(v.strip(), pyfmt).date().isoformat()
            except ValueError: continue
        return v
    if fn == "split_words":
        parts = v.strip().split()
        take = (args.get("take") or args.get("Take") or "").lower()
        if take == "first": return parts[0] if parts else None
        if take == "last": return parts[-1] if parts else None
        if take == "first-but-last": return " ".join(parts[:-1]) if len(parts) > 1 else None
        return v
    if fn == "enum_map":
        m = args.get("map") or args.get("Map") or {}
        return m.get(v.strip(), v)
    return v


# ------------------------- helpers -------------------------


def _find_anchor(text: str, anchor: str) -> int:
    """Return offset in `text` where (normalized) `anchor` starts, or -1.
    Search is diacritic-folded + lowercase; offset is in ORIGINAL text indices
    because NFKD normalisation rarely changes character count for Latin scripts
    (each combining mark is removed but the base character keeps position).
    For tighter offset alignment use _find_anchor_extent() below.
    """
    nt = _normalize(text); na = _normalize(anchor)
    return nt.find(na)


def _find_anchor_extent(text: str, anchor: str):
    """Return (start, end) in ORIGINAL `text` matching `anchor` diacritic-fold.
    end-start gives the actual length to skip; differs from len(anchor) when
    diacritic-stripped text shifts character counts.
    """
    nt = _normalize(text); na = _normalize(anchor)
    at = nt.find(na)
    if at < 0: return None
    # Walk original text mapping normalized index → original index
    o = 0; n = 0
    while o < len(text) and n < at:
        nc = _normalize(text[o])
        n += len(nc); o += 1
    start = o
    while o < len(text) and (n - at) < len(na):
        nc = _normalize(text[o])
        n += len(nc); o += 1
    return (start, o)


def _normalize(s: str) -> str:
    if not s: return ""
    nfkd = unicodedata.normalize("NFKD", s)
    out = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", out).strip().lower()


def _dotnet_to_strftime(f: str) -> str:
    # map most common .NET date format chars → strftime
    return (f.replace("yyyy", "%Y")
              .replace("MM", "%m").replace("M", "%m")
              .replace("dd", "%d").replace("d", "%d")
              .replace("HH", "%H").replace("mm", "%M").replace("ss", "%S"))
