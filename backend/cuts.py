from __future__ import annotations

from typing import List, Tuple, Dict
import pytesseract
from PIL import Image
from rapidfuzz.fuzz import ratio

from models import SegmentSpan, Stream, Line, BBox

def normalize_hebrew(text: str) -> str:
    return " ".join((text or "").strip().split())

def last_word(seg_text: str) -> str:
    t = normalize_hebrew(seg_text)
    parts = [p for p in t.split() if p]
    return parts[-1] if parts else ""

def tesseract_word_boxes_for_crop(img: Image.Image) -> List[Tuple[str, BBox]]:
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    out: List[Tuple[str, BBox]] = []
    n = len(data["level"])
    for i in range(n):
        if data["level"][i] != 5:
            continue
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        out.append((txt, BBox(
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            w=int(data["width"][i]),
            h=int(data["height"][i]),
        )))
    return out

def compute_boundary_cuts_for_spans(
    page_img: Image.Image,
    spans: List[SegmentSpan],
    streams: Dict[str, Stream],
    lines: Dict[str, Line],
    word_match_thresh: float = 60.0,
) -> List[Tuple[str, str]]:
    failures: List[Tuple[str, str]] = []

    for sp in spans:
        st = streams.get(sp.stream_id)
        if not st:
            sp.flags.append("missing_stream")
            failures.append((sp.stream_id, sp.seg_ref))
            continue

        try:
            idx = st.seg_refs.index(sp.seg_ref)
            seg_text = st.seg_texts[idx]
        except ValueError:
            sp.flags.append("missing_seg_ref")
            failures.append((sp.stream_id, sp.seg_ref))
            continue

        lw = last_word(seg_text)
        if not lw:
            sp.flags.append("no_last_word")
            failures.append((sp.stream_id, sp.seg_ref))
            continue

        end_line = lines.get(sp.end_line_id)
        if not end_line:
            sp.flags.append("missing_end_line")
            failures.append((sp.stream_id, sp.seg_ref))
            continue

        pad = 6
        x0 = max(0, end_line.bbox.x - pad)
        y0 = max(0, end_line.bbox.y - pad)
        x1 = min(page_img.width, end_line.bbox.x + end_line.bbox.w + pad)
        y1 = min(page_img.height, end_line.bbox.y + end_line.bbox.h + pad)
        crop = page_img.crop((x0, y0, x1, y1))

        word_boxes = tesseract_word_boxes_for_crop(crop)

        best_bbox = None
        best_sc = -1.0
        lw_n = normalize_hebrew(lw)

        for wt, wb in word_boxes:
            sc = ratio(normalize_hebrew(wt), lw_n)
            if sc > best_sc:
                best_sc = sc
                best_bbox = wb

        if best_bbox is None or best_sc < word_match_thresh:
            sp.flags.append("cut_failed")
            failures.append((sp.stream_id, sp.seg_ref))
            continue

        sp.end_cut_x = x0 + best_bbox.x
        sp.flags.append("cut_ok")

    return failures
