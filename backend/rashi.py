"""
Rashi blocks: split lines by colon-words (vertical split at word left edge),
order segments right-first, mark span ends, run Tesseract with rashi.tessdata.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import pytesseract
from PIL import Image

from models import Block, Line, BBox


def _word_boxes_for_crop(img: Image.Image) -> List[Tuple[str, BBox]]:
    """Word-level bboxes in crop-local coordinates (level=5)."""
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


def _split_points_for_line(line: Line, crop: Image.Image) -> List[int]:
    """Left-edge x (page coords) of every word that ends with ':' in this line crop."""
    word_boxes = _word_boxes_for_crop(crop)
    line_x = line.bbox.x
    points: List[int] = []
    for txt, box in word_boxes:
        if (txt or "").rstrip().endswith(":"):
            points.append(line_x + box.x)
    return sorted(set(points))


def split_rashi_lines(
    page_img: Image.Image,
    blocks: Dict[str, Block],
    lines: Dict[str, Line],
) -> None:
    """
    For each Rashi block, split every line at words ending with ':' (split at word left edge).
    Replace each such line with segment-lines ordered right-first; mark segments as span ends.
    Mutates blocks and lines in place.
    """
    for block in list(blocks.values()):
        if block.font != "rashi":
            continue
        new_line_ids: List[str] = []
        for lid in block.line_ids:
            ln = lines.get(lid)
            if not ln:
                continue
            pad = 2
            x0 = max(0, ln.bbox.x - pad)
            y0 = max(0, ln.bbox.y - pad)
            x1 = min(page_img.width, ln.bbox.x + ln.bbox.w + pad)
            y1 = min(page_img.height, ln.bbox.y + ln.bbox.h + pad)
            crop = page_img.crop((x0, y0, x1, y1))
            split_xs = _split_points_for_line(ln, crop)
            if not split_xs:
                new_line_ids.append(lid)
                continue
            # Segments: [line.x, p1), [p1, p2), ..., [pk, line.x+line.w]
            left_edges = [ln.bbox.x] + split_xs
            right_edges = split_xs + [ln.bbox.x + ln.bbox.w]
            segments: List[Tuple[int, int]] = [
                (left_edges[i], right_edges[i]) for i in range(len(left_edges))
            ]
            # Order: rightmost segment first (largest x first)
            segments.sort(key=lambda seg: -seg[0])
            # Create new line records (rightmost = index 0)
            for idx, (seg_x, seg_right) in enumerate(segments):
                seg_w = seg_right - seg_x
                seg_id = f"{lid}_s{idx}"
                seg_line = Line(
                    line_id=seg_id,
                    block_id=block.block_id,
                    bbox=BBox(x=seg_x, y=ln.bbox.y, w=seg_w, h=ln.bbox.h),
                    order_hint=ln.order_hint + idx * 0.0001,
                    is_span_end=True,
                )
                lines[seg_id] = seg_line
                new_line_ids.append(seg_id)
            del lines[lid]
        block.line_ids = new_line_ids


def run_rashi_tesseract(
    page_img: Image.Image,
    blocks: Dict[str, Block],
    lines: Dict[str, Line],
    tessdata_dir: str,
) -> None:
    """
    For every line in a Rashi block, run Tesseract with rashi.tessdata on the line crop
    and set line.rashi_tess_text. Mutates lines in place.
    """
    config = f"--tessdata-dir {tessdata_dir}"
    for block in blocks.values():
        if block.font != "rashi":
            continue
        for lid in block.line_ids:
            ln = lines.get(lid)
            if not ln:
                continue
            pad = 2
            x0 = max(0, ln.bbox.x - pad)
            y0 = max(0, ln.bbox.y - pad)
            x1 = min(page_img.width, ln.bbox.x + ln.bbox.w + pad)
            y1 = min(page_img.height, ln.bbox.y + ln.bbox.h + pad)
            crop = page_img.crop((x0, y0, x1, y1))
            try:
                text = pytesseract.image_to_string(crop, lang="rashi", config=config)
                ln.rashi_tess_text = (text or "").strip()
            except Exception:
                ln.rashi_tess_text = None


def fill_line_text_from_tesseract(
    page_img: Image.Image,
    blocks: Dict[str, Block],
    lines: Dict[str, Line],
) -> None:
    """
    Set vlm_text for every line from Tesseract only (no VLM).
    Rashi lines: use existing rashi_tess_text. Hebrew lines: run default Tesseract on crop.
    """
    for block in blocks.values():
        for lid in block.line_ids:
            ln = lines.get(lid)
            if not ln:
                continue
            if ln.rashi_tess_text is not None:
                ln.vlm_text = ln.rashi_tess_text
                continue
            pad = 2
            x0 = max(0, ln.bbox.x - pad)
            y0 = max(0, ln.bbox.y - pad)
            x1 = min(page_img.width, ln.bbox.x + ln.bbox.w + pad)
            y1 = min(page_img.height, ln.bbox.y + ln.bbox.h + pad)
            crop = page_img.crop((x0, y0, x1, y1))
            try:
                text = pytesseract.image_to_string(crop)
                ln.vlm_text = (text or "").strip()
            except Exception:
                ln.vlm_text = None
