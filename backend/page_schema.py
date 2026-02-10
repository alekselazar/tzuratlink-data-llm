"""
Convert pipeline state / session doc to tzuratlink-data Page schema:
ref, source_pdf, base64_data, bboxes (normalized 0-1), created_at, updated_at
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _page_ref_from_base_ref_range(base_ref_range: str) -> str:
    """e.g. 'Berakhot 2a:1-6' -> 'Berakhot 2a', 'Berakhot 2a' -> 'Berakhot 2a'."""
    if not base_ref_range:
        return "Unknown"
    if ":" in base_ref_range:
        return base_ref_range.split(":")[0].strip()
    return base_ref_range.strip()


def _segment_spans_to_bboxes(
    lines: Dict[str, Any],
    segment_spans: List[Dict[str, Any]],
    page_w: int,
    page_h: int,
) -> List[Dict[str, Any]]:
    """
    Build tzuratlink-data bboxes from segment_spans + lines.
    One bbox per line per segment (multiple bboxes per ref allowed).
    Normalized: top, left, width, height in [0, 1].
    """
    if not lines or page_w <= 0 or page_h <= 0:
        return []

    sorted_line_ids = sorted(lines.keys(), key=lambda lid: lines[lid].get("order_hint", 0))
    line_id_to_index = {lid: i for i, lid in enumerate(sorted_line_ids)}

    bboxes: List[Dict[str, Any]] = []
    for sp in segment_spans:
        seg_ref = sp.get("seg_ref")
        start_id = sp.get("start_line_id")
        end_id = sp.get("end_line_id")
        end_cut_x = sp.get("end_cut_x")

        if not seg_ref or start_id not in line_id_to_index or end_id not in line_id_to_index:
            continue

        i_start = line_id_to_index[start_id]
        i_end = line_id_to_index[end_id]
        line_ids_in_span = sorted_line_ids[i_start : i_end + 1]

        for i, lid in enumerate(line_ids_in_span):
            ln = lines.get(lid)
            if not ln or "bbox" not in ln:
                continue
            b = ln["bbox"]
            x = int(b.get("x", 0))
            y = int(b.get("y", 0))
            w = int(b.get("w", 0))
            h = int(b.get("h", 0))

            is_last_line = lid == end_id
            if is_last_line and end_cut_x is not None:
                right = min(x + w, int(end_cut_x))
                w = max(0, right - x)

            if w <= 0 or h <= 0:
                continue

            top = y / page_h
            left = x / page_w
            width = w / page_w
            height = h / page_h
            bboxes.append({
                "ref": seg_ref,
                "top": round(top, 6),
                "left": round(left, 6),
                "width": round(width, 6),
                "height": round(height, 6),
            })

    return bboxes


def session_doc_to_tzuratlink_page(session_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a session document (from DB or serialize_state) to tzuratlink-data Page schema.
    Session doc must have: pdf_url, base_ref_range, lines, segment_spans, page_image_w, page_image_h.
    Session doc should have base64_data (set by serialize_state when page_png_path exists).
    """
    now = datetime.utcnow()
    ref = _page_ref_from_base_ref_range(session_doc.get("base_ref_range", ""))
    source_pdf = (session_doc.get("pdf_url") or "").strip() or session_doc.get("source_pdf", "")
    base64_data = session_doc.get("base64_data") or ""

    page_w = int(session_doc.get("page_image_w") or 0)
    page_h = int(session_doc.get("page_image_h") or 0)
    lines = session_doc.get("lines") or {}
    segment_spans = session_doc.get("segment_spans") or []

    bboxes = _segment_spans_to_bboxes(lines, segment_spans, page_w, page_h)

    return {
        "ref": ref,
        "source_pdf": source_pdf,
        "base64_data": base64_data,
        "bboxes": bboxes,
        "created_at": session_doc.get("created_at") or now,
        "updated_at": now,
    }
