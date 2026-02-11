from __future__ import annotations

import base64
import os
from typing import Dict, List
from PIL import Image
from pdf2image import convert_from_path
import uuid

from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]

from models import PipelineState, Stream, SegmentSpan
from tess_layout import extract_blocks_lines, filter_margin_blocks
from vlm_client import vlm_classify_block_font
from rashi import split_rashi_lines, run_rashi_tesseract, fill_line_text_from_tesseract
from sefaria_client import extract_streams
from align import (
    assign_blocks_to_streams,
    align_segments_to_lines_for_stream,
    align_segments_to_lines_for_stream_embeddings,
    extract_commentary_spans_from_blocks,
    match_commentary_spans_to_streams,
)
from cuts import compute_boundary_cuts_for_spans
from validate import validate_state
from db import sessions, pages
from config import RASHI_TESSDATA_DIR, USE_EMBEDDINGS_FOR_MAIN_ALIGN
from pdf_utils import ensure_local_pdf
from page_schema import session_doc_to_tzuratlink_page


def node_render_page(state: PipelineState) -> PipelineState:
    pdf_url = state["pdf_url"]
    page_index = int(state["page_index"])

    session_id = state.get("session_id") or uuid.uuid4().hex
    state["session_id"] = session_id

    local_pdf = ensure_local_pdf(pdf_url, session_id)

    images = convert_from_path(local_pdf, dpi=350, first_page=page_index+1, last_page=page_index+1)
    img = images[0]
    out_path = f"/tmp/{session_id}_p{page_index}.png"
    img.save(out_path, "PNG")

    state["page_png_path"] = out_path
    state["page_image_w"] = img.width
    state["page_image_h"] = img.height
    return state


def node_extract_blocks_lines(state: PipelineState) -> PipelineState:
    blocks, lines = extract_blocks_lines(state["page_png_path"])
    state["blocks"] = blocks
    state["lines"] = lines
    return state


def node_filter_margin_blocks(state: PipelineState) -> PipelineState:
    """Remove margin blocks before classify_block_font so we don't waste tokens on blocks we drop."""
    blocks, lines = filter_margin_blocks(state["blocks"], state["lines"])
    state["blocks"] = blocks
    state["lines"] = lines
    return state


def node_classify_block_font(state: PipelineState) -> PipelineState:
    img = Image.open(state["page_png_path"]).convert("RGB")
    blocks = state["blocks"]
    items = []
    for bid, blk in blocks.items():
        pad = 4
        x0 = max(0, blk.bbox.x - pad)
        y0 = max(0, blk.bbox.y - pad)
        x1 = min(img.width, blk.bbox.x + blk.bbox.w + pad)
        y1 = min(img.height, blk.bbox.y + blk.bbox.h + pad)
        crop = img.crop((x0, y0, x1, y1))
        items.append((bid, crop))
    if not items:
        return state
    result = vlm_classify_block_font(items)
    for bid, font in result.items():
        if bid in blocks:
            blocks[bid].font = font
    return state


def node_split_rashi_lines(state: PipelineState) -> PipelineState:
    img = Image.open(state["page_png_path"]).convert("RGB")
    split_rashi_lines(img, state["blocks"], state["lines"])
    return state


def node_rashi_tesseract(state: PipelineState) -> PipelineState:
    img = Image.open(state["page_png_path"]).convert("RGB")
    run_rashi_tesseract(img, state["blocks"], state["lines"], RASHI_TESSDATA_DIR)
    return state


def node_fill_line_text(state: PipelineState) -> PipelineState:
    """Fill line text from Tesseract only (Rashi lines: rashi_tess_text; Hebrew: default Tesseract). No VLM."""
    img = Image.open(state["page_png_path"]).convert("RGB")
    fill_line_text_from_tesseract(img, state["blocks"], state["lines"])
    return state


def node_fetch_streams(state: PipelineState) -> PipelineState:
    ref_range = state["base_ref_range"]
    raw_streams = extract_streams(ref_range)

    streams: Dict[str, Stream] = {}
    for i, (title, seg_refs, seg_texts) in enumerate(raw_streams):
        sid = f"s{i}"
        streams[sid] = Stream(stream_id=sid, title=title, seg_refs=seg_refs, seg_texts=seg_texts)

    state["streams"] = streams
    return state


def node_assign_blocks(state: PipelineState) -> PipelineState:
    unknown, unassigned = assign_blocks_to_streams(
        blocks=state["blocks"],
        lines=state["lines"],
        streams=state["streams"],
    )
    state["unknown_block_ids"] = unknown
    state["unassigned_stream_ids"] = unassigned
    return state


def node_align_segments(state: PipelineState) -> PipelineState:
    blocks = state["blocks"]
    lines = state["lines"]
    streams = state["streams"]

    spans: List[SegmentSpan] = []
    main_sid = next(iter(streams.keys()), "s0")
    for sid, st in streams.items():
        stream_line_ids = [
            lid for blk in blocks.values()
            if blk.assigned_stream_id == sid
            for lid in blk.line_ids
        ]
        if not stream_line_ids:
            continue
        if USE_EMBEDDINGS_FOR_MAIN_ALIGN and sid == main_sid:
            spans.extend(
                align_segments_to_lines_for_stream_embeddings(st, stream_line_ids, lines)
            )
        else:
            spans.extend(align_segments_to_lines_for_stream(st, stream_line_ids, lines))

    state["segment_spans"] = spans
    return state


def node_match_commentary_spans(state: PipelineState) -> PipelineState:
    """Extract spans from Rashi blocks (by is_span_end), match to commentary segments via embeddings, append to segment_spans."""
    blocks = state["blocks"]
    lines = state["lines"]
    streams = state["streams"]
    main_sid = next(iter(streams.keys()), "s0")
    commentary_spans = extract_commentary_spans_from_blocks(blocks, lines)
    matched = match_commentary_spans_to_streams(commentary_spans, streams, main_sid)
    state["segment_spans"] = list(state.get("segment_spans") or []) + matched
    return state


def node_boundary_cuts(state: PipelineState) -> PipelineState:
    img = Image.open(state["page_png_path"]).convert("RGB")
    failures = compute_boundary_cuts_for_spans(
        page_img=img,
        spans=state.get("segment_spans", []),
        streams=state["streams"],
        lines=state["lines"],
    )
    state["boundary_cut_failures"] = failures
    return state


def node_validate(state: PipelineState) -> PipelineState:
    return validate_state(state)


def route_after_validate(state: PipelineState) -> str:
    return "persist" if not state.get("needs_human_review") else "pause_for_hitl"


def node_pause_for_hitl(state: PipelineState) -> PipelineState:
    sid = state["session_id"]
    sessions().update_one(
        {"_id": sid},
        {"$set": serialize_state(state), "$setOnInsert": {"_id": sid}},
        upsert=True
    )
    return state


def node_persist(state: PipelineState) -> PipelineState:
    doc = serialize_state(state)
    page_doc = session_doc_to_tzuratlink_page(doc)
    res = pages().insert_one(page_doc)
    state["persisted_page_id"] = str(res.inserted_id)

    sid = state["session_id"]
    sessions().update_one({"_id": sid}, {"$set": doc}, upsert=True)
    return state


def serialize_state(state: PipelineState) -> dict:
    def bbox_to_d(b): return {"x": b.x, "y": b.y, "w": b.w, "h": b.h}

    out = dict(state)

    blocks = out.get("blocks") or {}
    out["blocks"] = {
        bid: {
            "block_id": blk.block_id,
            "bbox": bbox_to_d(blk.bbox),
            "line_ids": blk.line_ids,
            "font": blk.font,
            "assigned_stream_id": blk.assigned_stream_id,
            "assign_score": blk.assign_score,
        }
        for bid, blk in blocks.items()
    }

    lines = out.get("lines") or {}
    out["lines"] = {
        lid: {
            "line_id": ln.line_id,
            "block_id": ln.block_id,
            "bbox": bbox_to_d(ln.bbox),
            "order_hint": ln.order_hint,
            "tess_text_weak": ln.tess_text_weak,
            "vlm_text": ln.vlm_text,
            "vlm_conf": ln.vlm_conf,
            "is_span_end": ln.is_span_end,
            "rashi_tess_text": ln.rashi_tess_text,
        }
        for lid, ln in lines.items()
    }

    streams = out.get("streams") or {}
    out["streams"] = {
        sid: {
            "stream_id": st.stream_id,
            "title": st.title,
            "lang": st.lang,
            "seg_refs": st.seg_refs,
            "seg_texts": st.seg_texts,
        }
        for sid, st in streams.items()
    }

    spans = out.get("segment_spans") or []
    out["segment_spans"] = [
        {
            "stream_id": sp.stream_id,
            "seg_ref": sp.seg_ref,
            "start_line_id": sp.start_line_id,
            "end_line_id": sp.end_line_id,
            "end_cut_x": sp.end_cut_x,
            "score": sp.score,
            "flags": sp.flags,
        }
        for sp in spans
    ]

    # tzuratlink-data compatibility: store base64 image for finalize/page export
    if out.get("page_png_path") and os.path.isfile(out["page_png_path"]):
        try:
            with open(out["page_png_path"], "rb") as f:
                out["base64_data"] = base64.b64encode(f.read()).decode("utf-8")
        except OSError:
            pass
    return out


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("render_page", node_render_page)
    g.add_node("extract_blocks_lines", node_extract_blocks_lines)
    g.add_node("filter_margin_blocks", node_filter_margin_blocks)
    g.add_node("classify_block_font", node_classify_block_font)
    g.add_node("split_rashi_lines", node_split_rashi_lines)
    g.add_node("rashi_tesseract", node_rashi_tesseract)
    g.add_node("fill_line_text", node_fill_line_text)
    g.add_node("fetch_streams", node_fetch_streams)
    g.add_node("assign_blocks", node_assign_blocks)
    g.add_node("align_segments", node_align_segments)
    g.add_node("match_commentary_spans", node_match_commentary_spans)
    g.add_node("boundary_cuts", node_boundary_cuts)
    g.add_node("validate", node_validate)
    g.add_node("pause_for_hitl", node_pause_for_hitl)
    g.add_node("persist", node_persist)

    g.set_entry_point("render_page")
    g.add_edge("render_page", "extract_blocks_lines")
    g.add_edge("extract_blocks_lines", "filter_margin_blocks")
    g.add_edge("filter_margin_blocks", "classify_block_font")
    g.add_edge("classify_block_font", "split_rashi_lines")
    g.add_edge("split_rashi_lines", "rashi_tesseract")
    g.add_edge("rashi_tesseract", "fill_line_text")
    g.add_edge("fill_line_text", "fetch_streams")
    g.add_edge("fetch_streams", "assign_blocks")
    g.add_edge("assign_blocks", "align_segments")
    g.add_edge("align_segments", "match_commentary_spans")
    g.add_edge("match_commentary_spans", "boundary_cuts")
    g.add_edge("boundary_cuts", "validate")
    g.add_conditional_edges("validate", route_after_validate, {
        "pause_for_hitl": "pause_for_hitl",
        "persist": "persist",
    })
    g.add_edge("pause_for_hitl", END)
    g.add_edge("persist", END)

    return g.compile()
