from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, TypedDict, Tuple

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

@dataclass
class Line:
    line_id: str
    block_id: str
    bbox: BBox
    order_hint: float
    tess_text_weak: Optional[str] = None
    vlm_text: Optional[str] = None
    vlm_conf: Optional[float] = None
    is_span_end: bool = False
    rashi_tess_text: Optional[str] = None

@dataclass
class Block:
    block_id: str
    bbox: BBox
    line_ids: List[str] = field(default_factory=list)
    font: Optional[Literal["hebrew", "rashi"]] = None
    assigned_stream_id: Optional[str] = None
    assign_score: Optional[float] = None

@dataclass
class Stream:
    stream_id: str
    title: str
    lang: Literal["he"] = "he"
    seg_refs: List[str] = field(default_factory=list)
    seg_texts: List[str] = field(default_factory=list)

@dataclass
class SegmentSpan:
    stream_id: str
    seg_ref: str
    start_line_id: str
    end_line_id: str
    end_cut_x: Optional[int] = None
    score: Optional[float] = None
    flags: List[str] = field(default_factory=list)

class PipelineState(TypedDict, total=False):
    session_id: str

    pdf_url: str
    page_index: int
    base_ref_range: str

    page_png_path: str
    page_image_w: int
    page_image_h: int

    blocks: Dict[str, Block]
    lines: Dict[str, Line]

    streams: Dict[str, Stream]

    unknown_block_ids: List[str]
    unassigned_stream_ids: List[str]

    segment_spans: List[SegmentSpan]
    boundary_cut_failures: List[Tuple[str, str]]

    validation_flags: List[str]
    needs_human_review: bool

    persisted_page_id: Optional[str]
