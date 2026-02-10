from __future__ import annotations
from typing import List
from models import PipelineState, SegmentSpan

def validate_state(state: PipelineState) -> PipelineState:
    flags: List[str] = []
    if state.get("unknown_block_ids"):
        flags.append("unknown_blocks")
    if state.get("boundary_cut_failures"):
        flags.append("boundary_cut_failures")
    spans: List[SegmentSpan] = state.get("segment_spans", [])
    if any("align_failed" in sp.flags for sp in spans):
        flags.append("align_failures")

    state["validation_flags"] = flags
    state["needs_human_review"] = len(flags) > 0
    return state
