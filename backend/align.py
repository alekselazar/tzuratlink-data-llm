from __future__ import annotations

from typing import Dict, List, Tuple
from rapidfuzz.fuzz import token_set_ratio

from models import Block, Line, Stream, SegmentSpan

def normalize_hebrew(text: str) -> str:
    # Beta: simple whitespace normalization.
    # TODO: add niqqud stripping and final-letter normalization.
    return " ".join((text or "").strip().split())

def score_text(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_n = normalize_hebrew(a)
    b_n = normalize_hebrew(b)
    return token_set_ratio(a_n, b_n) / 100.0

def assign_blocks_to_streams(
    blocks: Dict[str, Block],
    lines: Dict[str, Line],
    streams: Dict[str, Stream],
    thresh: float = 0.25,
    block_prefix_lines: int = 10,
    stream_prefix_segs: int = 3,
) -> Tuple[List[str], List[str]]:
    stream_prefix: Dict[str, str] = {}
    for sid, st in streams.items():
        k = min(stream_prefix_segs, len(st.seg_texts))
        stream_prefix[sid] = " ".join(st.seg_texts[:k])

    unknown: List[str] = []
    for bid, blk in blocks.items():
        if getattr(blk, "font", None) == "rashi":
            blk.assigned_stream_id = None
            blk.assign_score = None
            unknown.append(bid)
            continue
        line_ids = sorted(blk.line_ids, key=lambda lid: lines[lid].order_hint)
        m = min(block_prefix_lines, len(line_ids))
        block_text = " ".join([lines[lid].vlm_text or "" for lid in line_ids[:m]]).strip()

        best_sid, best_sc = None, -1.0
        for sid, sp in stream_prefix.items():
            sc = score_text(block_text, sp)
            if sc > best_sc:
                best_sid, best_sc = sid, sc

        if best_sid is None or best_sc < thresh:
            blk.assigned_stream_id = None
            blk.assign_score = best_sc if best_sc >= 0 else None
            unknown.append(bid)
        else:
            blk.assigned_stream_id = best_sid
            blk.assign_score = best_sc

    assigned = {b.assigned_stream_id for b in blocks.values() if b.assigned_stream_id}
    unassigned = [sid for sid in streams.keys() if sid not in assigned]
    return unknown, unassigned

def align_segments_to_lines_for_stream_embeddings(
    stream: Stream,
    line_ids: List[str],
    lines: Dict[str, Line],
    window: int = 15,
    min_score: float = 0.30,
) -> List[SegmentSpan]:
    """
    Main-text alignment using OpenAI embeddings: for each segment find the contiguous
    line range that maximizes cosine similarity with the segment text.
    """
    from embeddings import get_embeddings, cosine_similarity

    if not stream.seg_refs or not line_ids:
        return []
    ordered = sorted(set(line_ids), key=lambda lid: lines[lid].order_hint)
    line_texts = [normalize_hebrew(lines[lid].vlm_text or "") for lid in ordered]
    seg_texts = [normalize_hebrew(t) for t in stream.seg_texts]
    line_emb = get_embeddings(line_texts)
    seg_emb = get_embeddings(seg_texts)

    spans: List[SegmentSpan] = []
    p = 0
    for seg_idx, (seg_ref, _) in enumerate(zip(stream.seg_refs, stream.seg_texts)):
        if p >= len(ordered):
            break
        best_q, best_sc = None, -1.0
        for q in range(p, min(len(ordered), p + window)):
            # Aggregate embedding: mean of line embeddings in [p, q]
            n = q - p + 1
            mean_emb = [sum(line_emb[i][d] for i in range(p, q + 1)) / n for d in range(len(line_emb[0]))]
            sc = cosine_similarity(mean_emb, seg_emb[seg_idx])
            if sc > best_sc:
                best_sc, best_q = sc, q
        if best_q is None or best_sc < min_score:
            spans.append(
                SegmentSpan(
                    stream_id=stream.stream_id,
                    seg_ref=seg_ref,
                    start_line_id=ordered[p],
                    end_line_id=ordered[p],
                    score=best_sc if best_sc >= 0 else None,
                    flags=["align_embed_failed"],
                )
            )
            p = min(p + 1, len(ordered))
            continue
        spans.append(
            SegmentSpan(
                stream_id=stream.stream_id,
                seg_ref=seg_ref,
                start_line_id=ordered[p],
                end_line_id=ordered[best_q],
                score=best_sc,
                flags=["align_embed"],
            )
        )
        p = best_q
    return spans


def align_segments_to_lines_for_stream(
    stream: Stream,
    line_ids: List[str],
    lines: Dict[str, Line],
    window: int = 10,
    min_score: float = 0.20,
) -> List[SegmentSpan]:
    spans: List[SegmentSpan] = []
    if not stream.seg_refs or not line_ids:
        return spans

    ordered = sorted(set(line_ids), key=lambda lid: lines[lid].order_hint)
    line_texts = [normalize_hebrew(lines[lid].vlm_text or "") for lid in ordered]
    seg_texts = [normalize_hebrew(t) for t in stream.seg_texts]

    p = 0
    for seg_ref, seg_text in zip(stream.seg_refs, seg_texts):
        if p >= len(ordered):
            break

        best_q, best_sc = None, -1.0
        for q in range(p, min(len(ordered), p + window)):
            concat = " ".join(line_texts[p:q+1]).strip()
            sc = score_text(concat, seg_text)
            if sc > best_sc:
                best_sc, best_q = sc, q

        if best_q is None or best_sc < min_score:
            spans.append(SegmentSpan(
                stream_id=stream.stream_id,
                seg_ref=seg_ref,
                start_line_id=ordered[p],
                end_line_id=ordered[p],
                score=best_sc if best_sc >= 0 else None,
                flags=["align_failed"],
            ))
            p = min(p + 1, len(ordered))
            continue

        spans.append(SegmentSpan(
            stream_id=stream.stream_id,
            seg_ref=seg_ref,
            start_line_id=ordered[p],
            end_line_id=ordered[best_q],
            score=best_sc,
            flags=[],
        ))

        # boundary sharing enabled
        p = best_q

    return spans


def extract_commentary_spans_from_blocks(
    blocks: Dict[str, Block],
    lines: Dict[str, Line],
) -> List[Tuple[str, str, str]]:
    """
    From each Rashi block, extract spans: first span = first line to first is_span_end (inclusive),
    next = from next line to next is_span_end, etc.; last span ends with last line of block.
    Returns list of (start_line_id, end_line_id, span_text).
    """
    out: List[Tuple[str, str, str]] = []
    for block in blocks.values():
        if block.font != "rashi":
            continue
        ordered = sorted(block.line_ids, key=lambda lid: lines[lid].order_hint)
        if not ordered:
            continue
        start = 0
        for i, lid in enumerate(ordered):
            if lines[lid].is_span_end or i == len(ordered) - 1:
                end_id = lid
                seg_text = " ".join(
                    (lines[lid].vlm_text or "").strip()
                    for lid in ordered[start : i + 1]
                ).strip()
                out.append((ordered[start], end_id, seg_text))
                start = i + 1
    return out


def match_commentary_spans_to_streams(
    commentary_spans: List[Tuple[str, str, str]],
    streams: Dict[str, Stream],
    main_stream_id: str,
) -> List[SegmentSpan]:
    """
    For each (start_line_id, end_line_id, span_text), find best-matching commentary
    segment (across all streams except main) via embeddings. Return list of SegmentSpan.
    """
    from embeddings import get_embeddings, cosine_similarity

    commentary_stream_ids = [sid for sid in streams.keys() if sid != main_stream_id]
    if not commentary_stream_ids or not commentary_spans:
        return []

    # All commentary segments: (stream_id, seg_ref, seg_text)
    seg_triples: List[Tuple[str, str, str]] = []
    for sid in commentary_stream_ids:
        st = streams[sid]
        for ref, text in zip(st.seg_refs, st.seg_texts):
            seg_triples.append((sid, ref, (text or "").strip()))

    if not seg_triples:
        return []

    span_texts = [t for _, _, t in commentary_spans]
    seg_texts = [t for _, _, t in seg_triples]
    span_emb = get_embeddings(span_texts)
    seg_emb = get_embeddings(seg_texts)

    result: List[SegmentSpan] = []
    for idx, (start_id, end_id, _) in enumerate(commentary_spans):
        best_j = -1
        best_sc = -1.0
        for j in range(len(seg_triples)):
            sc = cosine_similarity(span_emb[idx], seg_emb[j])
            if sc > best_sc:
                best_sc, best_j = sc, j
        if best_j < 0:
            continue
        sid, seg_ref, _ = seg_triples[best_j]
        result.append(
            SegmentSpan(
                stream_id=sid,
                seg_ref=seg_ref,
                start_line_id=start_id,
                end_line_id=end_id,
                score=best_sc,
                flags=["commentary_embed"],
            )
        )
    return result
