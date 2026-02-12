from __future__ import annotations

from typing import Dict
import pytesseract
from PIL import Image

from models import Block, Line, BBox


def _bbox_intersects_vertically(a: BBox, b: BBox) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x)


def _bbox_intersects_horizontally(a: BBox, b: BBox) -> bool:
    return not (a.y + a.h <= b.y or b.y + b.h <= a.y)


def _filter_margin_blocks(
    blocks: Dict[str, Block], lines: Dict[str, Line]
) -> tuple[Dict[str, Block], Dict[str, Line]]:
    """
    Remove margin blocks:
    1) Find left-top and right-top corner blocks from TOP band only.
    2) Drop blocks that intersect the corner blocks *by strip/ray* (vertical strip = x-overlap,
       horizontal strip = y-overlap), but ONLY if those blocks are likely margin:
         - in left/right/top margin zone, OR
         - very small area
    """
    if len(blocks) <= 1:
        return blocks, lines

    block_list = list(blocks.values())

    # Estimate page size from bboxes (works even if caller doesn't pass image size)
    page_w = max((b.bbox.x + b.bbox.w) for b in block_list)
    page_h = max((b.bbox.y + b.bbox.h) for b in block_list)
    if page_w <= 0 or page_h <= 0:
        return blocks, lines

    # Tunable heuristics
    top_band_h = int(0.12 * page_h)          # search seeds in top 12% of page
    margin_x = int(0.12 * page_w)            # left/right margin width
    margin_y = int(0.10 * page_h)            # top margin height
    small_area_thr = 0.002 * page_w * page_h # tiny blocks are likely headers/folio/page marks

    # Prefer seeds from the top band; fallback to all blocks if none qualify
    top_candidates = [b for b in block_list if b.bbox.y < top_band_h]
    if not top_candidates:
        top_candidates = block_list

    # Left-top seed: leftmost, then topmost (within top band)
    left_top = min(top_candidates, key=lambda b: (b.bbox.x, b.bbox.y))

    # Right-top seed: by right edge, then topmost (within top band)
    right_top = min(top_candidates, key=lambda b: (-(b.bbox.x + b.bbox.w), b.bbox.y))

    def _in_left_margin(bb: BBox) -> bool:
        return bb.x < margin_x

    def _in_right_margin(bb: BBox) -> bool:
        return (bb.x + bb.w) > (page_w - margin_x)

    def _in_top_margin(bb: BBox) -> bool:
        return bb.y < margin_y

    def _is_small(bb: BBox) -> bool:
        return (bb.w * bb.h) < small_area_thr

    def _is_margin_like(bb: BBox) -> bool:
        # Only delete blocks that are plausibly margin junk.
        return _in_left_margin(bb) or _in_right_margin(bb) or _in_top_margin(bb) or _is_small(bb)

    to_remove: set[str] = set()

    for bid, block in blocks.items():
        bb = block.bbox

        # Keep obviously non-margin blocks even if they align with a seed strip
        if not _is_margin_like(bb):
            continue

        # "Intersects vertically": x-overlap with seed strip
        # "Intersects horizontally": y-overlap with seed strip
        hit_left = _bbox_intersects_vertically(bb, left_top.bbox) or _bbox_intersects_horizontally(bb, left_top.bbox)
        hit_right = _bbox_intersects_vertically(bb, right_top.bbox) or _bbox_intersects_horizontally(bb, right_top.bbox)

        if hit_left or hit_right:
            to_remove.add(bid)

    new_blocks = {bid: b for bid, b in blocks.items() if bid not in to_remove}
    new_lines = {lid: ln for lid, ln in lines.items() if ln.block_id not in to_remove}

    for b in new_blocks.values():
        b.line_ids = [lid for lid in b.line_ids if lid in new_lines]

    return new_blocks, new_lines

def filter_margin_blocks(
    blocks: Dict[str, Block], lines: Dict[str, Line]
) -> tuple[Dict[str, Block], Dict[str, Line]]:
    """Public wrapper: remove margin blocks and their lines. Call before classify_block_font to save tokens."""
    return _filter_margin_blocks(blocks, lines)


def _order_hint(b: BBox) -> float:
    return b.y * 1_000_000 + b.x

def extract_blocks_lines(png_path: str) -> tuple[Dict[str, Block], Dict[str, Line]]:
    img = Image.open(png_path).convert("RGB")
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    blocks: Dict[str, Block] = {}
    lines: Dict[str, Line] = {}

    n = len(data["level"])

    for i in range(n):
        if data["level"][i] != 2:
            continue
        block_num = data["block_num"][i]
        bid = f"b{block_num}"
        bbox = BBox(
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            w=int(data["width"][i]),
            h=int(data["height"][i]),
        )
        blocks[bid] = Block(block_id=bid, bbox=bbox, line_ids=[])

    for i in range(n):
        if data["level"][i] != 4:
            continue
        block_num = data["block_num"][i]
        par_num = data["par_num"][i]
        line_num = data["line_num"][i]
        bid = f"b{block_num}"
        lid = f"l{block_num}_{par_num}_{line_num}"

        bbox = BBox(
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            w=int(data["width"][i]),
            h=int(data["height"][i]),
        )
        ln = Line(
            line_id=lid,
            block_id=bid,
            bbox=bbox,
            order_hint=_order_hint(bbox),
        )
        lines[lid] = ln
        if bid not in blocks:
            blocks[bid] = Block(block_id=bid, bbox=bbox, line_ids=[])
        blocks[bid].line_ids.append(lid)

    return blocks, lines
