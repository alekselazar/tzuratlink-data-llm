from __future__ import annotations

from typing import Dict
import pytesseract
from PIL import Image

from models import Block, Line, BBox


def _bbox_intersects_vertically(a: BBox, b: BBox) -> bool:
    return not (a.y + a.h <= b.y or b.y + b.h <= a.y)


def _bbox_intersects_horizontally(a: BBox, b: BBox) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x)


def _filter_margin_blocks(
    blocks: Dict[str, Block], lines: Dict[str, Line]
) -> tuple[Dict[str, Block], Dict[str, Line]]:
    """Remove margin blocks: find left-top and right-top corner blocks (no block above
    and no block to their left/right), then drop any block that intersects either
    vertically or horizontally (and drop their lines)."""
    if len(blocks) <= 1:
        return blocks, lines
    block_list = list(blocks.values())
    # Left-top: block with no block to its left and none above → leftmost, then topmost
    left_top = min(block_list, key=lambda b: (b.bbox.x, b.bbox.y))
    # Right-top: block with no block to its right and none above → rightmost, then topmost
    right_top = min(block_list, key=lambda b: (-b.bbox.x, b.bbox.y))
    to_remove: set[str] = set()
    for bid, block in blocks.items():
        bbox = block.bbox
        if _bbox_intersects_vertically(bbox, left_top.bbox) or _bbox_intersects_horizontally(bbox, left_top.bbox):
            to_remove.add(bid)
        elif _bbox_intersects_vertically(bbox, right_top.bbox) or _bbox_intersects_horizontally(bbox, right_top.bbox):
            to_remove.add(bid)
    # Drop whole margin blocks and all their lines (not only the lines)
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
