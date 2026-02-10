"""
Line OCR via OpenAI Vision only. API key from .env via load_dotenv in config.
"""
from __future__ import annotations

import base64
import io
import time
from typing import List, Dict, Tuple, Literal

from PIL import Image
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT_S, OPENAI_MAX_RETRIES

DEFAULT_CONFIDENCE = 0.95


def _img_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vlm_read_lines(line_items: List[Tuple[str, Image.Image]]) -> Dict[str, Tuple[str, float]]:
    """
    line_items: [(line_id, PIL_crop), ...]
    returns: { line_id: (text, confidence) }
    Uses OpenAI Vision (gpt-4o / gpt-4o-mini). Key loaded from .env via config.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env and load with load_dotenv.")

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_S)
    out: Dict[str, Tuple[str, float]] = {}

    prompt = (
        "Extract the Hebrew (or Aramaic) text from this image of a single line. "
        "Return only the raw text, no explanation or punctuation changes."
    )

    for lid, img in line_items:
        b64 = _img_to_b64_png(img)
        data_url = f"data:image/png;base64,{b64}"

        last_err = None
        for attempt in range(1, OPENAI_MAX_RETRIES + 1):
            try:
                resp = client.responses.create(
                    model=OPENAI_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": data_url},
                            ],
                        }
                    ],
                    max_output_tokens=300,
                )
                content = (resp.output_text or "").strip()
                out[lid] = (content, DEFAULT_CONFIDENCE)
                break
            except Exception as e:
                last_err = e
                if attempt < OPENAI_MAX_RETRIES:
                    time.sleep(min(2 ** (attempt - 1), 8))
        else:
            out[lid] = ("", 0.0)
            if last_err:
                raise RuntimeError(f"OpenAI vision failed for line {lid}: {last_err}") from last_err

    return out


def vlm_classify_block_font(
    block_items: List[Tuple[str, Image.Image]],
) -> Dict[str, Literal["hebrew", "rashi"]]:
    """
    block_items: [(block_id, PIL_crop), ...]
    returns: { block_id: "hebrew" | "rashi" }
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env and load with load_dotenv.")

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_S)
    out: Dict[str, Literal["hebrew", "rashi"]] = {}

    prompt = (
        "Does this image show a block of text in regular Hebrew font or in Rashi script? "
        "Answer with exactly one word: hebrew or rashi."
    )

    for bid, img in block_items:
        b64 = _img_to_b64_png(img)
        data_url = f"data:image/png;base64,{b64}"
        last_err = None
        for attempt in range(1, OPENAI_MAX_RETRIES + 1):
            try:
                resp = client.responses.create(
                    model=OPENAI_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": data_url},
                            ],
                        }
                    ],
                    max_output_tokens=16,
                )
                raw = (resp.output_text or "").strip().lower()
                if raw == "rashi":
                    out[bid] = "rashi"
                else:
                    out[bid] = "hebrew"
                break
            except Exception as e:
                last_err = e
                if attempt < OPENAI_MAX_RETRIES:
                    time.sleep(min(2 ** (attempt - 1), 8))
        else:
            out[bid] = "hebrew"
            if last_err:
                raise RuntimeError(f"OpenAI block font failed for {bid}: {last_err}") from last_err
    return out
