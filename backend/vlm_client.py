from __future__ import annotations

import base64
import io
import time
from typing import List, Dict, Tuple, Literal

from PIL import Image
import time
import math
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT_S, OPENAI_MAX_RETRIES

DEFAULT_CONFIDENCE = 0.95


def _img_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vlm_classify_block_font(
    block_items: List[Tuple[str, Image.Image]],
) -> Dict[str, Literal["hebrew", "rashi"]]:
    """
    block_items: [(block_id, PIL_crop), ...]
    returns: { block_id: "hebrew" | "rashi" }
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env and load with load_dotenv.")

    # Optional but highly recommended: fail fast if you accidentally configured a non-vision model
    if not isinstance(OPENAI_MODEL, str) or "4o" not in OPENAI_MODEL:
        raise RuntimeError(f"OPENAI_MODEL doesn't look like a vision-capable 4o model: {OPENAI_MODEL!r}")

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_S)
    out: Dict[str, Literal["hebrew", "rashi"]] = {}

    prompt = (
        "Does this image show a block of text in regular Hebrew font or in Rashi script? "
        "Answer with exactly one word: hebrew or rashi."
    )

    # Safety thresholds to avoid 400 '$.input is invalid' due to empty/huge images
    MIN_W, MIN_H = 20, 20
    MAX_PIXELS = 2_000_000  # cap area to keep the data URL reasonable (tune if needed)

    for bid, img in block_items:
        # ---- Validation: image must be present and non-trivial ----
        if img is None:
            out[bid] = "hebrew"
            continue

        if getattr(img, "width", 0) < MIN_W or getattr(img, "height", 0) < MIN_H:
            out[bid] = "hebrew"
            continue

        # Ensure RGB (avoid odd modes causing encoding issues)
        img2 = img.convert("RGB")

        # ---- Validation: cap image size to prevent huge base64 data URLs ----
        area = img2.width * img2.height
        if area > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / float(area))
            new_w = max(MIN_W, int(img2.width * scale))
            new_h = max(MIN_H, int(img2.height * scale))
            img2 = img2.resize((new_w, new_h), resample=Image.LANCZOS)

        # Encode
        b64 = _img_to_b64_png(img2)

        # ---- Validation: base64 must be non-empty ----
        if not b64 or len(b64) < 100:
            out[bid] = "hebrew"
            continue

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

                # More robust parse: accept only exact labels (strip punctuation)
                raw_clean = "".join(ch for ch in raw if ch.isalpha())

                if raw_clean == "rashi":
                    out[bid] = "rashi"
                elif raw_clean == "hebrew":
                    out[bid] = "hebrew"
                else:
                    # If model didn't follow instruction, default (or you can retry with a stricter prompt)
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