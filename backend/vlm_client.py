from __future__ import annotations

import base64
import io
import time
from typing import List, Dict, Tuple, Literal

from PIL import Image
import time
import re
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
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env and load with load_dotenv.")

    # Sanity check: prevent accidental non-vision model usage
    if not isinstance(OPENAI_MODEL, str) or "4o" not in OPENAI_MODEL:
        raise RuntimeError(f"OPENAI_MODEL looks wrong for vision: {OPENAI_MODEL!r}")

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_S)
    out: Dict[str, Literal["hebrew", "rashi"]] = {}

    prompt = (
        "Classify the dominant script in this cropped block.\n"
        "Answer exactly one word: hebrew or rashi.\n"
        "hebrew = standard square Hebrew letters.\n"
        "rashi = Rashi script used in commentaries.\n"
        "If unsure, still choose the best of the two."
    )

    def _normalize_answer(s: str) -> str:
        s = (s or "").strip().lower()
        # keep only letters
        s = re.sub(r"[^a-z]", "", s)
        return s

    def _prep_img(img: Image.Image) -> Image.Image:
        # Ensure RGB
        im = img.convert("RGB")

        # Add padding (helps a lot)
        pad = 20
        padded = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), (255, 255, 255))
        padded.paste(im, (pad, pad))

        # Enforce minimum size by upscaling (avoid tiny unreadable crops)
        min_w, min_h = 512, 256
        scale = max(min_w / padded.width, min_h / padded.height, 1.0)
        if scale > 1.0:
            new_size = (int(padded.width * scale), int(padded.height * scale))
            padded = padded.resize(new_size, resample=Image.LANCZOS)

        return padded

    for bid, img in block_items:
        img2 = _prep_img(img)
        b64 = _img_to_b64_png(img2)
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

                raw = _normalize_answer(resp.output_text)
                if raw in ("hebrew", "rashi"):
                    out[bid] = raw  # type: ignore[assignment]
                    break

                # If model didn't follow instruction, retry once with stronger constraint
                if attempt < OPENAI_MAX_RETRIES:
                    continue

                # Final fallback: default hebrew (or choose rashi, but be explicit)
                out[bid] = "hebrew"
                break

            except Exception as e:
                last_err = e
                if attempt < OPENAI_MAX_RETRIES:
                    time.sleep(min(2 ** (attempt - 1), 8))
                else:
                    # Keep your behavior: default then raise
                    out[bid] = "hebrew"
                    raise RuntimeError(f"OpenAI block font failed for {bid}: {last_err}") from last_err

    return out