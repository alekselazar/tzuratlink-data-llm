from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from PIL import Image
import pytesseract

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/ocr")
def ocr():
    payload = request.get_json(force=True)
    images = payload.get("images", [])
    lang = payload.get("lang", "heb+eng")  # beta default

    results = []
    for item in images:
        lid = item.get("id")
        b64 = item.get("image_b64")
        if not lid or not b64:
            continue
        raw = base64.b64decode(b64.encode("utf-8"))
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Tesseract line OCR. This is only to make the beta runnable end-to-end.
        # Replace with your real recognizer later.
        txt = pytesseract.image_to_string(img, lang="heb+eng").strip()
        conf = 0.5 if txt else 0.1

        results.append({"id": lid, "text": txt, "confidence": conf})

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)
