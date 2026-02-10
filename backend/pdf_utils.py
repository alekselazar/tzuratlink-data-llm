from __future__ import annotations

import os
import re
import uuid
import requests

def is_http_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s or ""))

def ensure_local_pdf(pdf_url_or_path: str, session_id: str) -> str:
    """
    If input is http(s), download to /tmp/<session_id>.pdf
    Else, return as-is (must exist inside container).
    """
    if is_http_url(pdf_url_or_path):
        out = f"/tmp/{session_id}.pdf"
        with requests.get(pdf_url_or_path, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return out

    if not os.path.exists(pdf_url_or_path):
        raise FileNotFoundError(f"pdf path not found: {pdf_url_or_path}")
    return pdf_url_or_path
