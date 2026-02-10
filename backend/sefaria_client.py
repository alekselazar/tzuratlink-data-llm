from __future__ import annotations

from typing import Any, Dict, List, Tuple
import requests
from config import SEFARIA_BASE

def _get_json(path: str, params: Dict | None = None) -> Dict:
    url = f"{SEFARIA_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_text_with_commentary(ref_range: str) -> Dict[str, Any]:
    return _get_json(f"/api/texts/{ref_range}", params={"commentary": 1, "context": 0})

def extract_streams(
    ref_range: str,
    commentary_title_prefixes: List[str] | None = None,
) -> List[Tuple[str, List[str], List[str]]]:
    """
    Returns list of streams: (title, seg_refs, seg_texts).
    First stream is main text; rest are commentary filtered by title prefix
    (e.g. ["Rashi on", "Tosafot on"] for Gemara; set via commentary_config.json).
    """
    from config import get_commentary_title_prefixes

    if commentary_title_prefixes is None:
        commentary_title_prefixes = get_commentary_title_prefixes()

    payload = fetch_text_with_commentary(ref_range)

    streams: List[Tuple[str, List[str], List[str]]] = []

    base_title = payload.get("title", "Base")
    base_he = payload.get("he", [])
    base_segments = _flatten_segments(base_he)

    base_refs = payload.get("refs")
    if isinstance(base_refs, list) and len(base_refs) == len(base_segments):
        refs = base_refs
    else:
        refs = _make_fallback_seg_refs(ref_range, len(base_segments))

    streams.append((base_title, refs, base_segments))

    comm = payload.get("commentary", []) or payload.get("commentaries", [])
    if isinstance(comm, list):
        for c in comm:
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("collectiveTitle") or "Commentary"
            if not any(title.startswith(prefix) for prefix in commentary_title_prefixes):
                continue
            he = c.get("he", [])
            segs = _flatten_segments(he)
            if not segs:
                continue

            cref = c.get("refs")
            if isinstance(cref, list) and len(cref) == len(segs):
                c_refs = cref
            else:
                c_refs = _make_fallback_seg_refs(f"{title}:{ref_range}", len(segs))

            streams.append((title, c_refs, segs))

    return streams

def _flatten_segments(he_obj) -> List[str]:
    if isinstance(he_obj, str):
        return [he_obj]
    if isinstance(he_obj, list):
        out: List[str] = []
        for x in he_obj:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, list):
                out.extend([y for y in x if isinstance(y, str)])
        return [s for s in out if s and s.strip()]
    return []

def _make_fallback_seg_refs(ref_range: str, n: int) -> List[str]:
    return [f"{ref_range}#seg{i+1}" for i in range(n)]
