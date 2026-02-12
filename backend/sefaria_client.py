from __future__ import annotations

from typing import Any, Dict, List, Tuple, DefaultDict
from collections import defaultdict
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
    (e.g. ["Rashi on", "Tosafot on"]).
    """
    from config import get_commentary_title_prefixes

    if commentary_title_prefixes is None:
        commentary_title_prefixes = get_commentary_title_prefixes()

    payload = fetch_text_with_commentary(ref_range)

    streams: List[Tuple[str, List[str], List[str]]] = []

    # ---- Base text ----
    base_title = payload.get("title") or payload.get("book") or "Base"

    base_he = payload.get("he", [])
    base_segments = _flatten_segments(base_he)

    # v1 returns "refs" sometimes, and always has "ref" as a normalized ref string
    base_refs = payload.get("refs")
    if isinstance(base_refs, list) and len(base_refs) == len(base_segments):
        refs = base_refs
    else:
        # If we can't match segment refs, fall back to normalized base ref + seg index
        base_ref_norm = payload.get("ref") or ref_range
        refs = _make_fallback_seg_refs(base_ref_norm, len(base_segments))

    streams.append((base_title, refs, base_segments))

    # ---- Commentary links ----
    comm = payload.get("commentary", []) or payload.get("commentaries", [])
    if not isinstance(comm, list) or not comm:
        return streams

    # Group commentary link items into streams by a stable title
    grouped_refs: DefaultDict[str, List[str]] = defaultdict(list)
    grouped_texts: DefaultDict[str, List[str]] = defaultdict(list)

    for c in comm:
        if not isinstance(c, dict):
            continue

        # v1 "commentary" entries are link objects; they usually have "ref" and "he"
        cref = c.get("ref") or c.get("anchorRef") or c.get("sourceRef")
        che = c.get("he")

        if not isinstance(cref, str) or not cref.strip():
            continue
        if not isinstance(che, str) or not che.strip():
            continue

        # Determine a "title" for grouping
        # Prefer collectiveTitle if present; else derive from the ref string.
        title = c.get("collectiveTitle")
        if not isinstance(title, str) or not title.strip():
            title = _title_from_commentary_ref(cref)

        # Apply prefix filtering (same behavior you intended)
        if not any(title.startswith(prefix) for prefix in commentary_title_prefixes):
            continue

        grouped_refs[title].append(cref)
        grouped_texts[title].append(che)

    # Emit grouped streams in deterministic order
    for title in sorted(grouped_refs.keys()):
        streams.append((title, grouped_refs[title], grouped_texts[title]))

    return streams


def _title_from_commentary_ref(cref: str) -> str:
    """
    Best-effort title from a commentary ref.
    Example: "Rashi on Genesis 1:1:3" -> "Rashi on Genesis"
    Example: "Tosafot on Berakhot 2a:1" -> "Tosafot on Berakhot"
    """
    # Common format is "<Commentator> on <Base Title> <loc>"
    if " on " in cref:
        left, right = cref.split(" on ", 1)
        # right begins with base title and then location, e.g. "Genesis 1:1:3"
        base_title = right.split(" ", 1)[0] if right else ""
        # base_title can be multi-word (e.g. "Shulchan Arukh"), so be safer:
        # take tokens until we hit something that looks like a section (contains digit or daf marker)
        tokens = right.split()
        base_tokens: List[str] = []
        for t in tokens:
            if any(ch.isdigit() for ch in t) or t.endswith(("a", "b")) and any(ch.isdigit() for ch in t[:-1]):
                break
            base_tokens.append(t)
        base = " ".join(base_tokens).strip() or base_title
        return f"{left.strip()} on {base}".strip()
    # If not the standard format, just return the cref's leading chunk
    return cref.split(":", 1)[0].strip()


def _flatten_segments(he_obj: Any) -> List[str]:
    """
    Recursive flatten:
    - string -> [string]
    - nested lists -> all strings, depth-agnostic
    """
    out: List[str] = []

    def rec(x: Any) -> None:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, list):
            for y in x:
                rec(y)

    rec(he_obj)
    return out

def _make_fallback_seg_refs(ref_range: str, n: int) -> List[str]:
    return [f"{ref_range}#seg{i+1}" for i in range(n)]