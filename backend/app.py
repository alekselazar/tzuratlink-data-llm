from __future__ import annotations

import json
from flask import Flask, request, jsonify, Response, stream_with_context
from bson import ObjectId

from graph import build_graph, serialize_state
from db import sessions, pages
from page_schema import session_doc_to_tzuratlink_page

app = Flask(__name__)
graph_app = build_graph()


def _require_json():
    if not request.is_json:
        return None, ("Request must be JSON", 400)
    try:
        return request.get_json(force=True), None
    except Exception as e:
        return None, (str(e), 400)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/sessions/start")
def start_session():
    state, err = _parse_start_payload()
    if err:
        resp, code = err
        return resp, code

    try:
        out = graph_app.invoke(state)
    except FileNotFoundError as e:
        return jsonify({"error": f"PDF not found: {e}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sid = out["session_id"]
    sessions().update_one(
        {"_id": sid},
        {"$set": serialize_state(out), "$setOnInsert": {"_id": sid}},
        upsert=True,
    )

    return jsonify({
        "session_id": sid,
        "needs_human_review": bool(out.get("needs_human_review")),
        "validation_flags": out.get("validation_flags", []),
        "persisted_page_id": out.get("persisted_page_id"),
    })


def _parse_start_payload():
    """Shared validation for start_session and start_session_stream. Returns (state_dict, error_response)."""
    payload, err = _require_json()
    if err:
        return None, (jsonify({"error": err[0]}), err[1])

    pdf_url = payload.get("pdf_url")
    if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.strip():
        return None, (jsonify({"error": "pdf_url is required and must be a non-empty string"}), 400)

    page_refs = payload.get("page_refs")
    if isinstance(page_refs, list) and len(page_refs) > 0:
        first_ref = page_refs[0]
        if not isinstance(first_ref, str) or not first_ref.strip():
            return None, (jsonify({"error": "page_refs[0] must be a non-empty string"}), 400)
        base_ref_range = first_ref.strip()
        page_index = 0
    else:
        base_ref_range = payload.get("base_ref_range")
        if not base_ref_range or not isinstance(base_ref_range, str) or not base_ref_range.strip():
            return None, (jsonify({"error": "base_ref_range or page_refs is required"}), 400)
        base_ref_range = base_ref_range.strip()
        try:
            page_index = int(payload.get("page_index", 0))
        except (TypeError, ValueError):
            return None, (jsonify({"error": "page_index must be an integer"}), 400)
        if page_index < 0:
            return None, (jsonify({"error": "page_index must be >= 0"}), 400)

    return {
        "pdf_url": pdf_url.strip(),
        "page_index": page_index,
        "base_ref_range": base_ref_range,
    }, None


def _sse_message(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/sessions/start/stream")
def start_session_stream():
    """Run the pipeline and stream each stage (agent) as SSE for UX."""
    state, err = _parse_start_payload()
    if err:
        resp, code = err
        return resp, code

    def generate():
        try:
            last_state = None
            for event in graph_app.stream(state, stream_mode=["updates", "values"]):
                if isinstance(event, tuple) and len(event) == 2:
                    mode, chunk = event
                    if mode == "updates" and isinstance(chunk, dict) and chunk:
                        node_name = next(iter(chunk.keys()))
                        yield _sse_message({"stage": node_name, "status": "done"})
                    elif mode == "values" and isinstance(chunk, dict):
                        last_state = chunk
                elif isinstance(event, dict) and event:
                    node_name = next(iter(event.keys()))
                    yield _sse_message({"stage": node_name, "status": "done"})

            if last_state:
                sid = last_state.get("session_id")
                if sid:
                    sessions().update_one(
                        {"_id": sid},
                        {"$set": serialize_state(last_state), "$setOnInsert": {"_id": sid}},
                        upsert=True,
                    )
                    yield _sse_message({
                        "session_id": sid,
                        "needs_human_review": bool(last_state.get("needs_human_review")),
                        "validation_flags": last_state.get("validation_flags", []),
                        "persisted_page_id": last_state.get("persisted_page_id"),
                        "status": "complete",
                    })
        except FileNotFoundError as e:
            yield _sse_message({"error": f"PDF not found: {e}", "status": "error"})
        except Exception as e:
            yield _sse_message({"error": str(e), "status": "error"})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions/<sid>")
def get_session(sid: str):
    doc = sessions().find_one({"_id": sid})
    if not doc:
        return jsonify({"error": "not_found"}), 404
    doc["_id"] = str(doc["_id"])
    return jsonify(doc)

@app.post("/api/sessions/<sid>/apply_fixes")
def apply_fixes(sid: str):
    doc = sessions().find_one({"_id": sid})
    if not doc:
        return jsonify({"error": "not_found"}), 404

    payload, err = _require_json()
    if err:
        return jsonify({"error": err[0]}), err[1]

    ba = payload.get("block_assignments") or {}
    for bid, sid_new in ba.items():
        if "blocks" in doc and bid in doc["blocks"]:
            doc["blocks"][bid]["assigned_stream_id"] = sid_new

    cut_over = payload.get("cut_overrides") or []
    for ov in cut_over:
        for sp in doc.get("segment_spans", []):
            if sp["stream_id"] == ov["stream_id"] and sp["seg_ref"] == ov["seg_ref"]:
                sp["end_cut_x"] = int(ov["end_cut_x"])
                flags = sp.get("flags", [])
                if "cut_ok" not in flags:
                    flags.append("cut_ok")
                sp["flags"] = flags

    doc["needs_human_review"] = False
    doc["validation_flags"] = []

    sessions().update_one({"_id": sid}, {"$set": doc}, upsert=True)
    return jsonify({"ok": True})

@app.post("/api/sessions/<sid>/finalize")
def finalize(sid: str):
    doc = sessions().find_one({"_id": sid})
    if not doc:
        return jsonify({"error": "not_found"}), 404

    page_doc = session_doc_to_tzuratlink_page(doc)
    res = pages().insert_one(page_doc)

    sessions().update_one({"_id": sid}, {"$set": {"persisted_page_id": str(res.inserted_id)}})
    return jsonify({"ok": True, "persisted_page_id": str(res.inserted_id)})

@app.get("/api/pages/<page_id>")
def get_page(page_id: str):
    doc = pages().find_one({"_id": ObjectId(page_id)})
    if not doc:
        return jsonify({"error": "not_found"}), 404
    doc["_id"] = str(doc["_id"])
    doc["id"] = doc["_id"]  # tzuratlink-data compatibility
    return jsonify(doc)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
