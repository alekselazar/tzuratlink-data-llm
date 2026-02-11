const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const r = await fetch(url, options);
  const text = await r.text();
  if (!r.ok) {
    let msg = text;
    try {
      const j = JSON.parse(text);
      if (j && typeof j.error === "string") msg = j.error;
    } catch (_) {}
    throw new Error(msg);
  }
  return text ? JSON.parse(text) : {};
}

export async function startSession({ pdf_url, page_refs }) {
  return apiFetch("/api/sessions/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pdf_url, page_refs })
  });
}

/**
 * Run pipeline with SSE stream; call onStage(stageName) for each completed stage,
 * onComplete(result) when done, onError(message) on error.
 */
export function startSessionStream({ pdf_url, page_refs }, { onStage, onComplete, onError }) {
  const url = `${API_BASE}/api/sessions/start/stream`;
  fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pdf_url, page_refs })
  })
    .then(async (r) => {
      if (!r.ok) {
        const text = await r.text();
        let msg = text;
        try {
          const j = JSON.parse(text);
          if (j && typeof j.error === "string") msg = j.error;
        } catch (_) {}
        onError(msg);
        return;
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.stage) onStage(data.stage);
            if (data.status === "complete") onComplete(data);
            if (data.status === "error") onError(data.error || "Unknown error");
          } catch (_) {}
        }
      }
    })
    .catch((e) => onError(e.message || String(e)));
}

export async function getSession(session_id) {
  return apiFetch(`/api/sessions/${session_id}`);
}

export async function applyFixes(session_id, fixes) {
  return apiFetch(`/api/sessions/${session_id}/apply_fixes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fixes)
  });
}

export async function finalize(session_id) {
  return apiFetch(`/api/sessions/${session_id}/finalize`, { method: "POST" });
}
