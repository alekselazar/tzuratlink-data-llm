import React, { useEffect, useState } from "react";
import { getSession, applyFixes, finalize } from "../api";

/**
 * Build list of refs with their bboxes (normalized 0-1) and text from session doc.
 */
function buildRefsWithBboxes(doc) {
  const lines = doc.lines || {};
  const spans = doc.segment_spans || [];
  const streams = doc.streams || {};
  const pageW = doc.page_image_w || 1;
  const pageH = doc.page_image_h || 1;
  if (pageW <= 0 || pageH <= 0) return [];

  const sortedLineIds = Object.keys(lines).sort(
    (a, b) => (lines[a].order_hint ?? 0) - (lines[b].order_hint ?? 0)
  );
  const lineIdToIndex = {};
  sortedLineIds.forEach((lid, i) => {
    lineIdToIndex[lid] = i;
  });

  const refToText = {};
  Object.values(streams).forEach((st) => {
    (st.seg_refs || []).forEach((ref, i) => {
      refToText[ref] = (st.seg_texts || [])[i] ?? "";
    });
  });

  const refToBboxes = {};
  spans.forEach((sp) => {
    const segRef = sp.seg_ref;
    const startId = sp.start_line_id;
    const endId = sp.end_line_id;
    const endCutX = sp.end_cut_x;
    if (!segRef || lineIdToIndex[startId] == null || lineIdToIndex[endId] == null) return;
    const iStart = lineIdToIndex[startId];
    const iEnd = lineIdToIndex[endId];
    const lineIdsInSpan = sortedLineIds.slice(iStart, iEnd + 1);
    lineIdsInSpan.forEach((lid) => {
      const ln = lines[lid];
      if (!ln?.bbox) return;
      const b = ln.bbox;
      let x = b.x ?? 0,
        y = b.y ?? 0,
        w = b.w ?? 0,
        h = b.h ?? 0;
      const isLast = lid === endId;
      if (isLast && endCutX != null) {
        const right = Math.min(x + w, endCutX);
        w = Math.max(0, right - x);
      }
      if (w <= 0 || h <= 0) return;
      const box = {
        top: y / pageH,
        left: x / pageW,
        width: w / pageW,
        height: h / pageH,
      };
      if (!refToBboxes[segRef]) refToBboxes[segRef] = [];
      refToBboxes[segRef].push(box);
    });
  });

  return Object.entries(refToBboxes).map(([ref, bboxes]) => ({
    ref,
    text: refToText[ref] ?? "",
    bboxes,
  }));
}

export default function SessionView({ sessionId, startResult }) {
  const [doc, setDoc] = useState(null);
  const [err, setErr] = useState("");
  const [selectedRef, setSelectedRef] = useState(null);
  const [fixJson, setFixJson] = useState(`{
  "block_assignments": {},
  "cut_overrides": []
}`);

  async function refresh() {
    setErr("");
    try {
      const d = await getSession(sessionId);
      setDoc(d);
    } catch (e) {
      setErr(String(e));
    }
  }

  useEffect(() => {
    refresh();
  }, [sessionId]);

  async function onApplyFixes() {
    setErr("");
    try {
      const fixes = JSON.parse(fixJson);
      await applyFixes(sessionId, fixes);
      await refresh();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function onFinalize() {
    setErr("");
    try {
      const res = await finalize(sessionId);
      await refresh();
      alert(`Finalized: ${res.persisted_page_id}`);
    } catch (e) {
      setErr(String(e));
    }
  }

  const refsWithBboxes = doc ? buildRefsWithBboxes(doc) : [];
  const selected =
    (selectedRef ? refsWithBboxes.find((r) => r.ref === selectedRef) : null) ??
    refsWithBboxes[0] ??
    null;

  return (
    <div style={{ fontFamily: "sans-serif", padding: 16, maxWidth: 1400 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>Page for review</h2>
        <span style={{ color: "#666" }}>Session {sessionId}</span>
        <button onClick={refresh}>Refresh</button>
        <button onClick={onFinalize}>Finalize</button>
      </div>

      {err ? <pre style={{ color: "crimson" }}>{err}</pre> : null}

      {doc ? (
        <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 16, marginTop: 16 }}>
          {/* Left: ref list + selected ref text */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <h3 style={{ margin: "0 0 8px 0" }}>References</h3>
              <p style={{ margin: 0, fontSize: 13, color: "#666" }}>
                Click a ref to highlight its bboxes on the page.
              </p>
              <ul
                style={{
                  listStyle: "none",
                  padding: 0,
                  margin: "8px 0 0 0",
                  maxHeight: 280,
                  overflowY: "auto",
                  border: "1px solid #ddd",
                  borderRadius: 6,
                }}
              >
                {refsWithBboxes.map(({ ref }) => (
                  <li key={ref}>
                    <button
                      type="button"
                      onClick={() => setSelectedRef(ref)}
                      style={{
                        display: "block",
                        width: "100%",
                                textAlign: "left",
                        padding: "8px 12px",
                        border: "none",
                        background: selectedRef === ref ? "#e0f0ff" : "transparent",
                        cursor: "pointer",
                        fontSize: 14,
                        borderRadius: 4,
                      }}
                    >
                      {ref}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
            {selected && (
              <div
                style={{
                  border: "1px solid #ddd",
                  borderRadius: 6,
                  padding: 12,
                  background: "#fafafa",
                }}
              >
                <h4 style={{ margin: "0 0 8px 0" }}>{selected.ref}</h4>
                <p
                  style={{
                    margin: 0,
                    fontSize: 14,
                    lineHeight: 1.5,
                    whiteSpace: "pre-wrap",
                    fontFamily: "serif",
                  }}
                >
                  {selected.text || "(no text)"}
                </p>
              </div>
            )}
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: "pointer" }}>Validation &amp; fixes</summary>
              <pre style={{ fontSize: 12, overflow: "auto", maxHeight: 120 }}>
                {JSON.stringify(
                  {
                    needs_human_review: doc.needs_human_review,
                    validation_flags: doc.validation_flags,
                  },
                  null,
                  2
                )}
              </pre>
              <textarea
                style={{ width: "100%", height: 80, fontFamily: "monospace", fontSize: 12 }}
                value={fixJson}
                onChange={(e) => setFixJson(e.target.value)}
              />
              <button type="button" onClick={onApplyFixes} style={{ marginTop: 4 }}>
                Apply fixes
              </button>
            </details>
          </div>

          {/* Right: page image with bbox overlay */}
          <div>
            <h3 style={{ margin: "0 0 8px 0" }}>
              {selected ? `Highlighted: ${selected.ref}` : "Page image"}
            </h3>
            {doc.base64_data ? (
              <div
                style={{
                  position: "relative",
                  maxWidth: "100%",
                  background: "#f0f0f0",
                  borderRadius: 8,
                  overflow: "hidden",
                }}
              >
                <img
                  src={`data:image/png;base64,${doc.base64_data}`}
                  alt="Page"
                  style={{ display: "block", width: "100%", height: "auto", verticalAlign: "top" }}
                />
                {selected?.bboxes?.length > 0 && (
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      pointerEvents: "none",
                    }}
                  >
                    {selected.bboxes.map((box, i) => (
                      <div
                        key={i}
                        style={{
                          position: "absolute",
                          top: `${box.top * 100}%`,
                          left: `${box.left * 100}%`,
                          width: `${box.width * 100}%`,
                          height: `${box.height * 100}%`,
                          border: "2px solid #0066cc",
                          backgroundColor: "rgba(0, 102, 204, 0.15)",
                          boxSizing: "border-box",
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: "#888" }}>No page image in session (run pipeline to generate).</p>
            )}
          </div>
        </div>
      ) : (
        <p>Loading session…</p>
      )}
    </div>
  );
}
