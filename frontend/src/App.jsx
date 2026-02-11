import React, { useState } from "react";
import { startSessionStream } from "./api";
import SessionView from "./components/SessionView";

const STAGE_LABELS = {
  render_page: "Render page (PDF → PNG)",
  extract_blocks_lines: "Extract blocks & lines (Tesseract)",
  classify_block_font: "Classify block font (Hebrew / Rashi)",
  split_rashi_lines: "Split Rashi lines by colon",
  rashi_tesseract: "Rashi Tesseract OCR",
  fill_line_text: "Line text (Tesseract)",
  fetch_streams: "Fetch Sefaria streams",
  assign_blocks: "Assign blocks to streams",
  align_segments: "Align segments to lines",
  match_commentary_spans: "Match commentary spans (embeddings)",
  boundary_cuts: "Boundary cuts",
  validate: "Validate",
  pause_for_hitl: "Pause for review",
  persist: "Persist",
};

const STAGE_ORDER = [
  "render_page",
  "extract_blocks_lines",
  "filter_margin_blocks",
  "classify_block_font",
  "split_rashi_lines",
  "rashi_tesseract",
  "fill_line_text",
  "fetch_streams",
  "assign_blocks",
  "align_segments",
  "match_commentary_spans",
  "boundary_cuts",
  "validate",
  "pause_for_hitl",
  "persist",
];

export default function App() {
  const [pdfUrl, setPdfUrl] = useState("");
  const [pageRefs, setPageRefs] = useState([""]);
  const [sessionId, setSessionId] = useState(null);
  const [startResult, setStartResult] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [completedStages, setCompletedStages] = useState([]);
  const [currentStage, setCurrentStage] = useState(null);

  const handleAddPageRef = () => setPageRefs((prev) => [...prev, ""]);
  const handleRemovePageRef = (index) =>
    setPageRefs((prev) => prev.filter((_, i) => i !== index));
  const handlePageRefChange = (index, value) => {
    setPageRefs((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  };

  function onStart(e) {
    e?.preventDefault();
    setErr("");
    const url = (pdfUrl || "").trim();
    const trimmedRefs = pageRefs.map((r) => r.trim()).filter((r) => r.length > 0);

    if (!url) {
      setErr("Please provide a PDF URL.");
      return;
    }
    if (trimmedRefs.length === 0) {
      setErr("Please provide at least one page reference (e.g., Berakhot 2a).");
      return;
    }

    setLoading(true);
    setCompletedStages([]);
    setCurrentStage(STAGE_ORDER[0] ?? null);

    startSessionStream(
      { pdf_url: url, page_refs: trimmedRefs },
      {
        onStage: (stageName) => {
          setCompletedStages((prev) =>
            prev.includes(stageName) ? prev : [...prev, stageName]
          );
          const idx = STAGE_ORDER.indexOf(stageName);
          setCurrentStage(idx >= 0 && idx < STAGE_ORDER.length - 1 ? STAGE_ORDER[idx + 1] : null);
        },
        onComplete: (result) => {
          setLoading(false);
          setCurrentStage(null);
          setStartResult(result);
          setSessionId(result.session_id);
        },
        onError: (message) => {
          setLoading(false);
          setCompletedStages([]);
          setCurrentStage(null);
          setErr(message);
        },
      }
    );
  }

  if (sessionId) {
    return <SessionView sessionId={sessionId} startResult={startResult} />;
  }

  return (
    <div style={{ fontFamily: "sans-serif", padding: 16, maxWidth: 900 }}>
      <h2>Tagger Beta (block-first, segment-aligned)</h2>

      <form onSubmit={onStart} style={{ display: "grid", gap: 12 }}>
        <div>
          <label style={{ display: "block", marginBottom: 4 }}>
            Page refs (Page.refs)
          </label>
          {pageRefs.map((ref, idx) => (
            <div key={idx} style={{ display: "flex", gap: 8, marginBottom: 6 }}>
              <input
                type="text"
                style={{ flex: 1, boxSizing: "border-box" }}
                value={ref}
                onChange={(e) => handlePageRefChange(idx, e.target.value)}
                placeholder="For example: Berakhot 2a"
              />
              {pageRefs.length > 1 && (
                <button
                  type="button"
                  onClick={() => handleRemovePageRef(idx)}
                  aria-label="Remove"
                >
                  ✕
                </button>
              )}
            </div>
          ))}
          <button type="button" onClick={handleAddPageRef}>
            + Add Page Ref
          </button>
        </div>

        <div>
          <label style={{ display: "block", marginBottom: 4 }}>
            PDF URL (Page.source_pdf)
          </label>
          <input
            type="text"
            style={{ width: "100%", boxSizing: "border-box" }}
            value={pdfUrl}
            onChange={(e) => setPdfUrl(e.target.value)}
            placeholder="https://..."
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Running pipeline…" : "Load Page Data"}
        </button>
        {err ? <pre style={{ color: "crimson" }}>{err}</pre> : null}

        {loading && (
          <div style={{ marginTop: 16, padding: 12, background: "#f5f5f5", borderRadius: 8 }}>
            <h4 style={{ margin: "0 0 8px 0" }}>Pipeline stages</h4>
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {STAGE_ORDER.map((key) => {
                const done = completedStages.includes(key);
                const running = currentStage === key;
                const label = STAGE_LABELS[key] ?? key;
                return (
                  <li
                    key={key}
                    style={{
                      padding: "4px 0",
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      opacity: done ? 1 : running ? 1 : 0.6,
                    }}
                  >
                    <span style={{ width: 24, textAlign: "center" }}>
                      {done ? "✓" : running ? "…" : "○"}
                    </span>
                    <span style={{ fontWeight: running ? 600 : 400 }}>{label}</span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </form>

      <p style={{ marginTop: 12, color: "#555", fontSize: 14 }}>
        Same input as tzuratlink-data: paste a PDF link and one or more page refs (e.g. Berakhot 2a).
        The first page of the PDF and the first page ref are used for tagging.
      </p>
    </div>
  );
}
