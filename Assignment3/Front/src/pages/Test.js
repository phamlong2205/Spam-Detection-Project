// src/pages/Test.js — centered title, friendlier errors, same history/UX
import React, { useEffect, useState } from "react";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";
ChartJS.register(ArcElement, Tooltip, Legend);

const API =
  process.env.REACT_APP_API ||
  (window.location.hostname.endsWith("github.io")
    ? "https://YOUR-API-URL"
    : "http://localhost:8000");

const MIN_LEN = 3;
const MAX_LEN = 1000;

function doughnutData(prob) {
  const p = typeof prob === "number" ? Math.max(0, Math.min(1, prob)) : 0;
  return {
    labels: [
      `Spam (${Math.round(p * 100)}%)`,
      `Ham (${Math.round((1 - p) * 100)}%)`,
    ],
    datasets: [
      {
        label: "Random Forest",
        data: [p, 1 - p],
        backgroundColor: ["rgba(220,38,38,.85)", "rgba(22,163,74,.85)"],
        borderColor: ["#dc2626", "#16a34a"],
        borderWidth: 2,
      },
    ],
  };
}

export default function Test() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [rf, setRF] = useState(null);
  const [history, setHistory] = useState([]);
  const [limit, setLimit] = useState(30);

  // initial history
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API}/history?limit=${limit}`);
        const j = await r.json();
        setHistory(j.items || []);
      } catch {
        // ignore: user-friendly message will show on submit if backend is down
      }
    })();
  }, [limit]);

  const trimmed = text.trim();
  const isValid = trimmed.length >= MIN_LEN && trimmed.length <= MAX_LEN;

  function friendlyError(e) {
    const raw = String(e || "");
    if (/Failed to fetch|NetworkError|TypeError/i.test(raw)) {
      return "Can’t reach the server. Make sure the backend is running at http://localhost:8000 and try again.";
    }
    return raw || "Something went wrong. Please try again.";
    }

  async function submit(e) {
    e.preventDefault();
    if (!isValid) return;

    setLoading(true);
    setErr("");
    setRF(null);
    try {
      const res = await fetch(`${API}/predict/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, model: "rf" }),
      });
      if (!res.ok) {
        let msg = "";
        try {
          msg = await res.text();
        } catch {}
        throw new Error(msg || `Server error (${res.status})`);
      }
      const json = await res.json(); // {label_num, prob, ...}
      const display_conf =
        typeof json.prob === "number"
          ? json.label_num === 1
            ? json.prob
            : 1 - json.prob
          : null;

      const record = {
        ts: new Date().toISOString(),
        model: "rf",
        text: trimmed,
        prob: json.prob ?? null,
        display_conf,
        label_num: json.label_num,
      };

      setRF(record);
      setHistory((h) => [...h.slice(-(limit - 1)), record]);
    } catch (e) {
      setErr(friendlyError(e));
    } finally {
      setLoading(false);
    }
  }

  async function refreshHistory() {
    try {
      const r = await fetch(`${API}/history?limit=${limit}`);
      const j = await r.json();
      setHistory(j.items || []);
    } catch (e) {
      setErr(friendlyError(e));
    }
  }

  function exportCSV() {
    window.open(`${API}/export/predictions?format=csv`, "_blank");
  }

  // client-side delete/clear (local only; backend history remains)
  function deleteRow(idxFromTop) {
    const arr = [...history].reverse();
    arr.splice(idxFromTop, 1);
    setHistory(arr.reverse());
  }
  function clearLocal() {
    setHistory([]);
  }

  return (
    <>
      {/* Centered, cleaner title */}
      <div className="card">
        <h2 style={{ margin: 0, textAlign: "center" }}>
          Spam Detector — Message Tester
        </h2>

        <form onSubmit={submit} style={{ marginTop: 12 }}>
          <label>Enter text to classify</label>
          <textarea
            rows={6}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type a message… (e.g., WIN a FREE iPhone!!! click http://deal.com)"
            maxLength={MAX_LEN}
          />
          <div className="small" style={{ marginTop: 6 }}>
            {trimmed.length}/{MAX_LEN} characters
          </div>
          <ul
            className="small"
            style={{ color: "#b91c1c", marginTop: 6, marginBottom: 0 }}
          >
            {!trimmed && <li>Please enter some text.</li>}
            {trimmed && trimmed.length < MIN_LEN && (
              <li>Message is too short (min {MIN_LEN} chars).</li>
            )}
            {trimmed.length > MAX_LEN && (
              <li>Message is too long (max {MAX_LEN} chars).</li>
            )}
          </ul>

          <div className="row">
            <button disabled={!isValid || loading}>
              {loading ? "Predicting…" : "Predict (RF)"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setText("");
                setRF(null);
                setErr("");
              }}
            >
              Clear text
            </button>
          </div>
        </form>

        {err && <div className="error" style={{ marginTop: 12 }}>⚠ {err}</div>}
      </div>

      {rf && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Result — Random Forest</h3>
          <div className="verdict-row" style={{ marginBottom: 6 }}>
            <span
              className={`verdict-badge ${rf.label_num === 1 ? "spam" : "ham"}`}
            >
              {rf.label_num === 1 ? "Spam" : "Ham"}
            </span>
            {typeof rf.display_conf === "number" && (
              <span className="confidence">
                Confidence: {(rf.display_conf * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <div style={{ height: 300 }}>
            <Doughnut
              data={doughnutData(rf.prob ?? 0)}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: "bottom" } },
              }}
            />
          </div>

          <details style={{ marginTop: 10 }}>
            <summary>Raw JSON</summary>
            <pre className="json">
              {JSON.stringify(
                {
                  model: "rf",
                  label_num: rf.label_num,
                  prob: rf.prob,
                  confidence: rf.display_conf,
                },
                null,
                2
              )}
            </pre>
          </details>
        </div>
      )}

      {/* History remains unchanged */}
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: 8,
            alignItems: "center",
          }}
        >
          <h3 style={{ margin: 0 }}>History (saved on backend)</h3>
          <div className="row" style={{ gap: 8 }}>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="ghost"
              style={{ padding: "8px 10px" }}
              aria-label="History size"
            >
              {[10, 20, 30, 50, 100].map((n) => (
                <option key={n} value={n}>
                  Last {n}
                </option>
              ))}
            </select>
            <button className="ghost" onClick={refreshHistory} type="button">
              Refresh
            </button>
            <button className="ghost" onClick={exportCSV} type="button">
              Export CSV
            </button>
            <button className="ghost" onClick={clearLocal} type="button">
              Clear (local)
            </button>
          </div>
        </div>

        <div className="small" style={{ marginTop: 6, color: "#64748b" }}>
          Showing {history.length} item{history.length !== 1 ? "s" : ""}
        </div>

        <div style={{ marginTop: 10, overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ textAlign: "left" }}>
                <th style={{ padding: "8px 6px" }}>Time (UTC)</th>
                <th style={{ padding: "8px 6px" }}>Preview</th>
                <th style={{ padding: "8px 6px" }}>RF</th>
                <th style={{ padding: "8px 6px" }}>Confidence</th>
                <th style={{ padding: "8px 6px" }}>P(spam)</th>
                <th style={{ padding: "8px 6px" }}></th>
              </tr>
            </thead>
            <tbody>
              {[...history].reverse().map((h, i) => {
                const ts = h.ts || new Date().toISOString();
                const preview =
                  (h.text || "").length > 80
                    ? (h.text || "").slice(0, 80) + "…"
                    : h.text || "";
                const conf =
                  typeof h.display_conf === "number"
                    ? (h.display_conf * 100).toFixed(1) + "%"
                    : "—";
                const pspam =
                  typeof h.prob === "number" ? h.prob.toFixed(4) : "—";

                return (
                  <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                    <td style={{ padding: "8px 6px" }}>{ts}</td>
                    <td
                      style={{ padding: "8px 6px", color: "var(--sub)" }}
                      title={h.text}
                    >
                      {preview}
                    </td>
                    <td style={{ padding: "8px 6px" }}>
                      <span
                        className={`verdict-badge ${
                          h.label_num === 1 ? "spam" : "ham"
                        }`}
                      >
                        {h.label_num === 1 ? "Spam" : "Ham"}
                      </span>
                    </td>
                    <td style={{ padding: "8px 6px", fontWeight: 700 }}>
                      {conf}
                    </td>
                    <td style={{ padding: "8px 6px", fontFamily: "monospace" }}>
                      {pspam}
                    </td>
                    <td style={{ padding: "8px 6px" }}>
                      <button
                        type="button"
                        className="btn-ghost"
                        onClick={() => deleteRow(i)}
                        title="Remove this row locally"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
