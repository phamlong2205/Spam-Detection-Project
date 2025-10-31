// src/pages/Dashboard.js — Beginner-friendly, guided dashboard
import React, { useEffect, useMemo, useState } from "react";
import BarChart from "../components/BarChart";
import LineChart from "../components/LineChart";

const API =
  process.env.REACT_APP_API ||
  (window.location.hostname.endsWith("github.io")
    ? "https://YOUR-API-URL"
    : "http://localhost:8000");

// Safe number helper so we never show NaN to users
const n = (x, d = 0) => (Number.isFinite(Number(x)) ? Number(x) : d);

export default function Dashboard() {
  const [stats, setStats] = useState({
    total: 0,
    spam: 0,
    ham: 0,
    rate: 0,
    last_ts: null,
  });
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  async function loadAll() {
    try {
      setErr("");
      const [s, h] = await Promise.all([
        fetch(`${API}/stats`).then((r) => r.json()),
        fetch(`${API}/history?limit=200`).then((r) => r.json()),
      ]);
      // Patch values to avoid NaN
      const total = n(s?.total);
      const spam = n(s?.spam);
      const ham = Math.max(0, total - spam);
      const rate = total > 0 ? spam / total : 0;

      setStats({
        total,
        spam,
        ham,
        rate,
        last_ts: s?.last_ts ?? null,
      });
      setHistory(h?.items || []);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
    const id = setInterval(loadAll, 10000); // auto-refresh every 10s
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Build RF confidence series from history.json (display_conf already normalized 0..1)
  const rfSeries = useMemo(
    () =>
      history
        .filter((h) => typeof h.display_conf === "number")
        .map((h) => ({ t: h.ts || Date.now(), c: h.display_conf })),
    [history]
  );
  // If later you store SVM in history, map it the same way:
  const svmSeries = useMemo(() => [], []);

  // Human-readable helper for “1 in N” text
  const oneInN = (() => {
    const r = stats.rate;
    if (!r || r <= 0) return "rare in this sample";
    if (r >= 1) return "almost all messages";
    return `about 1 in ${Math.round(1 / r)}`;
  })();

  return (
    <div className="page">
      {/* ----- Intro / Overview ----- */}
      <div className="intro" style={{ marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>📊 Dashboard Overview</h1>
        <p style={{ fontSize: 15, color: "#475569", maxWidth: 720, marginTop: 8 }}>
          This page explains how the spam detector has been performing, based on recent
          messages tested. New to this? Start with the “What am I looking at?” box below.
        </p>

        <details
          style={{
            marginTop: 12,
            background: "#f8fafc",
            padding: 12,
            borderRadius: 8,
            border: "1px solid #e2e8f0",
          }}
        >
          <summary style={{ fontWeight: 700, cursor: "pointer" }}>
            💡 What am I looking at?
          </summary>
          <ul style={{ marginTop: 8, color: "#475569", fontSize: 15, lineHeight: 1.7 }}>
            <li>
              <b>Total predictions</b> — how many messages were checked by the model.
            </li>
            <li>
              <b>Spam</b> vs <b>Ham</b> — “Spam” means suspicious or unwanted. “Ham” means
              safe/legitimate.
            </li>
            <li>
              <b>Spam rate</b> — the share of all checked messages that were spam.
            </li>
            <li>
              <b>Confidence over time</b> — how sure the model was for each result (higher is
              more certain).
            </li>
            <li>
              <b>Recent predictions</b> — a short feed of the latest checks with their results.
            </li>
          </ul>
        </details>
      </div>

      {err && (
        <div className="error" style={{ marginBottom: 12 }}>
          ⚠ {err}
        </div>
      )}

      {/* ----- KPI Cards ----- */}
      <div
        className="kpis"
        style={{
          display: "grid",
          gap: 12,
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        }}
      >
        <div className="card" title="How many messages have been analysed so far?">
          <div className="muted">Total predictions</div>
          <div style={{ fontSize: 28, fontWeight: 800 }}>{stats.total}</div>
          <div className="small muted">All messages checked by the model.</div>
        </div>

        <div className="card" title="How many messages were marked as spam?">
          <div className="muted">Spam</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#b91c1c" }}>
            {stats.spam}
          </div>
          <div className="small muted">🟥 Suspicious / unwanted messages.</div>
        </div>

        <div className="card" title="How many messages were marked as safe?">
          <div className="muted">Ham</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#15803d" }}>
            {stats.ham}
          </div>
          <div className="small muted">🟩 Safe / legitimate messages.</div>
        </div>

        <div className="card" title="Share of all checked messages that were spam.">
          <div className="muted">Spam rate</div>
          <div style={{ fontSize: 28, fontWeight: 800 }}>
            {(n(stats.rate) * 100).toFixed(1)}%
          </div>
          <div className="small muted">
            That’s {oneInN} messages being spam in this sample.
          </div>
        </div>
      </div>

      {/* ----- Charts ----- */}
      <div
        className="grid"
        style={{
          marginTop: 16,
          display: "grid",
          gap: 16,
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        }}
      >
        <div className="card">
          <h3 style={{ marginTop: 0 }}>
            Messages classified as Spam or Safe{" "}
            <span
              title="The bar shows how many messages were spam (red) vs safe (green)."
              style={{ cursor: "help", color: "#64748b", marginLeft: 6 }}
            >
              ?
            </span>
          </h3>
          {loading ? (
            <div className="muted">Loading…</div>
          ) : (
            <BarChart spam={stats.spam} ham={stats.ham} />
          )}
          <div className="small muted" style={{ marginTop: 8 }}>
            🟥 = Spam, 🟩 = Safe (“ham”)
          </div>
        </div>

        <div className="card">
          <h3 style={{ marginTop: 0 }}>
            How confident the model feels over time{" "}
            <span
              title="Each point shows how sure the model was for a result. Closer to 1.0 means more certain."
              style={{ cursor: "help", color: "#64748b", marginLeft: 6 }}
            >
              ?
            </span>
          </h3>
          <div style={{ height: 320 }}>
            {loading ? (
              <div className="muted">Loading…</div>
            ) : (
              <LineChart rf={rfSeries} svm={svmSeries} />
            )}
          </div>
          <div className="small muted" style={{ marginTop: 8 }}>
            Tip: scroll to zoom, drag to pan. 🟦 = Confidence level (0–1).
          </div>
        </div>
      </div>

      {/* ----- Friendly summary (optional narrative) ----- */}
      <div
        className="card"
        style={{
          marginTop: 16,
          background: "#ecfdf5",
          border: "1px solid #d1fae5",
        }}
      >
        <h3 style={{ marginTop: 0, color: "#065f46" }}>Model Summary</h3>
        <p style={{ fontSize: 15, color: "#065f46", marginBottom: 8 }}>
          Based on <b>{stats.total}</b> checked messages, the detector marked{" "}
          <b>{stats.spam}</b> as spam and <b>{stats.ham}</b> as safe. The current spam
          rate is <b>{(n(stats.rate) * 100).toFixed(1)}%</b>, which means {oneInN} of
          messages are spam in this sample. Confidence trends (right) show how sure the
          model has been for recent results.
        </p>
      </div>

      {/* ----- Recent predictions table ----- */}
      <div className="card" style={{ marginTop: 16 }}>
        <div
          style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
        >
          <h3 style={{ marginTop: 0 }}>Recent predictions</h3>
          <button className="ghost" onClick={loadAll} disabled={loading} title="Reload data">
            Refresh
          </button>
        </div>
        {!history.length ? (
          <div className="muted">
            No history yet. Go to the <b>Test</b> page and make a prediction.
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "8px 6px" }}>Time (UTC)</th>
                  <th style={{ padding: "8px 6px" }}>Preview</th>
                  <th style={{ padding: "8px 6px" }}>Result</th>
                  <th style={{ padding: "8px 6px" }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {[...history].reverse().map((h, i) => {
                  const dt = h.ts
                    ? new Date(h.ts).toISOString().split("T")[1].slice(0, 8)
                    : "—";
                  const preview =
                    (h.text || "").length > 80 ? h.text.slice(0, 80) + "…" : h.text || "";
                  const conf =
                    typeof h.display_conf === "number"
                      ? (h.display_conf * 100).toFixed(1) + "%"
                      : "—";
                  return (
                    <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
                      <td style={{ padding: "8px 6px" }}>{dt}</td>
                      <td style={{ padding: "8px 6px", color: "var(--sub)" }} title={h.text}>
                        {preview}
                      </td>
                      <td style={{ padding: "8px 6px" }}>
                        <span className={`verdict-badge ${h.label_num === 1 ? "spam" : "ham"}`}>
                          {h.label_num === 1 ? "Spam" : "Safe"}
                        </span>
                      </td>
                      <td style={{ padding: "8px 6px", fontWeight: 700 }}>{conf}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
