// src/pages/Home.js — centered title, plain text, black "How it Works", CTAs at bottom
import React, { useEffect } from "react";
import { Link } from "react-router-dom";

export default function Home() {
  useEffect(() => window.scrollTo(0, 0), []);

  return (
    <div className="card hero">
      {/* Centered title, no extra tagline */}
      <h1 className="hero-title" style={{ textAlign: "center", marginBottom: 8 }}>
        AI4Cyber — Spam Detector
      </h1>

      {/* Plain paragraph (no bold/strong) */}
      <p className="lead" style={{ textAlign: "center" }}>
        A lightweight machine learning project that flags spam vs. ham in everyday messages.
        Built with React (frontend) and FastAPI (backend), and powered by a Random Forest
        model trained on mixed SMS + email data.
      </p>

      {/* Quick highlights (kept minimal, plain text) */}
      <div className="row" style={{ marginTop: 16, gap: 12, flexWrap: "wrap" }}>
        <div className="card" style={{ flex: "1 1 260px", minWidth: 240 }}>
          <h3 style={{ marginTop: 0 }}>Fast & Simple</h3>
          <p className="small">
            Paste a message and get an instant verdict with a confidence score.
          </p>
        </div>
        <div className="card" style={{ flex: "1 1 260px", minWidth: 240 }}>
          <h3 style={{ marginTop: 0 }}>Clean Pipeline</h3>
          <p className="small">
            Normalised datasets, TF-IDF features plus simple numeric signals.
          </p>
        </div>
        <div className="card" style={{ flex: "1 1 260px", minWidth: 240 }}>
          <h3 style={{ marginTop: 0 }}>Transparent UI</h3>
          <p className="small">
            See your prediction, browse recent history, and export CSV on the Test page.
          </p>
        </div>
      </div>

      {/* How it Works — all black text, no emphasis */}
      <div className="card" style={{ marginTop: 16, color: "#000" }}>
        <h3 style={{ marginTop: 0, color: "#000" }}>How it Works</h3>
        <ol className="small" style={{ paddingLeft: 18, lineHeight: 1.7 }}>
          <li>Preprocess — Clean text and engineer simple numeric features.</li>
          <li>Vectorise — TF-IDF + selected features form the model input.</li>
          <li>Predict — FastAPI serves a Random Forest probability and label.</li>
        </ol>
      </div>

      {/* Bottom CTAs */}
      <div
        className="cta"
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          justifyContent: "center",
          marginTop: 20,
        }}
      >
        <Link to="/test" className="btn-primary">Try the Tester →</Link>
        <Link to="/about" className="btn-ghost">Learn More</Link>
      </div>
    </div>
  );
}
