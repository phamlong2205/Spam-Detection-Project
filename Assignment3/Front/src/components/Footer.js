// src/components/Footer.js
export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="site-footer">
      <div className="wrap footer-grid">
        {/* Brand / tagline */}
        <div className="f-col">
          <div className="f-brand">AI4Cyber Spam Detector</div>
          <div className="f-muted">Session 18 · Group 06 · Swinburne</div>
          <div className="f-note">Open-source demo to help users spot spam & phishing.</div>
        </div>

        {/* Spacer (keeps the three-column balance) */}
        <div className="f-col f-spacer" />

        {/* Trusted By (moved to the right column) */}
        <div className="f-col f-right">
          <div className="f-title">Trusted By</div>
          <div className="f-note">🎓 Swinburne University</div>
          <div className="f-note">🔒 Privacy First</div>
          <div className="f-note">⚡ 95% Accuracy</div>
        </div>
      </div>

      <div className="footer-bottom">
        <div className="wrap tiny">© {year} AI4Cyber · Built for COS30049</div>
      </div>
    </footer>
  );
}
