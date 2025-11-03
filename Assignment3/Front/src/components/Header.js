// src/components/Header.js
import { NavLink } from "react-router-dom";

export default function Header({ theme = "light", toggleTheme = () => {} }) {
  const isDark = theme === "dark";

  return (
    <header className="site-header">
      <div className="wrap header-inner">
        <div className="brand">AI4Cyber Spam Detector</div>

        <nav className="links" style={{ alignItems: "center", gap: 10 }}>
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/test">Test</NavLink>
          <NavLink to="/about">About</NavLink>

          {/* Theme toggle */}
          <button
            type="button"
            onClick={toggleTheme}
            className="btn-mini outline"
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            style={{ marginLeft: 6 }}
          >
            {isDark ? "☀ Day" : "☽ Night"}
          </button>
        </nav>
      </div>
    </header>
  );
}
