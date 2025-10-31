// src/components/Header.js
import { NavLink } from "react-router-dom";

export default function Header() {
  return (
    <header className="site-header">
      <div className="wrap header-inner">
        <div className="brand">
          AI4Cyber Spam Detector
        </div>

        <nav className="links">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/test">Test</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </div>
    </header>
  );
}
