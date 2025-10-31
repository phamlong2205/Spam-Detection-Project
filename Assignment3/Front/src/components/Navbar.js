// src/Navbar.js

import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const loc = useLocation();
  const is = (p) => (loc.pathname === p ? "active" : "");
  return (
    <nav className="nav">
      <div className="brand">Simple Text Tester</div>
      <div className="links">
        <Link className={is("/")} to="/">Home</Link>
        <Link className={is("/test")} to="/test">Test</Link>
        <Link className={is("/about")} to="/about">About</Link>
        {/* New: Dashboard */}
        <Link className={is("/dashboard")} to="/dashboard">Dashboard</Link>
      </div>
    </nav>
  );
}
