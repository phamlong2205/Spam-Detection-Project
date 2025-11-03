// App.js
import { useEffect, useMemo, useState } from "react";
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Test from "./pages/Test";
import About from "./pages/About";
import Header from "./components/Header";
import Footer from "./components/Footer";
import "./App.css";

function getInitialTheme() {
  // 1) explicit stored choice wins
  const saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light") return saved;
  // 2) otherwise follow OS preference
  const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
  return mq && mq.matches ? "dark" : "light";
}

export default function App() {
  const [theme, setTheme] = useState(getInitialTheme);

  // keep <html data-theme="..."> in sync
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  // if user never chose and OS theme changes, you could listen & update:
  useEffect(() => {
    if (localStorage.getItem("theme")) return; // user has a choice, don't override
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = (e) => setTheme(e.matches ? "dark" : "light");
    if (mq && mq.addEventListener) {
      mq.addEventListener("change", onChange);
      return () => mq.removeEventListener("change", onChange);
    }
  }, []);

  const toggleTheme = useMemo(
    () => () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    []
  );

  return (
    <>
      <Header theme={theme} toggleTheme={toggleTheme} />
      <main className="wrap" style={{ paddingTop: 16, paddingBottom: 24 }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/test" element={<Test />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}
