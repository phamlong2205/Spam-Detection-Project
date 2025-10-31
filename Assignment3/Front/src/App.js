import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Test from "./pages/Test";
import About from "./pages/About";
import Header from "./components/Header";
import Footer from "./components/Footer";
import "./App.css";

export default function App() {
  return (
    <>
      <Header />
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
