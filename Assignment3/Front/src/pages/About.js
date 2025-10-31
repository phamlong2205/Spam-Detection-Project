// src/pages/About.js — RF training overview + 2 trust charts (images from public/assets)
import { useEffect } from "react";

const img = (name) => `${process.env.PUBLIC_URL}/assets/${name}`;

export default function About() {
  useEffect(() => { window.scrollTo(0, 0); }, []);

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>About the Model</h2>

      <p className="lead">
        This app deploys a <b>Random Forest</b> to detect <b>spam</b> vs <b>ham</b>. Below is a quick
        overview of our training pipeline and two training charts to build trust.
      </p>

      <div className="pipeline">
        <div className="step">
          <div className="step-num">1</div>
          <div className="step-title">Collect & Clean</div>
          <div className="step-body small">
            Merge SMS + Email → <code>(label, message)</code>; lowercase, strip HTML/URLs, normalise spaces.
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="step">
          <div className="step-num">2</div>
          <div className="step-title">Feature Engineering</div>
          <div className="step-body small">
            TF-IDF (1–2 grams) + numeric signals (length, digit/capital ratios, special chars, avg word length, URL count).
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="step">
          <div className="step-num">3</div>
          <div className="step-title">Split & Balance</div>
          <div className="step-body small">
            Stratified Train/Val/Test; balance <i>train only</i> when needed (e.g., SMOTE).
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="step">
          <div className="step-num">4</div>
          <div className="step-title">Train & Select</div>
          <div className="step-body small">
            Tune RF depth/trees; SelectKBest to keep strongest features; final metrics on held-out test.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <h3 style={{ marginTop: 0 }}>Model Card (Short)</h3>
        <ul className="small" style={{ lineHeight: 1.7 }}>
          <li><b>Algorithm:</b> Random Forest</li>
          <li><b>Vectoriser:</b> TF-IDF (1–2 grams) + numeric features</li>
          <li><b>Primary metric:</b> F1; we also track accuracy and train–val gap</li>
          <li><b>Serving threshold:</b> 0.50 on P(spam)</li>
          <li><b>Why RF:</b> fast inference, robust on sparse text, low overfit gap, easy to explain</li>
          <li><b>Limits:</b> very short/unseen obfuscated texts; OOD inputs compress probabilities</li>
        </ul>
      </div>

      {/* Two trust-building charts from public/assets */}
      <div className="viz-grid two-cols">
        <figure className="viz">
          <img src={img("rf_training_loss.png")} alt="Random Forest Training & Validation Loss" />
          <figcaption>Loss for both train and validation decreases steadily → stable learning.</figcaption>
        </figure>

        <figure className="viz">
          <img src={img("rf_training_accuracy.png")} alt="Random Forest Training & Validation Accuracy" />
          <figcaption>Accuracy rises together with a small gap → good generalisation (low overfitting).</figcaption>
        </figure>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <h3 style={{ marginTop: 0 }}>Transparency & Reproducibility</h3>
        <p className="small">
          TF-IDF, selector (if used), and the trained RF model are saved as artefacts. The API loads them
          to return a probability + label instantly; inputs are stored only when you click Predict (so
          you can see history and export CSV).
        </p>
      </div>
    </div>
  );
}
