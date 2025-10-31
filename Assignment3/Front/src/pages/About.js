// src/pages/About.js — RF training overview + validation charts
import { useEffect } from "react";

const img = (name) => `${process.env.PUBLIC_URL}/assets/${name}`;

export default function About() {
  useEffect(() => { window.scrollTo(0, 0); }, []);

  return (
    <div className="card">
      <h2 style={{ 
        marginTop: 0, 
        fontSize: '2rem', 
        fontWeight: 600, 
        marginBottom: '1rem',
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        About the Model
      </h2>

      <p style={{ 
        fontSize: '1.1rem', 
        lineHeight: 1.8, 
        marginBottom: '2.5rem', 
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        This app deploys a Random Forest classifier 
        to detect spam vs ham messages. 
        Below is an overview of our training pipeline and validation charts demonstrating model performance.
      </p>

      <h3 style={{ 
        fontSize: '2rem', 
        fontWeight: 600, 
        marginBottom: '1.5rem', 
        marginTop: '2.5rem',
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        Training Pipeline
      </h3>

      <div className="pipeline" style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1.5rem',
        marginBottom: '2rem'
      }}>
        <div className="step">
          <div className="step-num">1</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Collect & Clean
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Merge SMS and Email datasets into{' '}
            <code style={{ 
              padding: '2px 8px', 
              background: '#f3f4f6', 
              borderRadius: '4px',
              fontSize: '0.9rem',
              fontFamily: 'monospace'
            }}>
              (label, message)
            </code>{' '}
            format. Apply text normalisation including lowercasing, HTML/URL removal, and whitespace standardisation.
          </div>
        </div>
        
        <div className="step">
          <div className="step-num">2</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Feature Engineering
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Extract TF-IDF features with 1-2 grams, combined with numeric signals: message length, 
            digit and capital ratios, special character counts, average word length, and URL frequency.
          </div>
        </div>
        
        <div className="step">
          <div className="step-num">3</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Split & Balance
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Create stratified Train/Validation/Test splits. Apply SMOTE balancing to the training set only 
            when needed to address class imbalance while keeping validation and test sets pristine.
          </div>
        </div>
        
        <div className="step">
          <div className="step-num">4</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', color: '#1a1a1a' }}>
            Train & Select
          </div>
          <div style={{ fontSize: '0.95rem', lineHeight: 1.7, color: '#4a4a4a' }}>
            Tune Random Forest hyperparameters including tree depth and estimator count. 
            Use SelectKBest to retain the strongest features. Evaluate final metrics on the held-out test set.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '2rem', padding: '1.5rem' }}>
        <h3 style={{ marginTop: 0, fontSize: '2rem', fontWeight: 600, marginBottom: '1rem', color: '#1a1a1a', textAlign: 'center' }}>
          Model Card
        </h3>
        <ul style={{ lineHeight: 2, fontSize: '0.95rem', color: '#4a4a4a', paddingLeft: '1.25rem' }}>
          <li><strong>Algorithm:</strong> Random Forest Classifier</li>
          <li><strong>Vectoriser:</strong> TF-IDF with 1-2 grams plus numeric features</li>
          <li><strong>Primary metric:</strong> F1 Score (also tracking accuracy and train-validation gap)</li>
          <li><strong>Serving threshold:</strong> 0.50 probability for spam classification</li>
          <li><strong>Why Random Forest:</strong> Fast inference, robust on sparse text data, low overfitting, highly interpretable</li>
          <li><strong>Limitations:</strong> May struggle with very short or heavily obfuscated texts; out-of-distribution inputs can compress probability scores</li>
        </ul>
      </div>

      {/* Two trust-building charts from public/assets */}
      <h3 style={{ 
        fontSize: '2rem', 
        fontWeight: 600, 
        marginBottom: '1.5rem', 
        marginTop: '2.5rem',
        color: '#1a1a1a',
        textAlign: 'center'
      }}>
        Training Validation Charts
      </h3>

      <div className="viz-grid two-cols" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
        <figure className="viz" style={{ margin: 0 }}>
          <img 
            src={img("rf_training_loss.png")} 
            alt="Random Forest Training & Validation Loss" 
            style={{ width: '100%', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
          />
          <figcaption style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.75rem', textAlign: 'center', lineHeight: 1.6 }}>
            Loss decreases steadily for both training and validation sets, indicating stable learning without overfitting.
          </figcaption>
        </figure>

        <figure className="viz" style={{ margin: 0 }}>
          <img 
            src={img("rf_training_accuracy.png")} 
            alt="Random Forest Training & Validation Accuracy" 
            style={{ width: '100%', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
          />
          <figcaption style={{ fontSize: '0.9rem', color: '#6b7280', marginTop: '0.75rem', textAlign: 'center', lineHeight: 1.6 }}>
            Accuracy rises together with minimal gap between training and validation, demonstrating excellent generalisation.
          </figcaption>
        </figure>
      </div>

      <div className="card" style={{ marginTop: '2rem', padding: '1.5rem' }}>
        <h3 style={{ marginTop: 0, fontSize: '2rem', fontWeight: 600, marginBottom: '1rem', color: '#1a1a1a', textAlign: 'center' }}>
          Transparency & Reproducibility
        </h3>
        <p style={{ fontSize: '0.95rem', lineHeight: 1.8, color: '#1a1a1a', margin: 0, textAlign: 'center' }}>
          TF-IDF, selector (if used), and the trained RF model are saved as artefacts. The API loads them 
          to return a probability + label instantly; inputs are stored only when you click Predict (so 
          you can see history and export CSV).
        </p>
      </div>
    </div>
  );
}