# main.py — FastAPI spam/ham classifier (Random Forest only) with history
# Run: uvicorn main:app --reload --port 8000
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import os, json, threading, io, csv
import joblib
import numpy as np
import scipy.sparse as sp

# ===================== App & CORS =====================
app = FastAPI(title="Spam Detection API", version="3.0.0")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== History Persistence =====================
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "history.json")
_history_lock = threading.Lock()

def _read_history() -> List[Dict[str, Any]]:
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_history(items: List[Dict[str, Any]]) -> None:
    with _history_lock:
        try:
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Error writing history: {e}")

HISTORY_MAX = 200
HISTORY: List[Dict[str, Any]] = _read_history()

# ===================== Model (RF only) =====================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
SEL_PATH = os.path.join(MODEL_DIR, "feature_selector.joblib")   # optional
RF_PATH  = os.path.join(MODEL_DIR, "random_forest_model.joblib")

THRESHOLD = 0.50

def _load_vectorizer_selector():
    try:
        vect = joblib.load(VEC_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorizer: {e}")
    sel = None
    if os.path.exists(SEL_PATH):
        try:
            sel = joblib.load(SEL_PATH)
        except Exception as e:
            print(f"[WARN] Could not load selector: {e}")
    return vect, sel

def _load_rf():
    if not os.path.exists(RF_PATH):
        raise RuntimeError("Random Forest model file not found in /model")
    try:
        model = joblib.load(RF_PATH)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load RF model: {e}")

VECT, SEL = _load_vectorizer_selector()
RF_MODEL = _load_rf()

# ===================== Helpers =====================
def _get_spam_class_index(model) -> int:
    """Index of the 'spam' class in model.classes_ (binary)."""
    if hasattr(model, "classes_"):
        classes = [str(c).lower().strip() for c in model.classes_]
        # Prefer explicit 'spam' if present
        if "spam" in classes:
            return classes.index("spam")
        # Otherwise assume the positive class is index 1
        return 1
    return 1

def _apply_vectorizer_selector(texts: List[str]):
    X = VECT.transform(texts)
    if SEL is not None:
        X = SEL.transform(X)
    return X

def _model_expected_dim(model) -> Optional[int]:
    m = int(getattr(model, "n_features_in_", 0) or 0)
    return m or None

def _pad_or_truncate(X, expected: Optional[int]):
    if expected is None:
        return X, False
    have = X.shape[1]
    if have == expected:
        return X, False
    if have < expected:
        pad = sp.csr_matrix((X.shape[0], expected - have), dtype=X.dtype)
        return sp.hstack([X, pad], format="csr"), True
    return X[:, :expected], True

def _predict_rf(text: str) -> Dict[str, Any]:
    X = _apply_vectorizer_selector([text])
    X_fix, changed = _pad_or_truncate(X, _model_expected_dim(RF_MODEL))

    if hasattr(RF_MODEL, "predict_proba") and hasattr(RF_MODEL, "classes_"):
        spam_idx = _get_spam_class_index(RF_MODEL)
        proba = RF_MODEL.predict_proba(X_fix)[0]
        p_spam = float(proba[spam_idx])
        is_spam = p_spam >= THRESHOLD
        pred_idx = spam_idx if is_spam else (1 - spam_idx)
        label_text = str(RF_MODEL.classes_[pred_idx])
        return {
            "model": "rf",
            "label": label_text,
            "label_num": 1 if is_spam else 0,
            "prob": p_spam,
            "threshold": THRESHOLD,
            "length": len(text),
            "padded_or_truncated": changed,
        }
    else:
        raw = RF_MODEL.predict(X_fix)[0]
        is_spam = str(raw).lower() in ("spam", "1", "true")
        return {
            "model": "rf",
            "label": str(raw),
            "label_num": 1 if is_spam else 0,
            "prob": None,
            "threshold": THRESHOLD,
            "length": len(text),
        }

def _display_conf(res: Dict[str, Any]) -> Optional[float]:
    p = res.get("prob", None)
    if p is None:
        return None
    return float(p) if res.get("label_num") == 1 else (1.0 - float(p))

# ===================== Schemas =====================
class PredictIn(BaseModel):
    text: str = Field(..., description="Message to classify")

# ===================== Minimal Endpoints (used by front-end) =====================
@app.get("/")
def root():
    return {"message": "Spam Detection API (RF only)", "version": app.version}

@app.get("/health")
def health():
    return {
        "model": "rf",
        "vectorizer_features": VECT.transform(["x"]).shape[1],
        "selector_output_features": (
            SEL.transform(VECT.transform(["x"])).shape[1] if SEL is not None else None
        ),
        "status": "ok",
        "classes": [str(c) for c in getattr(RF_MODEL, "classes_", [])],
    }

@app.get("/history")
def get_history(limit: int = Query(20, ge=1, le=HISTORY_MAX)):
    items = HISTORY[-limit:] if HISTORY else []
    return {"count": len(items), "items": items}

@app.get("/export/predictions")
def export_predictions(format: str = Query("json", pattern="^(json|csv)$")):
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "text", "prediction", "confidence", "p_spam"])
        for h in HISTORY:
            writer.writerow([
                h.get("ts", ""),
                h.get("text", ""),
                "Spam" if h.get("label_num") == 1 else "Ham",
                f"{h.get('display_conf', 0)*100:.1f}%" if h.get('display_conf') is not None else "",
                f"{h.get('prob', 0):.4f}" if h.get('prob') is not None else "",
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
    return {"count": len(HISTORY), "predictions": HISTORY}

@app.post("/predict/save")
def predict_and_save(req: PredictIn):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Provide non-empty 'text'")

    try:
        res = _predict_rf(text)
        conf = _display_conf(res)

        HISTORY.append({
            "ts": datetime.utcnow().isoformat() + "Z",
            "model": "rf",
            "text": text,
            "prob": res.get("prob", None),
            "display_conf": conf,
            "label": res["label"],
            "label_num": res["label_num"],
            "length": res["length"],
        })
        if len(HISTORY) > HISTORY_MAX:
            del HISTORY[: len(HISTORY) - HISTORY_MAX]
        _write_history(HISTORY)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.delete("/history/{index}")
def delete_history_item(index: int):
    """Delete a specific history item by index (0 = oldest, -1 = newest)"""
    if not HISTORY:
        raise HTTPException(status_code=404, detail="History is empty")
    
    try:
        # Handle negative indices
        if index < 0:
            index = len(HISTORY) + index
        
        if index < 0 or index >= len(HISTORY):
            raise HTTPException(status_code=404, detail=f"Index {index} out of range")
        
        deleted_item = HISTORY.pop(index)
        _write_history(HISTORY)
        return {"message": "Item deleted", "deleted": deleted_item}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete item: {e}")

@app.delete("/history")
def clear_history():
    """Clear all history"""
    count = len(HISTORY)
    HISTORY.clear()
    _write_history(HISTORY)
    return {"message": "History cleared", "deleted_count": count}