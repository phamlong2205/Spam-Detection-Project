from pathlib import Path
import pandas as pd
from typing import Optional
from src.spam_detection.email_preprocessor import clean_email_text
# (optional) show a preview using the unified normalizer if you added it
try:
    from src.spam_detection.preprocessing import preprocess_text_unified
except Exception:
    preprocess_text_unified = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    candidates = [
        DATA_DIR / path_str,         # e.g. data/spam.csv
        BASE_DIR / path_str,         # e.g. spam.csv at project root
        p if p.is_absolute() else None
    ]
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find '{path_str}'. Tried: {(DATA_DIR / path_str)}, {(BASE_DIR / path_str)}"
    )


def load_dataset(path: str, dataset_type: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ['label','message'].
    - For SMS (spam.csv): maps v1->label, v2->message
    - For Enron email: uses 'Spam/Ham' as label and Subject+Message as message,
      then runs light HTML/quote/link cleanup.
    """
    path = _resolve(path)
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    if dataset_type == "sms":
        if not {"v1", "v2"}.issubset(df.columns):
            raise ValueError("SMS dataset must contain columns v1 (label) and v2 (message).")
        df = df[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
        df = df.dropna(subset=["label", "message"]).reset_index(drop=True)
        return df

    if dataset_type != "email":
        raise ValueError("dataset_type must be 'sms' or 'email'.")

    # ---- EMAIL (Enron) exact-ish schema handling ----
    norm_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=norm_map)

    # typical columns in your file: 'message id', 'subject', 'message', 'spam/ham', 'date'
    # pick label
    label_col = None
    for c in ("spam/ham", "label", "is_spam", "target", "category", "class"):
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("Email dataset must include a label column (e.g., 'Spam/Ham').")

    # pick text (prefer body, else subject, else both)
    subject_col = "subject" if "subject" in df.columns else None
    message_col = None
    for c in ("message", "text", "body", "content", "raw"):
        if c in df.columns:
            message_col = c
            break

    if subject_col and message_col:
        msg_series = (df[subject_col].fillna("") + " " + df[message_col].fillna("")).str.strip()
    elif message_col:
        msg_series = df[message_col].astype(str)
    elif subject_col:
        msg_series = df[subject_col].astype(str)
    else:
        raise ValueError("Email dataset must include a text-like column (e.g., 'Message' or 'Subject').")

    out = pd.DataFrame({
        "label": df[label_col].astype(str).str.strip().str.lower(),
        "message": msg_series.astype(str)
    })

    # normalize label values
    out["label"] = out["label"].replace({
        "1": "spam", "0": "ham",
        "true": "spam", "false": "ham",
        "yes": "spam", "no": "ham"
    })

    out = out[out["label"].isin(["ham", "spam"])]

    # email-specific light cleanup
    out["message"] = out["message"].map(clean_email_text)

    out = out[out["message"].str.strip().ne("")].reset_index(drop=True)
    if out.empty:
        raise ValueError("After normalization/cleanup, no valid email rows remain.")
    return out


if __name__ == "__main__":
    print("Dataset-aware preprocessing demo")
    print("=" * 60)

    # Load both datasets from ./data
    sms_df = load_dataset("spam.csv", dataset_type="sms")
    email_df = load_dataset("enron_spam_data.csv", dataset_type="email")

    print(f"SMS shape: {sms_df.shape}")
    print(sms_df["label"].value_counts())
    print("-" * 40)
    print(f"Email shape: {email_df.shape}")
    print(email_df["label"].value_counts())

    # Optional: preview unified normalization if available
    if preprocess_text_unified is not None:
        print("\nPreview (SMS):")
        for t in sms_df["message"].head(3).tolist():
            print("•", preprocess_text_unified(t, source="sms")[:120])

        print("\nPreview (Email):")
        for t in email_df["message"].head(3).tolist():
            print("•", preprocess_text_unified(t, source="email")[:120])

    # Save normalized copies for the next step
    DATA_DIR.mkdir(exist_ok=True)
    sms_out = DATA_DIR / "sms_messages_normalized.csv"
    email_out = DATA_DIR / "email_messages_normalized.csv"
    sms_df.to_csv(sms_out, index=False, encoding="utf-8")
    email_df.to_csv(email_out, index=False, encoding="utf-8")

    print("\nSaved:")
    print(f"  {sms_out}")
    print(f"  {email_out}")
    print("\n✅ demo_preprocessing.py finished.", flush=True)
