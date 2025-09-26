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
    - SMS (spam.csv): maps v1->label, v2->message
    - Email: tries wide list of label/text columns; if missing, auto-infers.
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
        return df.dropna(subset=["label", "message"]).reset_index(drop=True)

    if dataset_type != "email":
        raise ValueError("dataset_type must be 'sms' or 'email'.")

    # ---- EMAIL: normalize headers to lower, strip spaces
    orig_cols = list(df.columns)
    norm_map = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=norm_map, inplace=True)
    cols = set(df.columns)

    # 1) Try to find a label column
    label_candidates = [
        "spam/ham", "label", "is_spam", "is-spam", "target", "category",
        "class", "classes", "y", "tag", "label_num", "spam", "ham"
    ]
    label_col = next((c for c in label_candidates if c in cols), None)

    def _is_binary_like(series: pd.Series) -> bool:
        vals = (
            series.dropna()
                  .astype(str).str.strip().str.lower()
                  .unique().tolist()
        )
        if len(vals) < 1 or len(vals) > 6:  # avoid big categorical fields
            return False
        allowed = {"spam","ham","1","0","true","false","yes","no","s","h"}
        return set(vals).issubset(allowed)

    # If not found, auto-infer a binary-like label column
    if label_col is None:
        for c in df.columns:
            try:
                if _is_binary_like(df[c]):
                    label_col = c
                    break
            except Exception:
                continue

    if label_col is None:
        # help the user by printing all columns seen
        raise ValueError(
            "Email dataset must include a label column (e.g., 'Spam/Ham').\n"
            "Columns found:\n- " + "\n- ".join(f"{oc} -> {norm_map[oc]}" for oc in orig_cols)
        )

    # 2) Find text columns (prefer body, else subject, else longest text-like)
    text_candidates = ["message", "text", "body", "content", "raw", "emailtext", "mail"]
    subject_col = "subject" if "subject" in cols else None
    body_col = next((c for c in text_candidates if c in cols), None)

    if body_col is None and subject_col is None:
        # pick the "most text-like" column by average string length
        lens = []
        for c in df.columns:
            if c == label_col:
                continue
            try:
                s = df[c].astype(str)
                # skip columns that look like ids/dates/short codes
                if s.str.len().mean() < 20:
                    continue
                lens.append((s.str.len().mean(), c))
            except Exception:
                pass
        if lens:
            lens.sort(reverse=True)
            body_col = lens[0][1]

    if subject_col and body_col:
        msg_series = (df[subject_col].fillna("") + " " + df[body_col].fillna("")).str.strip()
    elif body_col:
        msg_series = df[body_col].astype(str)
    elif subject_col:
        msg_series = df[subject_col].astype(str)
    else:
        raise ValueError(
            "Email dataset must include a text-like column (e.g., 'Message' or 'Subject').\n"
            "Columns found:\n- " + "\n- ".join(f"{oc} -> {norm_map[oc]}" for oc in orig_cols)
        )

    out = pd.DataFrame({
        "label": df[label_col].astype(str).str.strip().str.lower(),
        "message": msg_series.astype(str)
    })

    # Normalize various encodings to 'ham'/'spam'
    out["label"] = out["label"].replace({
        "1": "spam", "0": "ham",
        "true": "spam", "false": "ham",
        "yes": "spam", "no": "ham",
        "s": "spam", "h": "ham"
    })

    out = out[out["label"].isin(["ham", "spam"])]
    # Email-specific light cleanup
    out["message"] = out["message"].map(clean_email_text)
    out = out[out["message"].str.strip().ne("")].reset_index(drop=True)
    if out.empty:
        raise ValueError("After normalization/cleanup, no valid email rows remain.")
    return out


if __name__ == "__main__":
    print("Dataset-aware preprocessing demo")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)

    # --- SMS ---
    sms_df = load_dataset("spam.csv", dataset_type="sms")
    print(f"SMS shape: {sms_df.shape}")
    print(sms_df["label"].value_counts())
    sms_out = DATA_DIR / "sms_messages_normalized.csv"
    sms_df.to_csv(sms_out, index=False, encoding="utf-8")

    # --- Enron email ---
    email_df = load_dataset("enron_spam_data.csv", dataset_type="email")
    print("-" * 40)
    print(f"Enron Email shape: {email_df.shape}")
    print(email_df["label"].value_counts())
    email_out = DATA_DIR / "email_messages_normalized.csv"
    email_df.to_csv(email_out, index=False, encoding="utf-8")

    # --- Optional: a third email dataset: emails.csv ---
    try:
        emails_extra = load_dataset("emails.csv", dataset_type="email")
        print("-" * 40)
        print(f"Extra Email (emails.csv) shape: {emails_extra.shape}")
        print(emails_extra["label"].value_counts())
        emails_out = DATA_DIR / "emails_messages_normalized.csv"
        emails_extra.to_csv(emails_out, index=False, encoding="utf-8")
        print(f"Saved extra normalized: {emails_out}")
    except FileNotFoundError:
        print("No emails.csv found — skipping optional dataset.")
    except Exception as e:
        # If the schema is odd, we’ll see the detailed error (with columns) here
        print(f"Could not normalize emails.csv: {e}")

    print("\nSaved:")
    print(f"  {sms_out}")
    print(f"  {email_out}")
    if (DATA_DIR / "emails_messages_normalized.csv").exists():
        print("  data/emails_messages_normalized.csv")
    print("\n✅ demo_preprocessing.py finished.", flush=True)
