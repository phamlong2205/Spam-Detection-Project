# process_full_dataset.py
import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
from feature_engineering_pipeline import apply_feature_engineering  # creates 'cleaned_message' etc.
from src.spam_detection.data_filters import is_dirty_row  # your filter

DATA_DIR = Path("data")

# These are what demo_preprocessing.py writes now
CANDIDATE_INPUTS = [
    DATA_DIR / "sms_messages_normalized.csv",
    DATA_DIR / "email_messages_normalized.csv",
    DATA_DIR / "emails_messages_normalized.csv",  # optional extra email dataset
]

OUT_COMBINED        = DATA_DIR / "combined_messages.csv"
OUT_FEATURES        = DATA_DIR / "spam_with_features.csv"
OUT_FEATURES_CLEAN  = DATA_DIR / "spam_with_features_clean.csv"


def ensure_inputs_exist() -> list[Path]:
    """Return the list of existing normalized inputs; error if none."""
    DATA_DIR.mkdir(exist_ok=True)
    existing = [p for p in CANDIDATE_INPUTS if p.exists()]
    if not existing:
        raise SystemExit(
            "No normalized inputs found.\n"
            "Expected one or more of:\n  - data/sms_messages_normalized.csv\n"
            "  - data/email_messages_normalized.csv\n"
            "  - data/emails_messages_normalized.csv (optional)\n"
            "Run demo_preprocessing.py first."
        )
    return existing


def safe_to_csv(df: pd.DataFrame, path: Path):
    """
    Atomic write to avoid partial files; if Windows locks the target (e.g., open in Excel),
    fall back to an alternate filename and continue.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)  # atomic on same volume
        print(f"✅ Wrote {path} ({df.shape})")
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(alt, index=False, encoding="utf-8")
        print(f"⚠️ '{path}' is locked (maybe open in Excel). Wrote '{alt}' instead ({df.shape}).")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    inputs = ensure_inputs_exist()
    print("Merging normalized inputs:")
    for p in inputs:
        print(" -", p)

    # 1) Read & merge
    dfs = [pd.read_csv(p) for p in inputs]
    df = pd.concat(dfs, ignore_index=True)

    # 2) Sanity & cleanup of schema
    df = df.dropna(subset=["label", "message"])
    df = df[df["label"].isin(["ham", "spam"])]

    # 3) (Optional) de-duplicate identical messages (case/space-insensitive)
    before = len(df)
    df["__key"] = df["message"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset="__key").drop(columns="__key").reset_index(drop=True)
    deduped = before - len(df)
    if deduped:
        print(f"🧹 Deduped {deduped} duplicate messages.")

    # 4) Shuffle for good measure
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # 5) Save combined raw (report appendix / traceability)
    safe_to_csv(df, OUT_COMBINED)

    # 6) Build features (uses your pipeline; produces cleaned_message & numeric features)
    df_feat = apply_feature_engineering(df, message_column="message", inplace=False)
    safe_to_csv(df_feat, OUT_FEATURES)

    # 7) Remove “dirty” rows using your heuristic filter
    before = len(df_feat)
    df_clean = df_feat[~df_feat.apply(is_dirty_row, axis=1)].reset_index(drop=True)
    removed = before - len(df_clean)
    pct = (removed / before * 100) if before else 0.0
    print(f"🧽 Removed {removed} noisy rows ({pct:.2f}%).")
    safe_to_csv(df_clean, OUT_FEATURES_CLEAN)


if __name__ == "__main__":
    main()
