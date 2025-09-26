# process_full_dataset.py
import os
import pandas as pd
from sklearn.utils import shuffle
from src.spam_detection.data_filters import is_dirty_row
from feature_engineering_pipeline import apply_feature_engineering  # creates 'cleaned_message' etc.

RAW_SMS   = "data/sms_messages_normalized.csv"
RAW_EMAIL = "data/email_messages_normalized.csv"

OUT_COMBINED = "data/combined_messages.csv"
OUT_FEATURES = "data/spam_with_features.csv"
OUT_FEATURES_CLEAN = "data/spam_with_features_clean.csv" 

def ensure_inputs_exist():
    missing = [p for p in [RAW_SMS, RAW_EMAIL] if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing normalized inputs: "
            + ", ".join(missing)
            + "\nRun demo_preprocessing.py first to create them."
        )

def main():
    ensure_inputs_exist()

    sms   = pd.read_csv(RAW_SMS)
    email = pd.read_csv(RAW_EMAIL)

    # Concatenate, shuffle, keep consistent schema
    df = pd.concat([sms, email], ignore_index=True)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # Save raw combined (handy for report appendix)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_COMBINED, index=False)

    # Build features (your existing function)
    df_feat = apply_feature_engineering(df, message_column="message", inplace=False)
    df_feat.to_csv(OUT_FEATURES, index=False)

    # ------ NEW: drop dirty rows -------
    before = len(df_feat)
    df_clean = df_feat[~df_feat.apply(is_dirty_row, axis=1)].reset_index(drop=True)
    removed = before - len(df_clean)
    pct = (removed / before * 100) if before else 0.0

    df_clean.to_csv(OUT_FEATURES_CLEAN, index=False)

    print(f"✅ Wrote {OUT_COMBINED} ({df.shape})")
    print(f"✅ Wrote {OUT_FEATURES} ({df_feat.shape})")
    print(f"✅ Wrote {OUT_FEATURES_CLEAN} ({df_clean.shape}) — removed {removed} rows ({pct:.2f}%).")
    
if __name__ == "__main__":
    main()
