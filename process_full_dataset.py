# process_full_dataset.py
import os
import pandas as pd
from sklearn.utils import shuffle

from feature_engineering_pipeline import apply_feature_engineering  # creates 'cleaned_message' etc.

RAW_SMS   = "data/sms_messages_normalized.csv"
RAW_EMAIL = "data/email_messages_normalized.csv"

OUT_COMBINED = "data/combined_messages.csv"
OUT_FEATURES = "data/spam_with_features.csv"

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

    # (Optional) sanity checks
    assert {"label","message"}.issubset(df.columns)
    assert set(df["label"].unique()) <= {"ham","spam"}

    # Save raw combined (handy for report appendix)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUT_COMBINED, index=False)

    # Create features expected by your training pipeline
    # This adds: cleaned_message, message_length, digit_ratio, capital_ratio, special_char_count
    # (and keeps original 'message' & optionally 'label')  ← matches your FE code
    df_feat = apply_feature_engineering(df, message_column="message", inplace=False)

    # Keep label column name as 'label' for model_comparison pipeline
    df_feat.to_csv(OUT_FEATURES, index=False)

    print(f"✅ Wrote {OUT_COMBINED} ({df.shape})")
    print(f"✅ Wrote {OUT_FEATURES} ({df_feat.shape})")

if __name__ == "__main__":
    main()
