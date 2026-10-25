
from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


# Paths
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "features_segmented55_2 (1).csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "alzheimer_rf_binary.joblib"

# Acoustic features used by the original notebook
FEATURES = [
    "MFCC2",
    "kurt_MFCC30",
    "mean_MFCC30",
    "skew_MFCC2",
    "mean_MFCC16",
    "flt_bnk_eng22",
    "MFCC30",
    "kurt_MFCC16",
    "flt_bnk_eng2",
    "flt_bnk_eng24",
    "flt_bnk_eng15",
    "MFCC1",
    "kurt_MFCC2",
    "flt_bnk_eng20",
    "flt_bnk_eng13",
    "n_sil_segments",
    "frac_silence",
    "min_sil_len",
    "jitter",
    "shimmer",
    "HNR",
]

TARGET = "ad/cn"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Place this script beside the CSV file."
        )

    df = pd.read_csv(CSV_PATH)

    required = ["ID", TARGET, *FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    # Keep valid binary labels: 0 = control, 1 = Alzheimer's disease.
    df = df.dropna(subset=["ID", TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Convert acoustic features to numeric values.
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove only the final segment suffix, e.g. _1, _2, _10.
    df["recording_id"] = (
        df["ID"].astype(str).str.replace(r"_\d+$", "", regex=True)
    )

    # A recording must never appear in both train and test sets.
    groups = df["recording_id"]
    X = df[FEATURES]
    y = df[TARGET]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    assert train_groups.isdisjoint(test_groups), (
        "Data leakage detected: a recording is in both sets."
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("=== ALZHEIMER'S SPEECH MODEL ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Unique recording IDs: {groups.nunique()}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Testing rows: {len(test_idx)}")
    print(f"Training recordings: {len(train_groups)}")
    print(f"Testing recordings: {len(test_groups)}")
    print(f"Features: {len(FEATURES)}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Control (CN)", "Alzheimer's disease (AD)"],
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "label_mapping": {
            0: "Control (CN)",
            1: "Alzheimer's disease (AD)",
        },
    }, MODEL_PATH)

    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
