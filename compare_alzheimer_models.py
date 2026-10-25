
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "features_segmented55_2 (1).csv"
OUTPUT = ROOT / "model_comparison.csv"

FEATURES = [
    "MFCC2", "kurt_MFCC30", "mean_MFCC30", "skew_MFCC2",
    "mean_MFCC16", "flt_bnk_eng22", "MFCC30", "kurt_MFCC16",
    "flt_bnk_eng2", "flt_bnk_eng24", "MFCC1", "flt_bnk_eng15",
    "kurt_MFCC2", "flt_bnk_eng20", "flt_bnk_eng13",
    "n_sil_segments", "frac_silence", "min_sil_len",
    "jitter", "shimmer", "HNR",
]
TARGET = "ad/cn"


def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["ID", TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep segments from each source recording in the same split.
    groups = df["ID"].astype(str).str.replace(
        r"_\d+$", "", regex=True
    )
    X, y = df[FEATURES], df[TARGET]

    split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(split.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = []
    print(f"Training rows: {len(train_idx)} | Test rows: {len(test_idx)}")
    print(f"Test recordings: {groups.iloc[test_idx].nunique()}")
    print("\n=== MODEL COMPARISON ===")

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob),
        }
        results.append(metrics)
        print(
            f"{name}: Accuracy={metrics['Accuracy']:.4f}, "
            f"F1={metrics['F1']:.4f}, "
            f"ROC-AUC={metrics['ROC_AUC']:.4f}"
        )

    result_df = pd.DataFrame(results).sort_values(
        "F1", ascending=False
    )
    result_df.to_csv(OUTPUT, index=False)
    print("\nResults saved to:", OUTPUT)
    print("\nRanked by F1:")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nExisting model was not modified.")


if __name__ == "__main__":
    main()
