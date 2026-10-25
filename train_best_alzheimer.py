
from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "features_segmented55_2 (1).csv")

FEATURES = [
    "MFCC2", "kurt_MFCC30", "mean_MFCC30", "skew_MFCC2",
    "mean_MFCC16", "flt_bnk_eng22", "MFCC30", "kurt_MFCC16",
    "flt_bnk_eng2", "flt_bnk_eng24", "MFCC1", "flt_bnk_eng15",
    "kurt_MFCC2", "flt_bnk_eng20", "flt_bnk_eng13",
    "n_sil_segments", "frac_silence", "min_sil_len",
    "jitter", "shimmer", "HNR",
]

df = df.dropna(subset=["ID", "ad/cn"]).copy()
df["ad/cn"] = pd.to_numeric(df["ad/cn"], errors="coerce")
df = df[df["ad/cn"].isin([0, 1])].copy()
df["ad/cn"] = df["ad/cn"].astype(int)

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

groups = df["ID"].astype(str).str.replace(r"_\d+$", "", regex=True)
X, y = df[FEATURES], df["ad/cn"]

split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups))

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        l2_regularization=1.0,
        random_state=42,
    )),
])

model.fit(X.iloc[train_idx], y.iloc[train_idx])
pred = model.predict(X.iloc[test_idx])

print("Accuracy:", round(accuracy_score(y.iloc[test_idx], pred), 4))
print(classification_report(
    y.iloc[test_idx], pred,
    labels=[0, 1],
    target_names=["Control (CN)", "Alzheimer's disease (AD)"],
    zero_division=0,
))

out = ROOT / "models" / "alzheimer_gradient_boosting.joblib"
out.parent.mkdir(exist_ok=True)
joblib.dump({
    "model": model,
    "features": FEATURES,
    "target": "ad/cn",
    "label_mapping": {
        0: "Control (CN)",
        1: "Alzheimer's disease (AD)",
    },
}, out)

print("Saved:", out)
