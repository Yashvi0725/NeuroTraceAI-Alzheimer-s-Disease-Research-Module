from pathlib import Path
import joblib
import math
import pandas as pd

# Model paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "alzheimer_gradient_boosting.joblib"

# Load the trained model once when this module is imported.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run train_best_alzheimer.py first."
    )

saved = joblib.load(MODEL_PATH)

model = saved["model"]
FEATURES = saved["features"]
LABELS = saved["label_mapping"]


def predict_features(feature_values):
    """
    Predict from a dictionary containing the 21 trained acoustic features.

    Example:
        result = predict_features({
            "MFCC2": 0.12,
            ...
        })
    """
    missing = [
        name for name in FEATURES
        if name not in feature_values
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = []

    for name in FEATURES:
        value = feature_values[name]

        if value is None:
            row.append(float("nan"))
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{name}' must be numeric."
            )

        if not math.isfinite(number):
            number = float("nan")

        row.append(number)

    # Use a DataFrame with feature names to avoid sklearn warnings.
    input_data = pd.DataFrame([row], columns=FEATURES)

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    # Get the class labels from the trained classifier.
    classes = model.named_steps["classifier"].classes_

    confidence = {
        LABELS[int(class_id)]: round(float(probability), 4)
        for class_id, probability in zip(classes, probabilities)
    }

    return {
        "prediction": LABELS[prediction],
        "class_id": prediction,
        "probabilities": confidence,
        "notice": (
            "Research prediction only; not a medical diagnosis."
        ),
    }


if __name__ == "__main__":
    # Quick test using one existing feature row from the dataset.
    csv_path = BASE_DIR / "features_segmented55_2 (1).csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    sample = pd.read_csv(csv_path, nrows=1)

    result = predict_features(sample.iloc[0].to_dict())

    print("=== ALZHEIMER'S PREDICTION MODULE TEST ===")
    print("Input: first row from the feature dataset")
    print("Prediction:", result["prediction"])
    print("Class ID:", result["class_id"])
    print("Model probabilities:", result["probabilities"])
    print(result["notice"])
