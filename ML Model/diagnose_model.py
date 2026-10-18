from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


def main():
    # Load the feature dataset
    feature_path = BASE_DIR / "dataset2" / "url_features.csv"
    if not feature_path.exists():
        print(f"ERROR: Feature file not found at: {feature_path}")
        print("Please run extract_features.py first.")
        raise SystemExit(1)

    candidate_models = [
        BASE_DIR / "models" / "linkguard_gradient_boosting.joblib",
        BASE_DIR / "models" / "linkguard_model_v2.joblib",
        BASE_DIR / "models" / "linkguard_random_forest.joblib",
        BASE_DIR / "models" / "linkguard_model.joblib",
    ]

    model_path = None
    for p in candidate_models:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        print("ERROR: No trained model found in models/ directory!")
        print("Please train a model first using train_gradient_boosting.py or train_random_forest.py.")
        raise SystemExit(1)

    print(f"Diagnosing model: {model_path.name}")
    df = pd.read_csv(feature_path)

    X = df.drop(columns=["type"])
    y = df["type"]

    # Recreate the stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load the trained model
    model = joblib.load(model_path)

    # Predict on test set
    print(f"Generating predictions on {len(X_test):,} test samples...")
    predictions = model.predict(X_test)

    # Calculate false positive rate on benign samples
    benign_mask = (y_test == "benign")
    total_benign = int(benign_mask.sum())
    false_phishing = int(((y_test == "benign") & (predictions == "phishing")).sum())
    false_malware = int(((y_test == "benign") & (predictions == "malware")).sum())
    false_defacement = int(((y_test == "benign") & (predictions == "defacement")).sum())
    true_benign = int(((y_test == "benign") & (predictions == "benign")).sum())

    pct_true = (true_benign / total_benign * 100) if total_benign > 0 else 0.0
    pct_phish = (false_phishing / total_benign * 100) if total_benign > 0 else 0.0
    pct_malw = (false_malware / total_benign * 100) if total_benign > 0 else 0.0
    pct_def = (false_defacement / total_benign * 100) if total_benign > 0 else 0.0

    print("\n" + "=" * 55)
    print("             BENIGN URL CLASSIFICATION HEALTH")
    print("=" * 55)
    print(f"Total Benign Test Samples:        {total_benign:,}")
    print(f"Correctly Classified as BENIGN:   {true_benign:,} ({pct_true:.2f}%)")
    print(f"Wrongly Flagged as PHISHING:      {false_phishing:,} ({pct_phish:.2f}%)")
    print(f"Wrongly Flagged as MALWARE:       {false_malware:,} ({pct_malw:.2f}%)")
    print(f"Wrongly Flagged as DEFACEMENT:    {false_defacement:,} ({pct_def:.2f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
