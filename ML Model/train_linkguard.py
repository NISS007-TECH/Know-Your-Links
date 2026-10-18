import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from extract_features import extract_features

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset2" / "split" / "train.csv"
TEST_PATH = BASE_DIR / "dataset2" / "split" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "linkguard_model_v2.joblib"
GB_MODEL_PATH = BASE_DIR / "models" / "linkguard_gradient_boosting.joblib"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists() or not TEST_PATH.exists():
        print(f"ERROR: Dataset split files not found at {DATA_PATH} or {TEST_PATH}")
        print("Please run python extract_features.py and python split_dataset.py first.")
        raise SystemExit(1)

    print("Loading training and testing datasets...")
    train_df = pd.read_csv(DATA_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["type"])
    y_train = train_df["type"]

    X_test = test_df.drop(columns=["type"])
    y_test = test_df["type"]

    print(f"Training samples: {len(X_train):,} | Features: {X_train.shape[1]}")
    print(f"Testing samples:  {len(X_test):,}")

    print("\nTraining LinkGuard High-Performance Classifier... Please wait.")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_depth=12,
        min_samples_leaf=20,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Model trained in {elapsed:.1f} seconds!")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("\n" + "=" * 50)
    print(f"LinkGuard Model Accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)

    print("\nClassification report:")
    print(classification_report(y_test, predictions, digits=3, zero_division=0))

    print("Confusion matrix (class order:", list(model.classes_), "):")
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model, GB_MODEL_PATH)
    print(f"\nSaved LinkGuard flagship model to: {MODEL_PATH}")
    print(f"Updated gradient boosting model at: {GB_MODEL_PATH}")


if __name__ == "__main__":
    main()
