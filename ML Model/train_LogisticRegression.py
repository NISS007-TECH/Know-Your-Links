
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent


def main():
    train_path = BASE_DIR / "dataset2" / "split" / "train.csv"
    test_path = BASE_DIR / "dataset2" / "split" / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print(f"ERROR: Split datasets not found at {train_path.parent}")
        print("Please run split_dataset.py first.")
        raise SystemExit(1)

    # Load training and testing datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features and labels
    X_train = train_df.drop(columns=["type"])
    y_train = train_df["type"]

    X_test = test_df.drop(columns=["type"])
    y_test = test_df["type"]

    # Create the ML model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )

    # Train the model
    print("Training the model... Please wait.")
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Display results
    print("\nModel training completed!")
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Save the trained model
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / "linkguard_model.joblib"
    joblib.dump(model, save_path)

    print(f"\nModel saved at: {save_path}")


if __name__ == "__main__":
    main()
