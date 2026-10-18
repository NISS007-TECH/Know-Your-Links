
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score

BASE_DIR = Path(__file__).resolve().parent


def main():
    test_path = BASE_DIR / "dataset2" / "split" / "test.csv"
    if not test_path.exists():
        print(f"ERROR: Testing dataset not found at: {test_path}")
        print("Please run split_dataset.py first.")
        raise SystemExit(1)

    print("Loading testing dataset...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["type"])
    y_test = test_df["type"]

    models = {
        "Logistic Regression": BASE_DIR / "models" / "linkguard_model.joblib",
        "Random Forest": BASE_DIR / "models" / "linkguard_random_forest.joblib",
        "Gradient Boosting": BASE_DIR / "models" / "linkguard_gradient_boosting.joblib",
        "LinkGuard Flagship (v2)": BASE_DIR / "models" / "linkguard_model_v2.joblib",
    }

    results = []

    for name, path in models.items():
        if not path.exists():
            print(f"Skipping {name} (model file not found at: {path.name})")
            continue

        print(f"\nEvaluating {name}...")
        try:
            model = joblib.load(path)
            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
            phishing_recall = recall_score(
                y_test,
                predictions,
                labels=["phishing"],
                average="macro",
                zero_division=0
            )

            results.append({
                "Model": name,
                "Accuracy (%)": round(accuracy * 100, 2),
                "Macro F1 (%)": round(macro_f1 * 100, 2),
                "Phishing Recall (%)": round(phishing_recall * 100, 2)
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    if not results:
        print("\nNo trained models were found to compare. Please train at least one model first.")
        return

    results_df = pd.DataFrame(results).sort_values(by="Macro F1 (%)", ascending=False)
    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    out_path = BASE_DIR / "models" / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
