
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Load the same testing dataset
test_df = pd.read_csv("dataset2/split/test.csv")

X_test = test_df.drop(columns=["type"])
y_test = test_df["type"]

# Model file paths
models = {
    "Logistic Regression": "models/linkguard_model.joblib",
    "Random Forest": "models/linkguard_random_forest.joblib",
    "Gradient Boosting": "models/linkguard_gradient_boosting.joblib"
}

results = []

# Evaluate each trained model
for name, path in models.items():
    print(f"\nEvaluating {name}...")

    model = joblib.load(path)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )
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

# Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro F1 (%)", ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv("models/model_comparison.csv", index=False)
print("\nResults saved to models/model_comparison.csv")
