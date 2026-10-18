
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load training and testing datasets
train_df = pd.read_csv("dataset2/split/train.csv")
test_df = pd.read_csv("dataset2/split/test.csv")

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
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/linkguard_model.joblib")

print("\nModel saved at: models/linkguard_model.joblib")
