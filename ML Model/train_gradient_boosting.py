import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load training and testing datasets
train_path = Path("dataset2/split/train.csv")
test_path = Path("dataset2/split/test.csv")

if not train_path.exists() or not test_path.exists():
    print("ERROR: Split datasets not found in dataset2/split/")
    print("Please run split_dataset.py first.")
    raise SystemExit(1)

print("Loading training and testing datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. Separate features and labels
X_train = train_df.drop(columns=["type"])
y_train = train_df["type"]

X_test = test_df.drop(columns=["type"])
y_test = test_df["type"]

print(f"Training samples: {len(X_train):,} | Features: {X_train.shape[1]}")
print(f"Testing samples:  {len(X_test):,}")

# 3. Create the Gradient Boosting model using HistGradientBoostingClassifier
# (Fast, highly accurate, supports large datasets, compact model file size)
model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.1,
    max_depth=12,
    min_samples_leaf=20,
    random_state=42
)

# 4. Train the model
print("\nTraining Gradient Boosting model... Please wait.")
t0 = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"Training completed in {elapsed:.1f} seconds!")

# 5. Predict categories for test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print(f"Gradient Boosting Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("Category order:", list(model.classes_))

# 7. Save the trained model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
save_path = model_dir / "linkguard_gradient_boosting.joblib"
joblib.dump(model, save_path)

print(f"\nModel successfully saved at: {save_path}")
