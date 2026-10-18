from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the feature dataset
feature_path = Path("dataset2/url_features.csv")
model_path = Path("models/linkguard_gradient_boosting.joblib")
if not model_path.exists():
    model_path = Path("models/linkguard_random_forest.joblib")

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
print("Generating predictions on 128k test samples...")
predictions = model.predict(X_test)

# Calculate false positive rate on benign samples
benign_mask = (y_test == "benign")
total_benign = benign_mask.sum()
false_phishing = ((y_test == "benign") & (predictions == "phishing")).sum()
false_malware = ((y_test == "benign") & (predictions == "malware")).sum()
false_defacement = ((y_test == "benign") & (predictions == "defacement")).sum()
true_benign = ((y_test == "benign") & (predictions == "benign")).sum()

print("\n" + "=" * 55)
print("             BENIGN URL CLASSIFICATION HEALTH")
print("=" * 55)
print(f"Total Benign Test Samples:        {total_benign:,}")
print(f"Correctly Classified as BENIGN:   {true_benign:,} ({true_benign/total_benign*100:.2f}%)")
print(f"Wrongly Flagged as PHISHING:      {false_phishing:,} ({false_phishing/total_benign*100:.2f}%)")
print(f"Wrongly Flagged as MALWARE:       {false_malware:,} ({false_malware/total_benign*100:.2f}%)")
print(f"Wrongly Flagged as DEFACEMENT:    {false_defacement:,} ({false_defacement/total_benign*100:.2f}%)")
print("=" * 55)
