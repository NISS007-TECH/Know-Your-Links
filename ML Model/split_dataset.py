from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the feature dataset
data_path = Path("dataset2/url_features.csv")
if not data_path.exists():
    print("ERROR: dataset2/url_features.csv not found!")
    print("Run extract_features.py first.")
    raise SystemExit(1)

print("Loading feature dataset...")
df = pd.read_csv(data_path)

# Separate features and labels
X = df.drop(columns=["type"])
y = df["type"]

# Split into 80% training and 20% testing
print("Splitting dataset (80% train, 20% test, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create output folder
output_folder = Path("dataset2/split")
output_folder.mkdir(parents=True, exist_ok=True)

# Save training and testing files
train_data = X_train.copy()
train_data["type"] = y_train

test_data = X_test.copy()
test_data["type"] = y_test

train_data.to_csv(output_folder / "train.csv", index=False)
test_data.to_csv(output_folder / "test.csv", index=False)

print("\nDataset split completed successfully!")
print("Total rows:", len(df))
print(f"Training rows: {len(train_data):,} -> {output_folder / 'train.csv'}")
print(f"Testing rows:  {len(test_data):,} -> {output_folder / 'test.csv'}")

# Class distribution in train and test
print("\nTraining category counts:")
print(train_data["type"].value_counts())
print("\nTesting category counts:")
print(test_data["type"].value_counts())
