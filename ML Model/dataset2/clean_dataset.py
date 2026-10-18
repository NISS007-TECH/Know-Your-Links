
import pandas as pd

# Load original dataset
df = pd.read_csv("dataset2/malicious_phish.csv")

print("Original dataset shape:", df.shape)

# Remove extra spaces from text columns
df["url"] = df["url"].astype(str).str.strip()
df["type"] = df["type"].astype(str).str.strip().str.lower()

# Remove rows with empty URL or label
df = df[
    df["url"].ne("") &
    df["type"].ne("")
].copy()

# Find URLs that have multiple labels
label_counts = df.groupby("url")["type"].nunique()
conflicts = label_counts[label_counts > 1].index

print("Conflicting URLs:", len(conflicts))

# Exclude conflicting URLs from this clean copy
df = df[~df["url"].isin(conflicts)].copy()

# Remove exact duplicate rows
df = df.drop_duplicates()

# Save a separate cleaned dataset
df.to_csv("dataset2/malicious_phish_clean.csv", index=False)

print("Cleaned dataset shape:", df.shape)
print("Saved: dataset2/malicious_phish_clean.csv")
print("\nCategory counts:")
print(df["type"].value_counts())
