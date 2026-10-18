
# import pandas as pd

# df = pd.read_csv("dataset2/malicious_phish.csv")

# print("Dataset Shape:", df.shape)
# print("\nColumn Names:", df.columns.tolist())
# print("\nFirst 5 Rows:\n", df.head())
# print("\nCategory Counts:\n", df["type"].value_counts())



# print("\n--- DATA QUALITY CHECK ---")

# # 1. Check missing values
# print("\nMissing values:")
# print(df.isnull().sum())

# # 2. Check duplicate rows
# print("\nDuplicate rows:")
# print(df.duplicated().sum())

# # 3. Check duplicate URLs
# print("\nDuplicate URLs:")
# print(df["url"].duplicated().sum())

# # 4. Check unique labels
# print("\nUnique categories:")
# print(df["type"].unique())

# # 5. Check URLs having different labels
# print("\nConflicting URLs:")
# label_counts = df.groupby("url")["type"].nunique()
# print("Number of conflicting URLs:", (label_counts > 1).sum())




# print("\n--- CONFLICTING URL DETAILS ---")

# label_counts = df.groupby("url")["type"].nunique()
# conflicts = label_counts[label_counts > 1].index

# conflict_data = df[df["url"].isin(conflicts)]

# print(conflict_data.to_string(index=False))





from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def main():
    data_path = BASE_DIR / "malicious_phish_clean.csv"
    if not data_path.exists():
        print(f"Cleaned dataset not found at: {data_path}")
        return

    df = pd.read_csv(data_path)

    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nConflicting URLs:")
    print((df.groupby("url")["type"].nunique() > 1).sum())

    print("\nCategory counts:")
    print(df["type"].value_counts())

    print("\n--- SAMPLE URLS BY CATEGORY ---")
    for category in df["type"].unique():
        print(f"\nCategory: {category}")
        print(df[df["type"] == category]["url"].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
