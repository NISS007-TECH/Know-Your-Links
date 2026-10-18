

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def main():
    input_path = BASE_DIR / "malicious_phish.csv"
    if not input_path.exists():
        input_path = BASE_DIR / "main_malicious_phish.csv"

    output_path = BASE_DIR / "malicious_phish_clean.csv"

    if not input_path.exists():
        print(f"Original dataset not found at: {input_path}")
        print("Please place 'malicious_phish.csv' or 'main_malicious_phish.csv' inside this dataset2 directory.")
        return

    # Load original dataset
    df = pd.read_csv(input_path)

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
    df.to_csv(output_path, index=False)

    print("Cleaned dataset shape:", df.shape)
    print(f"Saved: {output_path}")
    print("\nCategory counts:")
    print(df["type"].value_counts())


if __name__ == "__main__":
    main()
