
import pandas as pd

df = pd.read_csv("dataset2/malicious_phish_clean.csv")

urls = df["url"].fillna("").astype(str).str.strip()

has_scheme = urls.str.match(
    r"^[a-zA-Z][a-zA-Z0-9+.-]*://"
)

print("Total URLs:", len(urls))
print("URLs with http/https scheme:", has_scheme.sum())
print("URLs without scheme:", (~has_scheme).sum())

print("\nSample URLs from dataset:")
print(urls.head(10).to_string(index=False))

print("\nURL category counts:")
print(df["type"].value_counts())
