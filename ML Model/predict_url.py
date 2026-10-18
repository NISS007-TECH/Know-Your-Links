import sys
from pathlib import Path

import joblib
import pandas as pd

from extract_features import (
    BRANDS,
    KNOWN_TOP_DOMAINS,
    canonicalize_url,
    extract_features,
    get_registered_domain,
)

# Preferred model paths
MODEL_PATHS = [
    Path("models/linkguard_gradient_boosting.joblib"),
    Path("models/linkguard_model_v2.joblib"),
    Path("models/linkguard_random_forest.joblib"),
]

model = None
chosen_path = None
for p in MODEL_PATHS:
    if p.exists():
        chosen_path = p
        model = joblib.load(p)
        break

if model is None:
    print("ERROR: Trained model not found in models/ directory!")
    print("Please run python train_gradient_boosting.py or python train_random_forest.py first.")
    raise SystemExit(1)


def analyze_url(url_input):
    url = str(url_input).strip()
    if not url:
        print("Please enter a valid URL.")
        return

    canon = canonicalize_url(url)
    features_dict = extract_features(url)
    features_df = pd.DataFrame([features_dict])

    host = canon.split("/")[0]
    reg_domain = get_registered_domain(host)

    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    categories = model.classes_
    confidence = max(probabilities) * 100

    print("\n" + "=" * 55)
    print("           LINKGUARD URL SECURITY ANALYSIS")
    print("=" * 55)
    print(f"Original URL:     {url}")
    print(f"Canonical Host:   {host}")
    print(f"Registered Domain:{reg_domain}")
    print(f"Model Engine:     {chosen_path.name}")

    # Domain intelligence summary
    if features_dict.get("is_known_top_domain"):
        print("Domain Trust:     VERIFIED MAJOR GLOBAL DOMAIN (e.g. Google, YouTube, etc.)")
    elif features_dict.get("brand_impersonation"):
        print("Domain Alert:     POTENTIAL BRAND IMPERSONATION DETECTED!")
    elif features_dict.get("has_ip"):
        print("Domain Alert:     RAW IP ADDRESS HOST DETECTED")

    print("\n--- Model Probability Breakdown ---")
    for category, probability in sorted(
        zip(categories, probabilities),
        key=lambda item: item[1],
        reverse=True
    ):
        bar = "#" * int(probability * 25)
        print(f"  {category.capitalize():12}: {probability * 100:6.2f}%  [{bar:<25}]")

    print(f"\nHighest Confidence: {confidence:.2f}%")

    print("\n--- Final Assessment ---")
    if features_dict.get("brand_impersonation") and prediction == "benign":
        # Security safety net: if a suspicious brand spoof was flagged
        print("WARNING: SUSPICIOUS / POSSIBLE PHISHING")
        print("A popular brand name was detected in a third-party domain or subdomain.")
        print("Verify the exact domain before entering credentials.")
    elif prediction == "benign":
        print("STATUS: BENIGN (SAFE)")
        print("The link appears consistent with standard benign web traffic.")
    elif prediction == "phishing":
        print("WARNING: PHISHING DETECTED!")
        print("The model identified structural patterns characteristic of credential phishing.")
    elif prediction == "malware":
        print("WARNING: MALWARE DETECTED!")
        print("The link exhibits traits associated with malware distribution or malicious payloads.")
    elif prediction == "defacement":
        print("WARNING: DEFACEMENT / COMPROMISED SITE DETECTED!")
        print("The link shows characteristics of website defacement or injected scripts.")
    else:
        print(f"RESULT: {str(prediction).upper()}")

    print("=" * 55 + "\n")


def main():
    if len(sys.argv) > 1:
        raw_url = " ".join(sys.argv[1:]).strip()
    else:
        try:
            raw_url = input("Enter a URL to check: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if raw_url:
        analyze_url(raw_url)
    else:
        print("No URL provided. Exiting.")


if __name__ == "__main__":
    main()
