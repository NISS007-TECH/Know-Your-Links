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

# Load the best available LinkGuard ML model
MODEL_PATHS = [
    Path("models/linkguard_gradient_boosting.joblib"),
    Path("models/linkguard_model_v2.joblib"),
    Path("models/linkguard_random_forest.joblib"),
]

model = None
model_source = None
for path in MODEL_PATHS:
    if path.exists():
        model = joblib.load(path)
        model_source = path.name
        break

if model is None:
    print("ERROR: No trained model found in models/ directory!")
    print("Please run 'python train_gradient_boosting.py' first.")
    raise SystemExit(1)


def evaluate_link(url_str):
    url = str(url_str).strip()
    if not url:
        print("Please provide a valid URL.")
        return

    canon = canonicalize_url(url)
    features_dict = extract_features(url)
    features_df = pd.DataFrame([features_dict])

    host = canon.split("/")[0]
    reg_domain = get_registered_domain(host)
    is_known = bool(features_dict.get("is_known_top_domain", 0))
    is_spoofed = bool(features_dict.get("brand_impersonation", 0))
    has_ip = bool(features_dict.get("has_ip", 0))

    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    confidence = max(probabilities) * 100

    print("\n" + "=" * 60)
    print("             LINKGUARD ML THREAT INSPECTOR")
    print("=" * 60)
    print(f"Target URL:        {url}")
    print(f"Canonical URL:     {canon}")
    print(f"Hostname:          {host}")
    print(f"Registered Domain: {reg_domain or '(none)'}")
    print(f"Active ML Model:   {model_source}")
    print("-" * 60)

    print("Security Indicators:")
    if is_known:
        print("  [+] Top Domain Check: VERIFIED GLOBAL DOMAIN (e.g. Google, YouTube)")
    else:
        print("  [ ] Top Domain Check: Unverified / Custom domain")

    if is_spoofed:
        print("  [!] Spoofing Alert:   TARGETED BRAND NAME DETECTED IN UNRELATED DOMAIN!")
    if has_ip:
        print("  [!] IP Host Alert:    URL uses raw IP address instead of domain name!")
    if features_dict.get("suspicious_tld"):
        print("  [!] TLD Risk:         High-abuse TLD extension detected")
    if features_dict.get("suspicious_keyword_count", 0) > 0:
        print(f"  [!] Keywords:         {features_dict['suspicious_keyword_count']} sensitive keywords detected in path/query")

    print("-" * 60)
    print("Model Classification Probabilities:")
    for cat, prob in sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True):
        print(f"  {cat.capitalize():12}: {prob * 100:6.2f}%")

    print(f"\nHighest Model Confidence: {confidence:.2f}%")
    print("-" * 60)

    # Final verdict
    if is_spoofed and prediction == "benign":
        print("VERDICT: [!] SUSPICIOUS (Potential Phishing/Brand Impersonation)")
        print("Advisory: While general structure looks standard, a recognized brand name")
        print("          appears on an unauthorized domain. Do NOT enter credentials.")
    elif prediction == "benign":
        print("VERDICT: [+] BENIGN (Legitimate / Clean Link)")
        print("Advisory: This URL matches genuine, benign traffic patterns.")
    elif prediction == "phishing":
        print("VERDICT: [!] PHISHING DETECTED")
        print("Advisory: This URL exhibits deceptive structures characteristic of phishing.")
    elif prediction == "malware":
        print("VERDICT: [!] MALWARE DETECTED")
        print("Advisory: High probability of malware distribution or malicious exploit payload.")
    elif prediction == "defacement":
        print("VERDICT: [!] DEFACEMENT DETECTED")
        print("Advisory: Patterns indicate defaced websites, injection, or compromised servers.")
    else:
        print(f"VERDICT: {str(prediction).upper()}")

    print("=" * 60 + "\n")


def main():
    if len(sys.argv) > 1:
        target = " ".join(sys.argv[1:]).strip()
    else:
        try:
            target = input("Enter a URL to analyze with LinkGuard: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if target:
        evaluate_link(target)


if __name__ == "__main__":
    main()
