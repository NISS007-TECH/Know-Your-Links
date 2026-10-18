
import hmac
import os
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from extract_features import (
    canonicalize_url,
    extract_features,
    get_registered_domain,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    BASE_DIR / "models" / "linkguard_gradient_boosting.joblib",
    BASE_DIR / "models" / "linkguard_model_v2.joblib",
    BASE_DIR / "models" / "linkguard_random_forest.joblib",
]

model = None
chosen_path = None

for path in MODEL_PATHS:
    if path.is_file():
        model = joblib.load(path)
        chosen_path = path
        break

if model is None:
    raise RuntimeError(
        "No trained model found. Expected a .joblib file in models/."
    )

app = FastAPI(
    title="LinkGuard URL Analysis API",
    version="1.0.0",
    description="ML-based URL classification. A prediction is not a safety guarantee.",
)


class URLRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


def require_api_key(x_api_key: str | None):
    expected = os.getenv("API_TOKEN")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
        )


def validate_url(raw_url: str) -> str:
    url = raw_url.strip()

    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    # Accept URLs with or without an explicit HTTP(S) scheme.
    candidate = url if "://" in url else "https://" + url

    try:
        parsed = urlsplit(candidate)
        # Accessing .port also detects malformed port values.
        _ = parsed.port
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed URL.")

    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail="Only HTTP and HTTPS URLs are supported.",
        )

    if not parsed.hostname or any(ch.isspace() for ch in url):
        raise HTTPException(status_code=400, detail="Invalid URL or hostname.")

    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(
            status_code=400,
            detail="URLs containing embedded credentials are not supported.",
        )

    return url


@app.get("/")
def root():
    return {
        "service": "LinkGuard URL Analysis API",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "modelLoaded": model is not None,
        "modelFile": chosen_path.name,
    }


@app.post("/predict")
def predict(
    request: URLRequest,
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    require_api_key(x_api_key)
    original_url = validate_url(request.url)

    try:
        canonical = canonicalize_url(original_url)
        features = extract_features(original_url)
        features_df = pd.DataFrame([features])

        # Keep feature order aligned with the training pipeline, if recorded.
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            missing = set(expected_features) - set(features_df.columns)

            if missing:
                raise RuntimeError(
                    "Feature extraction is missing trained features: "
                    + ", ".join(sorted(missing))
                )

            features_df = features_df[expected_features]

        prediction = str(model.predict(features_df)[0])
        probabilities = model.predict_proba(features_df)[0]
        classes = [str(label) for label in model.classes_]

        probability_breakdown = {
            label: round(float(probability) * 100, 2)
            for label, probability in zip(classes, probabilities)
        }

        confidence = round(float(max(probabilities)) * 100, 2)

        # Match the host/domain handling in the existing feature pipeline.
        host = urlsplit("//" + canonical).hostname or ""
        registered_domain = get_registered_domain(host)

        brand_impersonation = bool(
            features.get("brand_impersonation", 0)
        )

        # Preserve the existing script's suspicious-brand safety net.
        effective_verdict = prediction
        if brand_impersonation and prediction.lower() == "benign":
            effective_verdict = "suspicious"

        if effective_verdict.lower() == "benign":
            risk_level = "LOW"
            recommendation = (
                "The model classified this URL as benign. "
                "This does not guarantee that the URL is safe."
            )
        elif effective_verdict.lower() == "suspicious":
            risk_level = "MEDIUM"
            recommendation = (
                "Potential brand impersonation detected. Verify the exact "
                "domain and check independent security sources."
            )
        else:
            risk_level = "HIGH"
            recommendation = (
                "The model classified this URL as potentially malicious. "
                "Avoid entering credentials or sensitive information."
            )

        return {
            "provider": "LinkGuard ML",
            "status": "completed",
            "url": original_url,
            "canonicalUrl": canonical,
            "host": host,
            "registeredDomain": registered_domain,
            "modelFile": chosen_path.name,
            "modelPrediction": prediction,
            "verdict": effective_verdict.upper(),
            "riskLevel": risk_level,
            "confidence": confidence,
            "probabilities": probability_breakdown,
            "signals": {
                "brandImpersonation": brand_impersonation,
                "hasIpAddress": bool(features.get("has_ip", 0)),
                "suspiciousTld": bool(features.get("suspicious_tld", 0)),
                "isKnownTopDomain": bool(
                    features.get("is_known_top_domain", 0)
                ),
                "suspiciousKeywordCount": int(
                    features.get("suspicious_keyword_count", 0)
                ),
            },
            "recommendation": recommendation,
            "limitations": [
                "This is an ML prediction, not proof of malicious behavior.",
                "A benign prediction does not guarantee safety.",
                "Combine this result with independent threat-intelligence providers.",
            ],
        }

    except HTTPException:
        raise
    except Exception:
        # Avoid exposing internal paths, model details, or tracebacks to callers.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API server logs.",
        )
