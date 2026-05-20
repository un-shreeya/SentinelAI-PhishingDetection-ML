import os
import sys
import joblib
import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from training.preprocess import clean_text


MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)

SMS_PIPELINE_PATH = os.path.join(
    MODEL_DIR,
    "sms_pipeline.joblib"
)

URL_PIPELINE_PATH = os.path.join(
    MODEL_DIR,
    "url_pipeline.joblib"
)

sms_artifact = None
url_artifact = None


def load_models():

    global sms_artifact
    global url_artifact

    if sms_artifact is None:
        sms_artifact = joblib.load(
            SMS_PIPELINE_PATH
        )

    if (
        url_artifact is None
        and os.path.exists(URL_PIPELINE_PATH)
    ):
        url_artifact = joblib.load(
            URL_PIPELINE_PATH
        )

    return sms_artifact, url_artifact


def format_risk(prob):

    if prob >= 0.80:
        return "HIGH RISK"

    if prob >= 0.55:
        return "MEDIUM RISK"

    return "LOW RISK"


def predict_sms(text):

    sms_artifact, _ = load_models()

    cleaned = clean_text(text)

    tfidf = sms_artifact["tfidf"]

    model = sms_artifact["model"]

    X = tfidf.transform([cleaned])

    prob = float(
        model.predict_proba(X)[0][1]
    )

    lower = text.lower()

    reasons = []

    suspicious_keywords = [
        "verify",
        "password",
        "login",
        "credentials",
        "bank",
        "account",
        "click",
        "urgent",
        "suspended",
        "confirm",
        "blocked",
        "security",
        "reward",
        "winner",
        "crypto"
    ]

    dangerous_phrases = [
        "verify your account",
        "confirm your identity",
        "login immediately",
        "click here now",
        "account suspended",
        "confirm credentials",
        "verify immediately",
        "banking access",
        "security alert",
        "password expired",
        "login to continue",
        "unlock your account"
    ]

    transactional_patterns = [
        "thank you for shopping",
        "order confirmation",
        "delivery update",
        "invoice attached",
        "meeting scheduled",
        "payment receipt",
        "booking confirmed",
        "project update",
        "application received"
    ]

    suspicious_count = 0

    matched_keywords = []

    for word in suspicious_keywords:

        if word in lower:

            suspicious_count += 1

            matched_keywords.append(word)

    if matched_keywords:

        reasons.append(
            "Suspicious indicators: "
            + ", ".join(matched_keywords)
        )

    dangerous_hits = 0

    for phrase in dangerous_phrases:

        if phrase in lower:

            dangerous_hits += 1

    if dangerous_hits >= 1:

        prob += 0.20

        reasons.append(
            "Credential-theft style phrasing detected"
        )

    if dangerous_hits >= 2:

        prob += 0.15

    if "http" in lower or "www" in lower:

        prob += 0.08

        reasons.append(
            "Contains external links"
        )

    if text.count("!") >= 2:

        prob += 0.05

        reasons.append(
            "Aggressive punctuation"
        )

    uppercase_ratio = (
        sum(c.isupper() for c in text)
        / max(len(text), 1)
    )

    if uppercase_ratio > 0.30:

        prob += 0.05

        reasons.append(
            "Excessive uppercase usage"
        )

    legit_hits = 0

    for pattern in transactional_patterns:

        if pattern in lower:

            legit_hits += 1

    if legit_hits >= 1 and dangerous_hits == 0:

        prob -= 0.12

        reasons.append(
            "Legitimate transactional language detected"
        )

    prob = max(0.0, min(prob, 1.0))

    label = int(prob >= 0.60)

    if not reasons:

        reasons.append(
            "No strong phishing indicators detected"
        )

    return {
        "probability": round(prob, 4),
        "label": label,
        "risk": format_risk(prob),
        "explanation": " | ".join(reasons)
    }


def predict_url(features_dict):

    _, url_artifact = load_models()

    if url_artifact is None:

        return {
            "probability": 0.0,
            "label": 0,
            "risk": "UNKNOWN",
            "explanation": "URL model not loaded"
        }

    model = url_artifact["model"]

    feature_order = url_artifact[
        "feature_order"
    ]

    X = np.array([
        [
            features_dict.get(f, 0)
            for f in feature_order
        ]
    ])

    prob = float(
        model.predict_proba(X)[0][1]
    )

    reasons = []

    url = features_dict.get(
        "full_url",
        ""
    ).lower()

    suspicious_brands = [
        "amazon",
        "paypal",
        "microsoft",
        "google",
        "apple",
        "netflix",
        "bank",
        "facebook",
        "instagram"
    ]

    suspicious_words = [
        "verify",
        "login",
        "secure",
        "update",
        "freegift",
        "bonus",
        "reward",
        "claim",
        "signin",
        "confirm"
    ]

    dangerous_tlds = [
        ".ru",
        ".xyz",
        ".tk",
        ".top",
        ".gq",
        ".ml",
        ".cf"
    ]

    brand_hits = 0
    word_hits = 0

    for brand in suspicious_brands:

        if brand in url:

            brand_hits += 1

    for word in suspicious_words:

        if word in url:

            word_hits += 1

    tld_hit = any(
        tld in url
        for tld in dangerous_tlds
    )

    if brand_hits >= 1 and word_hits >= 1:

        prob += 0.30

        reasons.append(
            "Brand impersonation pattern detected"
        )

    if tld_hit:

        prob += 0.20

        reasons.append(
            "Suspicious domain extension detected"
        )

    if features_dict.get("has_ip", 0):

        prob += 0.15

        reasons.append(
            "Contains raw IP address"
        )

    if features_dict.get("has_at", 0):

        prob += 0.10

        reasons.append(
            "Contains @ symbol"
        )

    if features_dict.get(
        "url_length",
        0
    ) > 75:

        prob += 0.08

        reasons.append(
            "Unusually long URL"
        )

    if features_dict.get(
        "num_dots",
        0
    ) > 4:

        prob += 0.08

        reasons.append(
            "Too many subdomains"
        )

    if features_dict.get(
        "suspicious_kw_count",
        0
    ) > 0:

        prob += 0.10

        reasons.append(
            "Suspicious keywords detected"
        )

    prob = max(0.0, min(prob, 1.0))

    label = int(prob >= 0.60)

    if not reasons:

        reasons.append(
            "No strong phishing indicators detected"
        )

    return {
        "probability": round(prob, 4),
        "label": label,
        "risk": format_risk(prob),
        "explanation": " | ".join(reasons)
    }