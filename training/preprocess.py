import pandas as pd
import re
from urllib.parse import urlparse
import tldextract
import numpy as np

SUSPICIOUS_KEYWORDS = [
    "free", "urgent", "verify", "click", "limited", "account",
    "suspend", "password", "login", "confirm", "winner", "win",
    "prize", "offer", "claim", "security", "bank", "alert",
    "invoice", "order", "shipping", "track", "delivery"
]

def load_csv(path):
    return pd.read_csv(path)

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_text(text):
    return clean_text(text).lower()

def extract_url_features(url):

    out = {}

    try:

        domain = tldextract.extract(url)

        out["url_length"] = len(url)

        out["num_dots"] = url.count(".")

        out["has_at"] = int("@" in url)

        out["has_ip"] = int(
            bool(
                re.search(
                    r"\d+\.\d+\.\d+\.\d+",
                    url
                )
            )
        )

        out["domain_len"] = len(
            domain.domain or ""
        )

        out["tld"] = domain.suffix or ""

        out["suspicious_kw_count"] = sum(
            1
            for kw in SUSPICIOUS_KEYWORDS
            if kw in url.lower()
        )

    except Exception:

        out = {
            "url_length": 0,
            "num_dots": 0,
            "has_at": 0,
            "has_ip": 0,
            "domain_len": 0,
            "tld": "",
            "suspicious_kw_count": 0
        }

    out["full_url"] = url.lower()

    return out

def extract_message_features(texts):
    features = []
    for text in np.atleast_1d(texts):
        raw = clean_text(text)
        lower = raw.lower()

        urgency_count = sum(int(kw in lower) for kw in [
            "urgent", "verify", "password", "account", "login", "click",
            "free", "offer", "limited", "suspend", "security", "alert"
        ])
        exclaim = raw.count("!")
        digits = sum(c.isdigit() for c in raw)
        uppercase_ratio = sum(1 for c in raw if c.isupper()) / max(1, len(raw))
        url_hint = int(bool(re.search(r"(https?://|www\.|\.com|\.net|\.org)", lower)))
        length = len(raw)
        words = len(lower.split())
        has_from = int("from:" in lower)
        has_subject = int("subject:" in lower)
        has_thank_you = int("thank you" in lower)
        has_order = int("order" in lower or "invoice" in lower or "shipment" in lower)
        suspicious_count = sum(int(kw in lower) for kw in SUSPICIOUS_KEYWORDS)

        features.append([
            urgency_count,
            exclaim,
            digits,
            uppercase_ratio,
            url_hint,
            has_from,
            has_subject,
            has_thank_you,
            has_order,
            suspicious_count,
            length,
            words,
        ])
    return np.array(features, dtype=float)