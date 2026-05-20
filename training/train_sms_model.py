import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from training.preprocess import clean_text


ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(ROOT_DIR, "data")

MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_sms_model():

    df = pd.read_csv(
        os.path.join(DATA_DIR, "sms_spam.csv")
    )

    df = df.rename(columns={
        "Message": "text",
        "spamORham": "label"
    })

    extra_ham = pd.DataFrame({
        "text": [
            "Your Amazon order has been shipped successfully",
            "Thank you for shopping with Amazon",
            "Your meeting is scheduled for tomorrow",
            "Invoice attached for your recent purchase",
            "Your package is out for delivery",
            "GitHub security alert for your repository",
            "Your LinkedIn profile appeared in searches",
            "Project update meeting moved to 3 PM",
            "Your Google verification code is 123456",
            "Netflix subscription renewed successfully",
            "Your payment receipt is attached",
            "Thank you for your payment",
            "Your delivery will arrive today",
            "Your booking has been confirmed",
            "Your interview is scheduled tomorrow",
            "Your OTP code is 458291",
            "Your account settings were updated",
            "Welcome to Microsoft Teams",
            "Your application has been received",
            "Password changed successfully"
        ],
        "label": [0] * 20
    })

    extra_spam = pd.DataFrame({
        "text": [
            "Verify your account immediately or it will be suspended",
            "Click here now to confirm your banking credentials",
            "Urgent action required to unlock your account",
            "You won a free iPhone click now",
            "Your password has expired verify immediately",
            "Claim your reward before it expires",
            "Limited time offer click here now",
            "Security alert login immediately",
            "Your account has been blocked",
            "Confirm your identity now",
            "Free crypto giveaway click here",
            "Bank account verification required",
            "Suspicious activity detected login now",
            "Reset your password immediately",
            "Your account will be terminated",
            "Act now to avoid suspension",
            "Click below to receive your reward",
            "Urgent verification required",
            "Login now to secure your account",
            "Your banking access has been blocked"
        ],
        "label": [1] * 20
    })

    df = pd.concat(
    [
        df[["text", "label"]],
        extra_ham,
        extra_spam
    ],
        ignore_index=True
    )

    df["label"] = df["label"].replace({
        "ham": 0,
        "spam": 1
    })

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    df["text"] = (
        df["text"]
        .astype(str)
        .apply(clean_text)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        C=0.3,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    y_proba = model.predict_proba(
        X_test_tfidf
    )[:, 1]

    print("\nSMS Model Results")
    print("-----------------")

    print(
        f"Accuracy: {accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"ROC AUC : {roc_auc_score(y_test, y_proba):.4f}"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    sms_artifact = {
        "tfidf": tfidf,
        "model": model
    }

    joblib.dump(
        sms_artifact,
        os.path.join(
            MODEL_DIR,
            "sms_pipeline.joblib"
        )
    )

    print("\nSaved sms_pipeline.joblib")


if __name__ == "__main__":
    train_sms_model()