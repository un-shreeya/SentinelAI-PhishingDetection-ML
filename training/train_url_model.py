import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(ROOT_DIR, "data")

MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_url_model():

    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "phishing_urls.csv"
        )
    )

    X = df.drop(columns=["target"])

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_proba = pipeline.predict_proba(
        X_test
    )[:, 1]

    print("\nURL Model Results")
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

    artifact = {
        "model": pipeline,
        "feature_order": X.columns.tolist()
    }

    joblib.dump(
        artifact,
        os.path.join(
            MODEL_DIR,
            "url_pipeline.joblib"
        )
    )

    print("\nSaved url_pipeline.joblib")


if __name__ == "__main__":
    train_url_model()