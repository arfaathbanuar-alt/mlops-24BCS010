import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.pkl"


def main():
    # Generate reproducible synthetic student data
    rng = np.random.default_rng(42)
    n_samples = 1000

    study_hours = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(30, 100, n_samples)
    internal_marks = rng.uniform(0, 100, n_samples)

    # Generate synthetic Pass/Fail labels
    score = (
        study_hours * 4
        + attendance * 0.25
        + internal_marks * 0.45
        + rng.normal(0, 5, n_samples)
    )

    results = np.where(score >= 55, "Pass", "Fail")

    X = pd.DataFrame({
        "study_hours": study_hours,
        "attendance": attendance,
        "internal_marks": internal_marks,
    })

    y = results

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    print(f"Test accuracy on synthetic data: {accuracy:.2%}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()