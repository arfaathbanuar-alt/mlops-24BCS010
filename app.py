from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
model = joblib.load(MODEL_PATH)

# Change this to 2.0 in GitHub for the deployment demonstration.
MODEL_VERSION = "1.0"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    study_hours = ""
    attendance = ""
    internal_marks = ""

    if request.method == "POST":
        study_hours = request.form.get("study_hours", "").strip()
        attendance = request.form.get("attendance", "").strip()
        internal_marks = request.form.get("internal_marks", "").strip()

        try:
            hours = float(study_hours)
            attendance_value = float(attendance)
            marks = float(internal_marks)

            if not 0 <= hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= attendance_value <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= marks <= 100:
                raise ValueError("Internal marks must be between 0 and 100.")

            student = pd.DataFrame([{
                "study_hours": hours,
                "attendance": attendance_value,
                "internal_marks": marks,
            }])

            prediction = str(model.predict(student)[0])

        except ValueError as exc:
            error = str(exc) or "Please enter valid numeric values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        model_version=MODEL_VERSION,
        study_hours=study_hours,
        attendance=attendance,
        internal_marks=internal_marks,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)