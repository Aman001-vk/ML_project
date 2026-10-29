"""ID-only Flask interface for the saved student outcome pipeline."""

from __future__ import annotations

import math
import re
import secrets
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "student_outcome_pipeline.joblib"
DATA_PATH = PROJECT_DIR / "data" / "students_with_id.csv"

STUDENT_ID_COLUMN = "Student_ID"
TARGET_COLUMN = "Target"
EXPECTED_CLASSES = {"Graduate", "Dropout", "Enrolled"}
SECOND_SEMESTER_PREFIX = "Curricular units 2nd sem ("
ID_PATTERN = re.compile(r"[0-9]{1,20}\Z")
DISPLAY_GROUPS = {
    "Admission information": {
        "Previous qualification (grade)": "Previous qualification grade",
        "Admission grade": "Admission grade",
    },
    "First semester": {
        "Curricular units 1st sem (credited)": "Units credited",
        "Curricular units 1st sem (enrolled)": "Units enrolled",
        "Curricular units 1st sem (evaluations)": "Evaluations",
        "Curricular units 1st sem (approved)": "Units approved",
        "Curricular units 1st sem (grade)": "Average grade",
        "Curricular units 1st sem (without evaluations)": "Units without evaluations",
    },
    "Second semester · historical, not used by model": {
        "Curricular units 2nd sem (credited)": "Units credited",
        "Curricular units 2nd sem (enrolled)": "Units enrolled",
        "Curricular units 2nd sem (evaluations)": "Evaluations",
        "Curricular units 2nd sem (approved)": "Units approved",
        "Curricular units 2nd sem (grade)": "Average grade",
        "Curricular units 2nd sem (without evaluations)": "Units without evaluations",
    },
}
DISPLAY_COLUMNS = {
    column for fields in DISPLAY_GROUPS.values() for column in fields
} | {TARGET_COLUMN}


def _normalise_id(value: Any) -> str | None:
    """Return a canonical numeric ID, or None for a malformed/empty value."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not ID_PATTERN.fullmatch(text):
        return None
    return str(int(text))


def _format_display_value(value: Any) -> str:
    """Format genuine dataset values for display without exposing missing markers."""
    if pd.isna(value):
        return "Not available"
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _load_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved model pipeline was not found: {model_path}. "
            "The application will not train a replacement model."
        )
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the saved model pipeline at {model_path}: {exc}"
        ) from exc

    feature_names = list(getattr(model, "feature_names_in_", []))
    if len(feature_names) != 30:
        raise RuntimeError(
            f"The saved pipeline must expect exactly 30 features; found {len(feature_names)}."
        )
    forbidden = {STUDENT_ID_COLUMN, TARGET_COLUMN}
    forbidden.update(
        name for name in feature_names if name.startswith(SECOND_SEMESTER_PREFIX)
    )
    used_forbidden = forbidden.intersection(feature_names)
    if used_forbidden:
        raise RuntimeError(
            "The saved pipeline includes ineligible inputs: "
            + ", ".join(sorted(used_forbidden))
        )

    classifier = getattr(model, "named_steps", {}).get("classifier")
    classes = set(getattr(classifier, "classes_", []))
    if classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "The saved pipeline classes do not match Graduate, Dropout, and Enrolled."
        )
    return model, feature_names


def _load_student_records(data_path: Path, feature_names: list[str]) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(f"Student dataset was not found: {data_path}")

    try:
        available_columns = set(pd.read_csv(data_path, nrows=0).columns)
    except Exception as exc:
        raise RuntimeError(f"Could not read the student dataset header: {exc}") from exc

    required_columns = {STUDENT_ID_COLUMN, *feature_names}
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise RuntimeError(
            "The student dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    display_columns = [column for column in DISPLAY_COLUMNS if column in available_columns]
    # Read the lookup key, model inputs, and a minimal whitelist of relevant
    # academic fields. Target is kept separate from the model feature frame.
    try:
        students = pd.read_csv(
            data_path,
            usecols=[STUDENT_ID_COLUMN, *feature_names, *display_columns],
            dtype={STUDENT_ID_COLUMN: "string"},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not read the student feature records: {exc}") from exc

    invalid_ids = students[STUDENT_ID_COLUMN].map(_normalise_id).isna()
    if invalid_ids.any():
        raise RuntimeError(
            f"The dataset contains {int(invalid_ids.sum())} invalid or empty Student_ID values."
        )

    # Coerce malformed numeric cells to missing values so the requested record
    # can be rejected gracefully instead of reaching the estimator.
    for column in feature_names:
        students[column] = pd.to_numeric(students[column], errors="coerce")

    students["_lookup_id"] = students[STUDENT_ID_COLUMN].map(_normalise_id)
    return students


def create_app(
    *,
    model_path: Path = MODEL_PATH,
    data_path: Path = DATA_PATH,
    model=None,
    student_data: pd.DataFrame | None = None,
) -> Flask:
    """Create the app; optional model/data arguments support isolated route checks."""
    app = Flask(__name__)
    model_path = Path(model_path)
    data_path = Path(data_path)

    if model is None:
        model, feature_names = _load_model(model_path)
    else:
        feature_names = list(getattr(model, "feature_names_in_", []))
        if len(feature_names) != 30:
            raise RuntimeError("Injected model must expect exactly 30 features.")
        if STUDENT_ID_COLUMN in feature_names or TARGET_COLUMN in feature_names:
            raise RuntimeError("Injected model includes an ineligible input column.")
        if any(name.startswith(SECOND_SEMESTER_PREFIX) for name in feature_names):
            raise RuntimeError("Injected model includes a second-semester input.")

    if student_data is None:
        students = _load_student_records(data_path, feature_names)
        display_columns = [column for column in DISPLAY_COLUMNS if column in students.columns]
    else:
        missing_columns = {STUDENT_ID_COLUMN, *feature_names} - set(student_data.columns)
        if missing_columns:
            raise RuntimeError(
                "The student data is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        # Keep only the ID, eligible inputs and explicitly whitelisted display
        # fields, even for injected data. Display fields never enter inference.
        display_columns = [column for column in DISPLAY_COLUMNS if column in student_data.columns]
        selected_columns = list(dict.fromkeys([STUDENT_ID_COLUMN, *feature_names, *display_columns]))
        students = student_data.loc[:, selected_columns].copy()
        students[STUDENT_ID_COLUMN] = students[STUDENT_ID_COLUMN].astype("string")
        for column in feature_names:
            students[column] = pd.to_numeric(students[column], errors="coerce")
        students["_lookup_id"] = students[STUDENT_ID_COLUMN].map(_normalise_id)

    if students["_lookup_id"].isna().any():
        raise RuntimeError("The student dataset contains invalid or empty Student_ID values.")

    lookup: dict[str, pd.DataFrame] = {
        str(key): group.drop(columns="_lookup_id")
        for key, group in students.groupby("_lookup_id", sort=False)
    }

    app.extensions["student_model"] = model
    app.extensions["feature_names"] = feature_names
    app.extensions["display_columns"] = display_columns
    app.extensions["student_lookup"] = lookup
    app.extensions["model_path"] = model_path

    def render_home(*, error: str | None = None, student_id: str = ""):
        return render_template(
            "index.html",
            error=error,
            student_id=student_id,
            outcome=None,
            prediction_event_id=secrets.token_urlsafe(18),
        )

    @app.get("/")
    def home():
        return render_home()

    @app.post("/predict")
    def predict():
        submitted_id = request.form.get("student_id", "").strip()
        if not submitted_id:
            return render_home(error="Enter a Student ID to continue."), 400

        normalized_id = _normalise_id(submitted_id)
        if normalized_id is None:
            return render_home(
                error="Student ID must contain digits only (up to 20 digits).",
                student_id=submitted_id,
            ), 400

        matching_rows = lookup.get(normalized_id)
        if matching_rows is None:
            return render_home(
                error="No student record was found for that ID.",
                student_id=submitted_id,
            ), 404
        if len(matching_rows) != 1:
            return render_home(
                error="This ID matches multiple records. No prediction was generated.",
                student_id=submitted_id,
            ), 409

        feature_row = matching_rows.loc[:, feature_names].iloc[[0]].copy()
        if feature_row.isna().any(axis=None):
            return render_home(
                error=(
                    "This student record is missing information required by the model. "
                    "No prediction was generated."
                ),
                student_id=submitted_id,
            ), 422
        if not all(
            math.isfinite(float(value))
            for value in feature_row.iloc[0].tolist()
        ):
            return render_home(
                error=(
                    "This student record contains invalid model inputs. "
                    "No prediction was generated."
                ),
                student_id=submitted_id,
            ), 422

        try:
            pipeline = app.extensions["student_model"]
            prediction = str(pipeline.predict(feature_row)[0])
            probabilities = None
            if hasattr(pipeline, "predict_proba"):
                raw_probabilities = pipeline.predict_proba(feature_row)[0]
                classes = pipeline.named_steps["classifier"].classes_
                probabilities = [
                    {"name": str(class_name), "percent": round(float(probability) * 100, 1)}
                    for class_name, probability in zip(classes, raw_probabilities)
                ]
        except Exception:
            # Do not log the submitted ID or student feature values.
            app.logger.exception("Prediction failed for a student record.")
            return render_home(
                error="The model could not process this record. Please try again later.",
                student_id=submitted_id,
            ), 500

        if prediction not in EXPECTED_CLASSES:
            app.logger.error("The saved model returned an unexpected class label.")
            return render_home(
                error="The model returned an unsupported outcome label.",
                student_id=submitted_id,
            ), 500

        record = matching_rows.iloc[0]
        academic_sections = []
        for section_name, fields in DISPLAY_GROUPS.items():
            section_values = [
                {"label": label, "value": _format_display_value(record[column])}
                for column, label in fields.items()
                if column in matching_rows.columns
            ]
            if section_values:
                academic_sections.append({"name": section_name, "fields": section_values})
        recorded_outcome = None
        if TARGET_COLUMN in matching_rows.columns:
            target_value = record[TARGET_COLUMN]
            if not pd.isna(target_value):
                recorded_outcome = str(target_value)

        prediction_event_id = request.form.get("prediction_event_id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", prediction_event_id):
            prediction_event_id = secrets.token_urlsafe(18)

        return render_template(
            "result.html",
            student_id=submitted_id,
            outcome=prediction,
            probabilities=probabilities,
            academic_sections=academic_sections,
            recorded_outcome=recorded_outcome,
            prediction_event_id=prediction_event_id,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
