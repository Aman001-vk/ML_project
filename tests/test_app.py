from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from flask import Flask

import app as student_app


EXPECTED_CLASSES = {"Dropout", "Graduate", "Enrolled"}


def displayed_prediction(response) -> str:
    page = response.get_data(as_text=True)
    match = re.search(r'<h1 id="result-title">\s*([^<]+?)\s*</h1>', page)
    assert match, "The result page did not display a predicted outcome."
    return match.group(1).strip()


def test_application_initializes_successfully():
    production_app = student_app.create_app()
    assert isinstance(production_app, Flask)
    assert production_app.extensions["model_path"] == student_app.MODEL_PATH
    assert production_app.extensions["student_model"] is not None
    assert len(production_app.extensions["feature_names"]) == 30


def test_saved_model_loads_and_has_expected_classes(model):
    loaded, feature_names = student_app._load_model(student_app.MODEL_PATH)
    assert loaded is not None
    assert len(feature_names) == 30
    assert set(loaded.named_steps["classifier"].classes_) == EXPECTED_CLASSES


def test_home_page_responds(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student Outcome Prediction System" in response.data
    assert b"student_id" in response.data


def test_valid_id_produces_a_legitimate_prediction(client, model, student_data, feature_names, valid_student_id):
    response = client.post("/predict", data={"student_id": valid_student_id})
    assert response.status_code == 200
    assert displayed_prediction(response) in EXPECTED_CLASSES


def test_flask_prediction_matches_direct_saved_pipeline(
    client, model, student_data, feature_names, valid_student_id
):
    record = student_data.loc[
        student_data[student_app.STUDENT_ID_COLUMN].astype(str) == valid_student_id,
        feature_names,
    ]
    expected_prediction = str(model.predict(record)[0])
    response = client.post("/predict", data={"student_id": valid_student_id})

    assert response.status_code == 200
    assert displayed_prediction(response) == expected_prediction


@pytest.mark.parametrize("student_id", ["", "12X", "-1", "1.5"])
def test_empty_or_malformed_ids_return_400(client, student_id):
    response = client.post("/predict", data={"student_id": student_id})
    assert response.status_code == 400


def test_nonexistent_id_returns_404(client):
    response = client.post("/predict", data={"student_id": "999999999"})
    assert response.status_code == 404
    assert b"No student record was found" in response.data


def test_duplicate_ids_return_409(model, student_data, valid_student_id):
    duplicate = pd.concat([student_data, student_data.iloc[[0]]], ignore_index=True)
    duplicate_app = student_app.create_app(model=model, student_data=duplicate)
    response = duplicate_app.test_client().post(
        "/predict", data={"student_id": valid_student_id}
    )
    assert response.status_code == 409


def test_missing_required_feature_returns_422(model, student_data, feature_names, valid_student_id):
    incomplete = student_data.copy()
    incomplete.loc[0, feature_names[0]] = np.nan
    incomplete_app = student_app.create_app(model=model, student_data=incomplete)
    response = incomplete_app.test_client().post(
        "/predict", data={"student_id": valid_student_id}
    )
    assert response.status_code == 422


def test_prediction_exception_returns_generic_500(model, student_data, feature_names, valid_student_id):
    class FailingModel:
        feature_names_in_ = np.asarray(feature_names, dtype=object)

        def predict(self, feature_frame):
            raise RuntimeError("intentional test failure")

    failing_app = student_app.create_app(model=FailingModel(), student_data=student_data)
    failing_app.logger.disabled = True
    response = failing_app.test_client().post(
        "/predict", data={"student_id": valid_student_id}
    )
    assert response.status_code == 500
    assert b"model could not process this record" in response.data


def test_id_target_and_second_semester_are_not_model_features(model, feature_names):
    headers = pd.read_csv(student_app.DATA_PATH, nrows=0).columns.tolist()
    second_semester = {
        name for name in headers if name.startswith(student_app.SECOND_SEMESTER_PREFIX)
    }
    forbidden = {student_app.STUDENT_ID_COLUMN, student_app.TARGET_COLUMN, *second_semester}

    assert len(feature_names) == 30
    assert not set(feature_names).intersection(forbidden)


def test_saved_feature_names_and_order_match_dataset_and_app(model, feature_names, flask_app):
    headers = pd.read_csv(student_app.DATA_PATH, nrows=0).columns.tolist()
    excluded = {
        student_app.STUDENT_ID_COLUMN,
        student_app.TARGET_COLUMN,
        *[name for name in headers if name.startswith(student_app.SECOND_SEMESTER_PREFIX)],
    }
    expected_order = [name for name in headers if name not in excluded]

    assert feature_names == expected_order
    assert flask_app.extensions["feature_names"] == feature_names


def test_flask_passes_features_to_model_in_saved_order(
    model, student_data, feature_names, valid_student_id
):
    class RecordingModel:
        def __init__(self, wrapped_model):
            self.wrapped_model = wrapped_model
            self.feature_names_in_ = wrapped_model.feature_names_in_
            self.named_steps = wrapped_model.named_steps
            self.received_columns = None

        def predict(self, feature_frame):
            self.received_columns = list(feature_frame.columns)
            return self.wrapped_model.predict(feature_frame)

        def predict_proba(self, feature_frame):
            return self.wrapped_model.predict_proba(feature_frame)

    recorder = RecordingModel(model)
    spy_app = student_app.create_app(model=recorder, student_data=student_data)
    response = spy_app.test_client().post(
        "/predict", data={"student_id": valid_student_id}
    )

    assert response.status_code == 200
    assert recorder.received_columns == feature_names
