from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import app as student_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model():
    loaded, feature_names = student_app._load_model(student_app.MODEL_PATH)
    assert feature_names == list(loaded.feature_names_in_)
    return loaded


@pytest.fixture(scope="session")
def feature_names(model):
    return list(model.feature_names_in_)


@pytest.fixture(scope="session")
def student_data(feature_names):
    # Read only the lookup key and model inputs. Target is intentionally absent.
    return pd.read_csv(
        PROJECT_ROOT / "data" / "students_with_id.csv",
        usecols=[student_app.STUDENT_ID_COLUMN, *feature_names],
        dtype={student_app.STUDENT_ID_COLUMN: "string"},
    )


@pytest.fixture
def flask_app(model, student_data):
    return student_app.create_app(model=model, student_data=student_data)


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


@pytest.fixture(scope="session")
def valid_student_id(student_data):
    return str(student_data.iloc[0][student_app.STUDENT_ID_COLUMN])
