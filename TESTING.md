# Environment setup and project commands

The package versions in `requirements.txt` match the saved pipeline's training/runtime environment: Python 3.13.16, Flask 3.1.3, joblib 1.4.2, NumPy 2.2.1, pandas 2.2.3, scikit-learn 1.6.1, and SciPy 1.18.0. `requirements.txt` is the Flask application/runtime dependency set and was left unchanged.

`requirements-ml.txt` includes the runtime requirements and adds XGBoost 3.4.1, the one extra dependency used by `evaluate_model.py` for its training-only tree-model comparison and tuning. The Flask app does not import XGBoost. `requirements-test.txt` contains pytest 8.4.2 for the automated tests.

| Package | Pinned set | Current project `.venv` |
|---|---:|---:|
| Python | 3.13.16 used for validation | 3.13.16 |
| Flask | 3.1.3 | 3.1.3 |
| Werkzeug (Flask dependency) | Resolved by Flask | 3.1.9 |
| Jinja2 (Flask dependency) | Resolved by Flask | 3.1.6 |
| itsdangerous (Flask dependency) | Resolved by Flask | 2.2.0 |
| click (Flask dependency) | Resolved by Flask | 8.5.0 |
| blinker (Flask dependency) | Resolved by Flask | 1.9.0 |
| MarkupSafe (Flask dependency) | Resolved by Flask | 3.0.4 |
| NumPy | 2.2.1 | 2.5.4 |
| pandas | 2.2.3 | 3.1.0 |
| SciPy | 1.18.0 | 1.18.1 |
| scikit-learn | 1.6.1 | 1.6.1 |
| joblib | 1.4.2 | 1.6.0 |
| XGBoost | 3.4.1 (ML only) | 3.4.1 |
| pytest | 8.4.2 (test only) | 8.4.2 |

## Create and activate a virtual environment

The project was checked on Python 3.13.16. If `.venv` does not already exist, create it with an installed Python 3.13 interpreter:

```powershell
cd D:\ML_project
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install application dependencies and start Flask

```powershell
python -m pip install -r requirements.txt
python app.py
```

## Install test dependencies and run tests

With the virtual environment activated:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The suite uses the saved model and read-only dataset records. It does not retrain the model or compare predictions with historical `Target` values. Duplicate, incomplete, and failing-model cases use in-memory test data.
The student-level CSV is not included in the repository; provide an authorized copy at `data/students_with_id.csv` before running the suite.

## Install evaluation dependencies and reproduce model evaluation

With the virtual environment activated:

```powershell
python -m pip install -r requirements-ml.txt
python evaluate_model.py
```

The evaluation command performs model fitting and validation, evaluates the selected candidate on the held-out split, and rewrites `MODEL_EVALUATION_RESULTS.md`. It does not overwrite the saved Flask pipeline.

## Verification notes

The existing project `.venv` currently passes the 16-test suite, loads the saved pipeline without a scikit-learn version warning, and reports no broken requirements via `pip check`. Its NumPy, pandas, SciPy, and joblib versions differ from the pinned set. The pinned set matches the environment used to train and evaluate the current artifact. A clean virtual-environment installation was not performed during this dependency audit, so these instructions should not be described as clean-install verified yet.
