# Student Outcome Prediction Using Machine Learning

## 1. Project overview

This college project explores whether a student's academic outcome can be classified after first-semester information is available. It compares several classification approaches, selects a model using training-only cross-validation, evaluates the selected model on a held-out test set, and provides a local Flask demonstration that looks up an existing record by Student ID.

The three outcome labels are **Dropout**, **Enrolled**, and **Graduate**. The app's output is a model estimate; it is not a verified outcome or a guarantee of a student's future.

## 2. Problem statement

Academic records contain admission, background, and early academic information that can be used to estimate a student's eventual recorded outcome. This project studies a three-class classification problem using features available after the first semester. Because the source dataset already contains final outcome labels, the evaluation is retrospective. It does not establish how well the model will predict outcomes for future students.

## 3. Project objectives

- Inspect and summarize the student dataset.
- Select model inputs available by the chosen prediction time.
- Keep the student identifier, outcome label, and later-semester results out of model features.
- Compare baseline and tuned classifiers using training-only validation.
- Select a model using macro F1, while considering class-specific recall and balanced accuracy.
- Evaluate the selected model once on a held-out test split.
- Demonstrate ID-based record lookup and inference through a local Flask app.

## 4. Key features

- One numeric Student ID is entered in the web form; academic marks or demographics are not entered manually.
- The app retrieves the matching row from the local CSV and passes only the model's 30 expected features to the saved pipeline.
- The saved preprocessing and classifier are loaded together from `models/student_outcome_pipeline.joblib`.
- The result page displays the predicted class and model-estimated class probabilities.
- Empty and malformed IDs, unknown IDs, duplicate matches, incomplete records, and inference errors are handled with explicit responses.
- Automated tests cover routes, feature eligibility and order, saved-model loading, and direct-versus-Flask prediction consistency.

## 5. Technology stack

- Python 3.13.16 (the version used for the documented evaluation environment)
- Flask 3.1.3 for the local web app
- NumPy 2.2.1, pandas 2.2.3, and SciPy 1.18.0 for data and numerical operations
- scikit-learn 1.6.1 for preprocessing pipelines, Logistic Regression, Random Forest, validation, and metrics
- XGBoost 3.4.1 for the evaluated XGBoost candidates
- joblib 1.4.2 for the saved pipeline
- pytest 8.4.2 for automated tests
- Jupyter Notebook for the training workflow
- HTML and CSS for the Flask pages

Exact Python package pins are separated by purpose: `requirements.txt` for the app/runtime, `requirements-ml.txt` for the evaluation dependency set, and `requirements-test.txt` for pytest.

## 6. Dataset overview

The application and evaluation script expect a local dataset at `data/students_with_id.csv`. The student-level CSV is intentionally not included in this repository. Provide an authorized copy locally before running the app, tests, notebook, or evaluation script. The dataset used for the documented results contains **4,424 records and 38 columns**, including a unique `Student_ID`, the `Target`, 30 eligible admission/background and first-semester features, and six second-semester fields.

Recorded target distribution:

| Recorded target | Records |
|---|---:|
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled | 794 |

These are historical labels in the dataset. They are used as the supervised training/evaluation target; the Flask app does not load or display a record's `Target` when making a prediction.

## 7. Data preprocessing and feature selection

The notebook is `notbooks/ML_project.ipynb` (the folder is named `notbooks` in this project). It defines the 30-feature input matrix by excluding:

- `Student_ID`, which is used only for lookup;
- `Target`, the label to be predicted; and
- all six columns beginning `Curricular units 2nd sem (`, which are later than the selected prediction time.

The 30 features consist of 17 coded categorical fields and 13 numeric fields. The pipeline one-hot encodes categorical inputs with unknown categories ignored. Logistic Regression scales the numeric inputs; tree models use unscaled numeric values. Preprocessing is inside the model pipelines, so it is fit only on the relevant training fold or training partition.

The inspected CSV has no missing values, exact duplicate rows, or duplicate Student IDs. The app still validates records and handles missing or ambiguous lookups rather than assuming these conditions will always hold.

## 8. Exploratory data analysis and model training

The notebook includes schema and missing-value checks, class distribution summaries, descriptive statistics, outcome plots, first- and second-semester boxplots, and a numeric correlation heatmap. Second-semester plots are descriptive of this historical dataset and their fields remain excluded from model inputs.

The training workflow uses a seeded, stratified **80/20 train/test split** (`random_state=42`): 3,539 training records and 885 held-out test records. The split is made before preprocessing or tuning, and the notebook asserts that train and test row indices do not overlap.

## 9. Candidate models and model-selection methodology

The comparison includes a majority-class dummy baseline; weighted and unweighted Logistic Regression, Random Forest, and XGBoost; and tuned Random Forest and XGBoost reference candidates. Random Forest and XGBoost hyperparameters are tuned with five-fold stratified cross-validation on the training data only, using macro F1. The candidate comparison uses shared five-fold stratified out-of-fold predictions on the training partition (`random_state=2026`). The held-out test set is not used for tuning or model selection.

Macro F1 is the primary model-selection metric. The documented one-standard-error procedure first identifies the candidate with the highest mean CV macro F1, then forms an eligible set within one standard error of that score. Among eligible candidates, the notebook prioritizes mean Enrolled recall, then mean balanced accuracy, then mean macro F1. Accuracy is not the sole selection criterion.

The highest CV macro F1 was **XGBoost with balanced sample weights** (0.6849). Balanced Logistic Regression scored 0.6803 and fell within the one-standard-error range (selection floor 0.6754). It had the strongest mean Enrolled recall among eligible candidates (0.5685), so the documented rule selected it.

## 10. Final model: balanced Logistic Regression

The active saved model is `models/student_outcome_pipeline.joblib`, a scikit-learn `Pipeline` containing the fitted `ColumnTransformer` and balanced `LogisticRegression` estimator. It expects the same 30 ordered features used by training and returns the original string labels `Dropout`, `Enrolled`, and `Graduate`.

Flask loads this artifact at startup. It does not train a replacement model during a request. The older `models/student_dropout_model.pkl` and `models/label_encoder.pkl` files are present in the repository but are not used by the current Flask app and are not validated as current artifacts.

## 11. Evaluation metrics and confusion matrix

The selected model was evaluated on the held-out test split after the training-only model-selection procedure was locked. Results below are copied from `MODEL_EVALUATION_RESULTS.md`.

| Metric | Held-out test result |
|---|---:|
| Accuracy | 69.94% |
| Macro precision | 66.52% |
| Macro recall | 66.77% |
| Macro F1 | 65.93% |
| Weighted F1 | 71.21% |

Confusion matrix (rows = actual class, columns = predicted class; order is **Dropout, Enrolled, Graduate**):

| Actual \\ Predicted | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Dropout | 194 | 63 | 27 |
| Enrolled | 28 | 89 | 42 |
| Graduate | 26 | 80 | 336 |

## 12. Per-class performance

| Class | Precision | Recall | F1-score | Test support |
|---|---:|---:|---:|---:|
| Dropout | 78.23% | 68.31% | 72.93% | 284 |
| Enrolled | 38.36% | 55.97% | 45.52% | 159 |
| Graduate | 82.96% | 76.02% | 79.34% | 442 |

**Enrolled is the weakest-performing class**, particularly in precision and F1. The model has both false positives and false negatives for this class. These metrics describe one held-out split of the available historical dataset and should not be interpreted as guarantees of future performance.

## 13. Limitations and future improvements

- The dataset contains final outcomes that are already known. The app uses historical records for a demonstration and is not a prospective student service.
- A requested Student ID must already exist in the CSV. Predictions for new students require an authorized source of their eligible feature records.
- Enrolled-class precision and F1 are materially lower than for Dropout and Graduate.
- The results represent one seeded holdout split. The tuned tree reference candidates were selected using training data and then compared through another training-only CV procedure; the untouched test split remains the final reported evaluation.
- The current project `.venv` has passed the test suite and loads the model, but some installed package versions differ from the pinned versions. A clean virtual-environment installation has **not** been verified.
- The working notebook's final experiment cells do not currently contain saved execution outputs. The evaluation script provides the reproducible route for regenerating the documented results; a Jupyter kernel did not start reliably during the prior verification.
- Future work could validate on an independent cohort, assess probability calibration and subgroup performance, and improve documentation of data provenance and deployment safeguards. Any new model selection should use training/validation data and a genuinely untouched final test set.

## 14. Project folder structure

Key project files (generated `.venv` and cache directories are omitted):

```text
D:\ML_project\
├── app.py
├── evaluate_model.py
├── README.md
├── TESTING.md
├── requirements.txt
├── requirements-ml.txt
├── requirements-test.txt
├── MODEL_EVALUATION_RESULTS.md
├── MODEL_INTEGRATION_CHECK.md
├── PROJECT_AUDIT.md
├── PROJECT_COMPLETION_AUDIT.md
├── models\
│   ├── student_outcome_pipeline.joblib
│   ├── student_dropout_model.pkl       # legacy; not used by current app
│   └── label_encoder.pkl              # legacy; not used by current app
├── notbooks\
│   ├── ML_project.ipynb
│   └── (notebook backup files are excluded)
├── static\
│   ├── css\style.css                 # current stylesheet
│   └── style.css                      # older stylesheet
├── templates\
│   ├── index.html
│   ├── result.html
│   └── results.html                   # empty legacy template
└── tests\
    ├── conftest.py
    └── test_app.py
```

## 15. Installation and setup on Windows

Use Python 3.13.16 for the closest match to the documented model environment. If `.venv` does not already exist, create it and activate it in PowerShell:

```powershell
cd D:\ML_project
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The student-level dataset is not stored in Git. Before running the app, tests, or evaluation script, place an authorized copy at `data\students_with_id.csv`.

Install only the Flask app/runtime requirements:

```powershell
python -m pip install -r requirements.txt
```

`requirements-ml.txt` includes the runtime requirements and adds XGBoost for evaluation. `requirements-test.txt` contains pytest for tests. See [TESTING.md](TESTING.md) for the dependency split and the note about the existing environment. The commands are documented, but a clean environment installation has not been verified.

## 16. How to run the Flask application

With the virtual environment activated and app dependencies installed:

```powershell
cd D:\ML_project
python app.py
```

Then open `http://127.0.0.1:5000/` in a browser. The local server binds to `127.0.0.1:5000` with Flask debug mode disabled.

## 17. How to run automated tests

With the virtual environment activated:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The suite uses pytest and covers app startup, home page, valid/invalid/unknown IDs, duplicate and incomplete records, error handling, saved-model loading, feature exclusions/order, and parity between direct pipeline predictions and Flask. Test fixtures use read-only project records; special cases use in-memory data. The latest recorded run passed 16 tests.

## 18. How to reproduce model evaluation

With the virtual environment activated:

```powershell
python -m pip install -r requirements-ml.txt
python evaluate_model.py
```

This script runs tuning and training-only candidate validation, applies the documented selection rule, evaluates the selected candidate on the held-out set, and regenerates `MODEL_EVALUATION_RESULTS.md`. It reads but does not overwrite `models/student_outcome_pipeline.joblib`. Evaluation is more computationally intensive than running the Flask app or tests.

## 19. Example prediction workflow

1. Start the local Flask app and open its home page.
2. Enter a numeric Student ID that exists in `data/students_with_id.csv`.
3. The app validates the ID and looks up exactly one matching record.
4. It selects the 30 eligible features in the saved pipeline's expected order and runs the complete preprocessing-and-classifier pipeline.
5. The result page shows the model-predicted class and, when available, estimated probabilities for Dropout, Enrolled, and Graduate.

The app does not display the record's historical `Target`. An unknown or ambiguous ID, or a record missing required inputs, produces an error rather than a substituted prediction.

## 20. Ethical considerations and appropriate use

This project is an educational demonstration. Student outcomes are sensitive, and historical patterns can reflect structural inequities or limitations in data collection. Predictions should not be used to deny opportunities, allocate support automatically, or make disciplinary or other high-impact decisions. A predicted class is not a measure of a student's ability, potential, or worth. Any real-world use would require appropriate authorization, privacy protections, independent validation, bias and calibration assessment, human oversight, and a current feature source. The displayed probabilities are model estimates, not guarantees.
