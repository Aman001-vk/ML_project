# Student Outcome Prediction — Project Completion Audit

**Audit date:** 2026-10-29  
**Scope:** Read-only review of `D:\ML_project`. No model was retrained, and no existing project file was changed. This report is the only new file.

## Summary

The project has a functional ID-based Flask demo and a saved, leakage-aware 30-feature pipeline. The current artifact loads and predicts in both the system Python environment and the project `.venv`; Flask test-client requests pass for valid and invalid inputs. The main submission gaps are that the working notebook's final experiment cells are unexecuted in the saved notebook, there is no README or automated test suite, and the run-time environment does not match all pins in `requirements.txt`. The dataset contains historical final outcomes, so this should be presented as a retrospective demonstration rather than a live student risk service.

The working notebook is under the misspelled folder `notbooks/`. The active dataset is `data/students_with_id.csv`; the active pipeline is `models/student_outcome_pipeline.joblib`.

## 1. Dataset loading, cleaning, preprocessing, and EDA

**Status: Complete**

**Evidence:** The notebook resolves the CSV from the project root or notebook parent, validates the expected schema and labels, reports missing values and duplicates without silently deleting records, and confirms unique `Student_ID` values. Independent read-only inspection found 4,424 rows and 38 columns, all expected columns present, 0 missing values, 0 exact duplicate rows, 0 duplicate IDs, and an `int64` unique ID column. Target counts are Graduate 2,209, Dropout 1,421, and Enrolled 794. The notebook describes nominal coded fields as categorical, one-hot encodes 17 such features, and keeps 13 ordered/continuous features numeric. Numeric scaling is applied only for Logistic Regression. No cleaning transformation appears necessary for the current CSV.

EDA includes shape/schema checks, null counts, class counts/plot, descriptive statistics, first- and second-semester boxplots, and a numeric correlation heatmap. Second-semester EDA is descriptive; those columns are excluded from model features.

**Problem:** The numeric correlation heatmap includes `Student_ID`, which is an identifier and has no meaningful numeric interpretation. This does not make it a model input, but it can mislead readers. Some EDA summaries are repeated.

**Next action:** Exclude `Student_ID` from the correlation visualization and label any second-semester plots clearly as retrospective/descriptive. This is a presentation cleanup, not a modeling blocker.

## 2. Model training, comparison, tuning, and selection

**Status: Needs Verification**

**Evidence:** The notebook contains Logistic Regression, Random Forest, and XGBoost pipelines; preprocessing is included in each pipeline. Random Forest and XGBoost use `GridSearchCV` with seeded stratified five-fold CV and macro F1 scoring on `X_train`. Class-imbalance candidates are compared on the same training-only folds. The documented one-standard-error rule selects balanced Logistic Regression: the CV macro-F1 leader was weighted XGBoost (0.6849); the eligible Logistic Regression candidate had CV macro F1 0.6803, Enrolled recall 0.5685, and balanced accuracy 0.6875. The selection rule was documented before the one-time test evaluation.

The existing `PROJECT_AUDIT.md` reports that all notebook cells were sequentially executed in a clean Python process. However, in the saved working notebook, only 17 of 31 code cells have execution counts, and the experiment/selection/finalization cells 31–44 have no execution counts or outputs. The report's historical run is useful evidence, but the current notebook alone does not reproduce its results visibly. The audit could not independently re-run training because retraining was explicitly excluded from this read-only task.

**Problem:** The rule's prose mentions simplicity as a tie-breaker, but the code's actual sort keys are Enrolled recall, balanced accuracy, and macro F1; simplicity is not directly encoded. The current selection is nevertheless documented and asserts the selected name.

**Next action:** Before submission, run the saved notebook from a fresh kernel (if permitted) and preserve outputs or attach a run log; align the stated tie-break rule with the implemented sort keys. Do not use the held-out test set to revise selection.

## 3. Train/test split and leakage prevention

**Status: Complete**

**Evidence:** The notebook derives one feature matrix by excluding `Student_ID`, `Target`, and all six `Curricular units 2nd sem (...)` columns. It asserts 30 features and unique IDs. A seeded (`random_state=42`) stratified 80/20 split is made once: 3,539 train rows and 885 test rows. The notebook asserts the row indices are disjoint. All preprocessing is inside estimator pipelines, so transformers are fitted within training folds during CV and on training rows for the final fit. Hyperparameter tuning and candidate selection use only training data. The final selected candidate is scored once on the held-out test split in cell 39. The active artifact's 30 `feature_names_in_` exactly match the eligible dataset columns in notebook order; forbidden columns are absent.

**Problem:** The notebook's final cells are not executed/saved in the current `.ipynb`, limiting auditability of the split and training-only claims to source code and the prior audit record.

**Next action:** Preserve reproducibility evidence from an ordered fresh-kernel run. Keep the test split locked for future changes.

## 4. Evaluation metrics and class-wise performance

**Status: Needs Verification**

**Evidence:** The notebook code reports accuracy, macro and weighted precision/recall/F1, balanced accuracy, per-class precision/recall/F1/support, and a confusion matrix in `[Dropout, Enrolled, Graduate]` order. `PROJECT_AUDIT.md` records the selected pipeline's one-time held-out results:

| Metric | Test result |
|---|---:|
| Accuracy | 0.6994 |
| Macro precision | 0.6652 |
| Macro recall / balanced accuracy | 0.6677 |
| Macro F1 | 0.6593 |
| Weighted precision | 0.7343 |
| Weighted recall | 0.6994 |
| Weighted F1 | 0.7121 |

Reported test class results (precision / recall / F1, support): Dropout 0.78 / 0.68 / 0.73 (284); Enrolled 0.38 / 0.56 / 0.46 (159); Graduate 0.83 / 0.76 / 0.79 (442). The reported confusion matrix is `[[194, 63, 27], [28, 89, 42], [26, 80, 336]]` with rows actual and columns predicted in the same class order. These are taken from the existing audit document; the final notebook cells currently contain no saved outputs, and these results were not recomputed during this audit.

**Problem:** Enrolled precision and F1 remain substantially weaker than the other classes. The historical audit reports lower accuracy and macro F1 than the earlier unweighted Logistic Regression result, trading performance in those measures for improved Enrolled recall.

**Next action:** Include these limitations plainly in the submission. If retraining is later authorized, compare changes only through training-only CV and reserve a new untouched test set for any materially revised model.

## 5. Saved pipeline and training-to-Flask consistency

**Status: Complete**

**Evidence:** `models/student_outcome_pipeline.joblib` loads as a scikit-learn `Pipeline` with `ColumnTransformer` preprocessing and balanced `LogisticRegression`. It expects exactly 30 features and exposes the three classes `Dropout`, `Enrolled`, and `Graduate`. The fitted preprocessor expands these to 238 transformed features. The active Flask loader derives input order from `feature_names_in_`, verifies the expected 30 inputs/classes, and reads only `Student_ID` plus those features. `Target` and second-semester columns are not loaded by the serving lookup. The training feature order and inference order agree. A direct prediction for ID 1001 and the Flask test-client result both returned `Dropout`; the page contained probabilities.

Two older artifacts, `models/student_dropout_model.pkl` and `models/label_encoder.pkl`, remain in the project but are not referenced by the current Flask app and have not been validated for this pipeline.

**Problem:** Multiple model artifacts may confuse a future maintainer about which file is authoritative.

**Next action:** Identify the active pipeline explicitly in the README; leave legacy artifact cleanup until its provenance/use has been confirmed.

## 6. Flask prediction workflow, validation, and errors

**Status: Complete**

**Evidence:** `app.py` loads the pipeline and feature records once at startup. The user supplies only a numeric Student ID; the ID is a lookup key, not a model feature. The app normalizes leading zeros, rejects empty/non-digit IDs, returns 404 for an unknown ID, returns 409 for duplicate matches, and returns 422 if a selected record has a missing/invalid model feature. Inference errors and model artifact errors have explicit handling. The result view reports predicted class and probabilities, not the CSV's recorded `Target`. The app does not retrain, and it runs with debug disabled.

Read-only Flask test-client checks in this audit returned: home 200; valid ID 1001 200 (Dropout); empty ID 400; malformed ID 400; unknown ID 404; duplicate ID in an isolated in-memory app 409; missing required value in an isolated in-memory app 422. The current system environment successfully loaded the artifact and served the valid request.

**Problem:** The dataset is historical and already contains each student's actual final target. While the app does not display that target, a prediction for a known historical record is not a prospective outcome. The CSV is suitable for a local demonstration, not a current-student service.

**Next action:** Label the app clearly as a historical-data demonstration. Before any real deployment, provide an authorized current-student feature source and decide how actual outcomes will be distinguished from predictions.

## 7. Project structure and dependencies

**Status: Needs Verification**

**Evidence:** The project has `app.py`, `data/`, `models/`, `notbooks/`, `templates/`, `static/`, `requirements.txt`, and `PROJECT_AUDIT.md`. The application requirements pin Flask 3.1.3, joblib 1.4.2, NumPy 2.2.1, pandas 2.2.3, scikit-learn 1.6.1, and SciPy 1.18.0. System Python has those versions and passed inference. The project `.venv` currently has Flask 3.1.3, scikit-learn 1.6.1, joblib 1.6.0, NumPy 2.5.4, pandas 3.1.0, and SciPy 1.18.1. The artifact loaded and ID 1001 route returned HTTP 200 in that environment during this audit, but its installed versions do not match the pinned dependency file.

The project contains both current and legacy presentation files: `static/css/style.css` and `templates/result.html` are used by current routes; `static/style.css` is older, and `templates/results.html` is empty. No changes were made to them. The notebook also imports development/analysis packages such as XGBoost, SHAP, Matplotlib, and Seaborn; these are not in the Flask runtime `requirements.txt`, and there is no separate notebook/development dependency file.

**Problem:** Directory spelling `notbooks` is unconventional; the virtual environment is not reproducible from current pins without package alignment; the runtime requirements do not describe notebook dependencies. Legacy files are unexplained.

**Next action:** Create a README with exact environment setup and run commands; align `.venv` to `requirements.txt`; document notebook dependencies separately if the notebook must be reproducible. Do not remove legacy files without checking their history.

## 8. Existing tests, README, and documentation

**Status: Incomplete**

**Evidence:** No `README.md`, test directory, or automated test file is present in the project inventory. `PROJECT_AUDIT.md` records prior validation and test-client cases, and this audit repeated key Flask input cases without changing files. The notebook includes assertions for schema, feature exclusions, split disjointness, selection, artifact reload, and a prediction smoke test. In the saved notebook, later experiment/finalization cells do not carry execution outputs.

**Problem:** There is no simple submission guide or repeatable automated regression suite. Validation history is split across notebook state and audit prose.

**Next action:** Add a README and a small non-retraining test suite covering app startup, valid/invalid/missing/duplicate IDs, feature schema, and saved-pipeline prediction. Clearly mark tests that use the historical CSV.

## Remaining tasks before submission, prioritized

1. Add a README that explains project scope, historical-data limitation, environment setup, and exact Flask launch command.
2. Align the project `.venv` to the pinned runtime dependencies and rerun the application smoke checks.
3. Make notebook execution evidence complete and visible: run it in order when permitted, retain outputs or a run log, and correct the documented selection tie-breaker mismatch.
4. Add repeatable non-retraining app tests for valid, malformed, missing, duplicate, and incomplete records, plus model-feature consistency.
5. In the notebook, remove `Student_ID` from the numeric correlation plot and clearly label second-semester EDA as retrospective.
6. Document that live use needs authorized current-student features; do not describe the historical CSV demo as a prediction service for prospective/current students.
7. Document the active artifact and investigate the unused legacy model/template/style files before deciding whether any cleanup is warranted.
