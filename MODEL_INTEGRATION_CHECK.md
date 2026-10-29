# Model–Flask Integration Check

**Date:** 2026-10-30  
**Scope:** Read-only integration verification of `evaluate_model.py`, `MODEL_EVALUATION_RESULTS.md`, `models/student_outcome_pipeline.joblib`, and `app.py`. No models were retrained, and no existing project files were modified. This report is the only new file.

## Conclusion

**The evaluation script, saved pipeline, and Flask app use the same selected model: balanced Logistic Regression.** The evaluation results identify balanced Logistic Regression as the one-standard-error selection. The script's `MODEL_PATH` and Flask's `MODEL_PATH` both resolve to `D:\ML_project\models\student_outcome_pipeline.joblib`. Flask loads that pipeline at startup; it does not train a new model.

The evaluation results record that the saved artifact matched the fresh selected fit in input schema, held-out predictions, and Logistic Regression coefficients (absolute tolerance `1e-12`). This audit independently loaded the artifact and Flask-loaded model and confirmed their model structure and fitted state match.

## Model and preprocessing consistency

- Serialized estimator: scikit-learn `Pipeline` with `ColumnTransformer` preprocessing followed by `LogisticRegression`.
- Classifier configuration: `class_weight="balanced"`, `solver="newton-cholesky"`, `max_iter=2000`, `random_state=42`.
- Preprocessing: 13 numeric features use `StandardScaler`; 17 categorical fields use `OneHotEncoder(handle_unknown="ignore")`; unlisted columns are dropped.
- The evaluation script derives its feature list by excluding `Student_ID`, `Target`, and all columns beginning `Curricular units 2nd sem (`. The saved model's ordered `feature_names_in_` exactly matches the script-derived list and Flask's input list.
- The 30 ordered model features are: `Marital status`, `Application mode`, `Application order`, `Course`, `Daytime/evening attendance\t`, `Previous qualification`, `Previous qualification (grade)`, `Nacionality`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation`, `Admission grade`, `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `Age at enrollment`, `International`, `Curricular units 1st sem (credited)`, `Curricular units 1st sem (enrolled)`, `Curricular units 1st sem (evaluations)`, `Curricular units 1st sem (approved)`, `Curricular units 1st sem (grade)`, `Curricular units 1st sem (without evaluations)`, `Unemployment rate`, `Inflation rate`, and `GDP`.
- `Student_ID`, `Target`, and all six second-semester fields are absent from the estimator input names. Flask's in-memory lookup contains 31 columns (ID plus the 30 features); it does not load `Target` or second-semester fields.
- Model class order is `Dropout`, `Enrolled`, `Graduate` in both the artifact and Flask-loaded copy. Coefficients, intercepts, numeric scaler means, and one-hot category definitions matched between independently loaded instances.

## Direct and Flask predictions

Existing dataset records were selected to cover each recorded target class. Direct predictions from the serialized pipeline matched the predictions rendered by the Flask test client. The recorded target is shown only to identify which target-class record was tested; the app does not use or display it.

| Student ID | Recorded historical target | Direct prediction | Flask result | Dropout probability | Enrolled probability | Graduate probability |
|---|---|---|---|---:|---:|---:|
| 1001 | Dropout | Dropout | HTTP 200; matched | 66.1% | 12.6% | 21.4% |
| 1002 | Graduate | Dropout | HTTP 200; matched | 56.2% | 7.9% | 35.9% |
| 1017 | Enrolled | Dropout | HTTP 200; matched | 66.1% | 28.5% | 5.4% |

The probabilities are model estimates, not guarantees. The two prediction/target differences are model errors on historical examples; they do not indicate a discrepancy between direct inference and Flask. They are consistent with the evaluation report's weaker Enrolled performance (test precision 0.3836, recall 0.5597, F1 0.4552).

## Validation/error-path checks

Flask test-client checks actually run during this audit:

| Case | Result |
|---|---|
| Valid ID 1001 | HTTP 200; predicted class and probabilities rendered |
| Valid ID 1002 | HTTP 200; direct and Flask predictions matched |
| Valid ID 1017 | HTTP 200; direct and Flask predictions matched |
| Empty ID | HTTP 400 |
| Invalid ID format (`12X`) | HTTP 400 |
| Missing record (`999999999`) | HTTP 404 |
| Duplicate ID added to isolated in-memory test data | HTTP 409; no prediction |
| Required feature set missing in isolated in-memory test data | HTTP 422; no prediction |
| Deliberately failing estimator injected into an isolated app | HTTP 500 with generic inference error handling |

Duplicate, missing-feature, and prediction-exception checks used isolated in-memory app instances. The dataset and saved pipeline were not changed.

## Remaining issues

1. This app looks up historical student records whose final outcomes are already known in the source dataset. It is a demonstration, not a prospective service; future use requires a current, authorized feature source.
2. Predictions for IDs 1002 and 1017 differ from their recorded outcomes. Keep the result labeled as an ML estimate and present the measured class-wise limitations from `MODEL_EVALUATION_RESULTS.md`.
3. The application currently handles a failing estimator with a generic HTTP 500 response as intended. The diagnostic deliberately triggered this path; no failure was observed with the saved pipeline.

## Files changed

Only `MODEL_INTEGRATION_CHECK.md` was created. The evaluation script, results report, app, dataset, UI, and saved pipeline were read or tested without modification.
