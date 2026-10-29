# Project Audit

## Scope and dataset

Inspected `notbooks/ML_project.ipynb`, its STEP 3 and STEP 4 backups, `data/students_with_id.csv`, Flask scaffold files, and installed package versions. The actual notebook folder is named `notbooks`.

- CSV: 4,424 rows, 38 columns; no missing values, exact duplicates, duplicate IDs, or nonnumeric predictors.
- Full target counts: Dropout 1,421; Enrolled 794; Graduate 2,209.
- Prediction time: after first-semester information is available. The 30 model features exclude `Student_ID`, `Target`, and all six second-semester fields.
- Exact feature order: `Marital status`, `Application mode`, `Application order`, `Course`, `Daytime/evening attendance\t`, `Previous qualification`, `Previous qualification (grade)`, `Nacionality`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation`, `Admission grade`, `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `Age at enrollment`, `International`, `Curricular units 1st sem (credited)`, `Curricular units 1st sem (enrolled)`, `Curricular units 1st sem (evaluations)`, `Curricular units 1st sem (approved)`, `Curricular units 1st sem (grade)`, `Curricular units 1st sem (without evaluations)`, `Unemployment rate`, `Inflation rate`, `GDP`.
- The CSV is historical and contains known outcomes. ID-only application predictions require a current feature lookup source; an actual recorded outcome must not be displayed as a model prediction.

## STEP 4 completed

The STEP 4 notebook backup is `notbooks/ML_project.step4_backup.ipynb`. One reproducible stratified split was established (3,539 train / 885 test, zero row overlap). Preprocessing is inside pipelines: 17 coded categorical features are one-hot encoded; 13 numeric features are scaled for Logistic Regression and passed through unscaled for tree models. Random Forest and XGBoost tuning used only training data and seeded five-fold stratified CV. The prior Logistic Regression optimizer warning was fixed with `newton-cholesky`. SHAP class-specific plots ran for the three classes. No final model was saved.

STEP 4 measured held-out results are retained in history but were not used in the STEP 5 class-weight comparison. The Jupyter kernel launcher hangs before a minimal cell responds; STEP 4 cells were instead run sequentially in a clean Python process. EDA has a harmless font warning due to a tab in one column label.

## STEP 5 experiment

The STEP 4 notebook was preserved before editing as `notbooks/ML_project.step4_backup.ipynb`. The working notebook removes the repeated held-out test evaluation. Test data is created by the original split for later one-time final evaluation, but no model fit, preprocessing fit, cross-validation score, class-weight calculation, classification report, or confusion matrix in STEP 5 reads it.

Compared nine candidates on the same new `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` folds over the training partition: majority-class dummy; unweighted and balanced Logistic Regression; unweighted and balanced Random Forest; STEP 4 tuned Random Forest reference; unweighted and training-fold-weighted XGBoost; STEP 4 tuned XGBoost reference. Weighting strategies and all preprocessing remain inside fold training; XGBoost sample weights are computed separately from each fold’s training labels. No SMOTE or resampling was used. OOF classification reports and confusion matrices are generated for all candidates in `[Dropout, Enrolled, Graduate]` order.

Training-partition class counts: Dropout 1,137; Enrolled 635; Graduate 1,767.

| Candidate | CV macro F1 mean | CV balanced accuracy mean | Enrolled recall mean |
|---|---:|---:|---:|
| XGBoost balanced sample weights | 0.6849 | 0.6878 | 0.5449 |
| Logistic Regression balanced | 0.6803 | 0.6875 | 0.5685 |
| Random Forest tuned reference | 0.6779 | 0.6714 | 0.4157 |
| XGBoost tuned reference | 0.6754 | 0.6672 | 0.3669 |
| Logistic Regression unweighted | 0.6693 | 0.6613 | 0.3244 |
| XGBoost unweighted | 0.6652 | 0.6574 | 0.3323 |
| Random Forest balanced | 0.6344 | 0.6307 | 0.2142 |
| Random Forest unweighted | 0.6298 | 0.6288 | 0.1969 |
| Majority dummy | 0.2220 | 0.3333 | 0.0000 |

Weighted-minus-unweighted changes averaged across the same folds:

| Model family | Macro F1 change | Balanced accuracy change | Enrolled recall change |
|---|---:|---:|---:|
| Logistic Regression | +0.0110 | +0.0262 | +0.2441 |
| Random Forest | +0.0046 | +0.0019 | +0.0173 |
| XGBoost (fold-specific sample weights) | +0.0197 | +0.0305 | +0.2126 |

OOF class reports show the trade-off. For Logistic Regression, Enrolled precision/recall/F1 changed from 0.538/0.324/0.405 unweighted to 0.423/0.569/0.485 balanced. For XGBoost it changed from 0.490/0.332/0.396 to 0.437/0.545/0.485 with balanced sample weights. Random Forest balancing changed Enrolled from 0.576/0.197/0.293 to 0.548/0.214/0.308. Per-class support is 1,137 Dropout, 635 Enrolled, and 1,767 Graduate. The notebook prints the full precision/recall/F1/support reports and confusion matrices for every candidate.

Class weighting improved Enrolled recall substantially for Logistic Regression and XGBoost, with lower Enrolled precision and reduced recall in other classes. Random Forest gains were small. XGBoost balanced weighting had the highest mean CV macro F1 in this experiment, while Logistic Regression balanced had slightly higher Enrolled recall; these are training-only candidates, not a final model selection. The previous STEP 4 tuned tree scores are references from an earlier CV search and were reevaluated here on new training-only folds.

### Training out-of-fold per-class reports

The following are precision/recall/F1 values from pooled OOF predictions; supports are Dropout 1,137, Enrolled 635, Graduate 1,767 for each candidate. Column order within each class entry is P/R/F1.

| Candidate | Dropout P/R/F1 | Enrolled P/R/F1 | Graduate P/R/F1 |
|---|---:|---:|---:|
| Majority dummy | 0.000/0.000/0.000 | 0.000/0.000/0.000 | 0.499/1.000/0.666 |
| Logistic Regression unweighted | 0.779/0.756/0.768 | 0.538/0.324/0.405 | 0.778/0.903/0.836 |
| Logistic Regression balanced | 0.796/0.706/0.748 | 0.423/0.569/0.485 | 0.830/0.788/0.808 |
| Random Forest unweighted | 0.783/0.756/0.770 | 0.576/0.197/0.293 | 0.741/0.933/0.826 |
| Random Forest balanced | 0.795/0.744/0.769 | 0.548/0.214/0.308 | 0.741/0.934/0.826 |
| Random Forest tuned reference | 0.792/0.733/0.761 | 0.484/0.416/0.447 | 0.788/0.866/0.825 |
| XGBoost unweighted | 0.791/0.731/0.760 | 0.490/0.332/0.396 | 0.781/0.909/0.840 |
| XGBoost balanced sample weights | 0.805/0.696/0.747 | 0.437/0.545/0.485 | 0.823/0.823/0.823 |
| XGBoost tuned reference | 0.787/0.735/0.760 | 0.504/0.367/0.425 | 0.789/0.899/0.840 |

Confusion matrices below use rows=actual, columns=predicted, with class order `[Dropout, Enrolled, Graduate]`:

- Majority dummy: `[[0, 0, 1137], [0, 0, 635], [0, 0, 1767]]`
- Logistic Regression unweighted: `[[860, 97, 180], [153, 206, 276], [91, 80, 1596]]`
- Logistic Regression balanced: `[[803, 210, 124], [113, 361, 161], [93, 282, 1392]]`
- Random Forest unweighted: `[[860, 54, 223], [158, 125, 352], [80, 38, 1649]]`
- Random Forest balanced: `[[846, 70, 221], [143, 136, 356], [75, 42, 1650]]`
- Random Forest tuned reference: `[[833, 134, 170], [130, 264, 241], [89, 148, 1530]]`
- XGBoost unweighted: `[[831, 125, 181], [154, 211, 270], [66, 95, 1606]]`
- XGBoost balanced sample weights: `[[791, 209, 137], [114, 346, 175], [77, 236, 1454]]`
- XGBoost tuned reference: `[[836, 124, 177], [153, 233, 249], [73, 105, 1589]]`

## Final model selection and one-time test evaluation

Selection rule was documented in the notebook before looking at the final test metrics:

1. Rank candidates by mean training-only five-fold macro F1.
2. Define the one-standard-error set using the top candidate’s fold-score standard error (`sample SD / sqrt(5)`).
3. Within the set, prefer higher mean Enrolled recall, then balanced accuracy, then the simpler model.

The highest mean CV macro F1 was XGBoost with fold-specific balanced sample weights (0.6849). Its one-SE allowance was 0.0095, giving an eligibility floor of 0.6754. Balanced Logistic Regression (CV macro F1 0.6803) qualified; it had the highest mean Enrolled recall among eligible candidates (0.5685), with nearly the same balanced accuracy as the CV leader (0.6875 versus 0.6878), and is the simpler model. It was selected without consulting test metrics.

The selected pipeline was refit on all 3,539 training rows and evaluated once on the existing 885-row test split:

| Metric | Test result |
|---|---:|
| Accuracy | 0.6994 |
| Macro precision | 0.6652 |
| Macro recall | 0.6677 |
| Macro F1 | 0.6593 |
| Balanced accuracy | 0.6677 |
| Weighted precision | 0.7343 |
| Weighted recall | 0.6994 |
| Weighted F1 | 0.7121 |

Per-class test report: Dropout precision/recall/F1 = 0.78/0.68/0.73 (support 284); Enrolled = 0.38/0.56/0.46 (support 159); Graduate = 0.83/0.76/0.79 (support 442). Test confusion matrix, rows=actual and columns=predicted in `[Dropout, Enrolled, Graduate]` order: `[[194, 63, 27], [28, 89, 42], [26, 80, 336]]`.

Compared with the prior unweighted Logistic Regression result on the same split (accuracy 0.7492, macro F1 0.6680, Enrolled recall 0.33), balancing improved Enrolled recall to 0.56, while accuracy and macro F1 decreased. This is the measured trade-off; selection remains based on the documented training-only procedure.

The complete preprocessing-and-classifier pipeline was saved as `models/student_outcome_pipeline.joblib`. Reload validation confirmed 30 expected feature names and the three original string class labels. A prediction from one eligible test record succeeded (`Graduate`). No separate encoder or metadata artifact was necessary. Older `student_dropout_model.pkl` and `label_encoder.pkl` files were not overwritten and remain unvalidated for the current pipeline.

## Validation and remaining limitations

- The current STEP 5 notebook was backed up as `notbooks/ML_project.step5_backup.ipynb` before the finalization edits.
- All 46 notebook cells ran in order in a fresh Python process. Training-only CV completed, the documented selection rule selected Logistic Regression with balanced weights, and the selected model was evaluated once on the held-out test split.
- The pipeline was saved, loaded in the notebook process, and tested on an eligible test record. A second independent Python process also loaded the artifact, verified its 30 input feature names and three string classes, and predicted `Graduate`; probabilities were Dropout 0.0573, Enrolled 0.2959, Graduate 0.6467.
- Notebook JSON and all code cell syntax checks pass. A fresh Jupyter kernel still could not be confirmed because the kernel launcher hangs before executing a minimal cell. The sequential Python execution completed without model or SHAP errors. The tab-character EDA font warning is nonfatal.
- Test accuracy and macro F1 are below the prior unweighted Logistic Regression result, while Enrolled recall increased from 0.33 to 0.56. This trade-off is reported without using the test result to change the selection.
- The prediction target is historical and the current CSV contains already-known outcomes. A live ID-only application still needs an appropriate source of current student features; it must not present recorded outcomes as predictions.
- Existing legacy pickle artifacts were not overwritten and are not validated for the current pipeline. Flask implementation and validation are documented below.

## Remaining work

- User review of the selected pipeline, its one-time test results, and the artifact.
- Define a current-student feature lookup source before any live deployment; the current Flask demo looks up the historical CSV.
- Keep future model changes independent of the held-out test results; do not retrain or reselect based on that test split.

## Flask application implementation and validation

Created/updated `app.py`, `templates/index.html`, `templates/result.html`, and `static/css/style.css`; added runtime pins in `requirements.txt`. The previous `templates/results.html` and `static/style.css` files were left in place but are no longer used by the new routes.

- Startup loads `models/student_outcome_pipeline.joblib` once and verifies its 30 input names and three string labels. It does not retrain.
- The CSV is read relative to the project root. Only `Student_ID` and the 30 feature columns are loaded; `Target` and all semester-two columns are not loaded by the serving code.
- Raw CSV `Student_ID` is `int64`; the app reads it as text to preserve exact ID handling, validates numeric input, and canonicalizes leading zeros.
- The feature row is selected in `pipeline.feature_names_in_` order. Missing/nonfinite feature values, invalid IDs, missing IDs, duplicate IDs, inference errors, and missing artifacts receive clear errors. Ambiguous IDs never fall through to another record.
- The result page shows only submitted ID, predicted label, and model probability estimates. It never displays the CSV’s `Target`.

Runtime package versions are pinned to the environment that created and loaded the pipeline: Flask 3.1.3, joblib 1.4.2, NumPy 2.2.1, pandas 2.2.3, scikit-learn 1.6.1, and SciPy 1.18.0. The existing project `.venv` currently has Flask missing and incompatible versions (pandas 3.1.0, scikit-learn 1.9.1, NumPy 2.5.4, joblib 1.6.0). Loading the artifact there failed with `AttributeError: _RemainderColsList`; using `requirements.txt` in that environment will align its model runtime versions. The system Python used for model creation already has the pinned versions and Flask, and it was used for functional route tests.

Actual functional checks using Flask’s test client and existing CSV records:

| Case | Result |
|---|---|
| Home page | HTTP 200 |
| Valid ID `1001` | HTTP 200, predicted `Dropout`; probabilities rendered |
| Empty ID | HTTP 400 |
| Invalid ID format | HTTP 400 |
| Unknown ID | HTTP 404 |
| Real record with a required feature temporarily set missing in an isolated test app | HTTP 422 |
| Duplicate ID introduced only in an isolated test app | HTTP 409; no prediction |
| Missing model path | Clear `FileNotFoundError`; no fallback training |
| Pipeline load | 30 features and `Dropout`, `Enrolled`, `Graduate` classes verified |

The app also launched with `python app.py` and reported listening at `127.0.0.1:5000`, but HTTP requests from this restricted environment were blocked with Windows `WinError 10013`. The server was stopped. The Flask test client passed the route checks, but an external browser connection could not be verified here.

PowerShell commands from the project directory:

```powershell
cd D:\ML_project
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

A compatible interpreter already available in this environment can also start it with `python app.py`; the project `.venv` should first be aligned using the pinned requirements. Flask remains a local demonstration app, not a production deployment. A real live prediction service needs current feature records, and estimates must remain distinct from already-recorded outcomes.

## Prediction behavior investigation (2026-10-29)

Investigated the report that IDs 1001, 1002, and 1017 all display `Dropout`.

- The notebook's finalization cells select `Logistic Regression balanced` using the documented training-only five-fold one-standard-error procedure, refit that pipeline on the training partition, and save it to `models/student_outcome_pipeline.joblib`. The code asserts the 30 eligible feature names and original string target classes before saving. This matches the artifact loaded by the Flask application; no training or feature mismatch was found.
- The deployed pipeline uses the same 30 first-semester/background inputs. Its lookup ignores `Target`, `Student_ID` is only used to retrieve a row, and no second-semester inputs are loaded. The output is a model prediction and is not copied from the recorded target.
- Direct saved-pipeline inference and Flask test-client output agree for all three queried IDs (HTTP 200):

| Student ID | Recorded dataset outcome (diagnostic only) | Predicted outcome | Dropout | Enrolled | Graduate |
|---|---|---|---:|---:|---:|
| 1001 | Dropout | Dropout | 66.1% | 12.6% | 21.4% |
| 1002 | Graduate | Dropout | 56.2% | 7.9% | 35.9% |
| 1017 | Enrolled | Dropout | 66.1% | 28.5% | 5.4% |

The percentages are the fitted classifier's estimated class probabilities, not guarantees. The three examples therefore confirm a model error on two examples, not an ID lookup or rendering defect. A batch inference over all 4,424 feature records produced all three classes: Graduate 2,089, Dropout 1,255, and Enrolled 1,080. This is a prediction-distribution diagnostic, not a new evaluation; the held-out test metrics above remain the evaluation results and were not used for selection or modification.

No model or application code was changed, and no new model was trained. The current evidence supports that these `Dropout` outputs are the selected model's actual predictions. The open browser page could not be independently inspected because the computer-use layer could not verify its active URL; the Flask test client confirmed the current route behavior. If the browser displays different output after a fresh app restart, verify it is running `D:\ML_project\app.py` with the compatible environment and the artifact path above.
