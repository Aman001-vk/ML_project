# Model Evaluation Results

**Run date:** 2026-10-30  
**Evaluation script:** `evaluate_model.py`  
**Dataset:** `data\students_with_id.csv` (read-only)  
**Selected model:** Logistic Regression balanced  
**Saved Flask pipeline:** `models\student_outcome_pipeline.joblib` (read-only; not replaced)

## Execution environment

Reported results were computed with Python 3.13.16, NumPy 2.2.1, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.18.0, joblib 1.4.2, and XGBoost 3.4.1.

## Data and features

The dataset has 4,424 rows and 38 columns. The model uses 30 admission/background and first-semester features. `Student_ID`, `Target`, and all six second-semester fields are excluded from inputs. Student IDs are unique. Full target counts: Dropout: 1421; Enrolled: 794; Graduate: 2209.

## Split and validation method

A seeded stratified 80/20 split (`random_state=42`) produced 3,539 training rows and 885 held-out test rows. Training counts: Dropout: 1137; Enrolled: 635; Graduate: 1767. Test counts: Dropout: 284; Enrolled: 159; Graduate: 442.

Random Forest (36 settings) and XGBoost (27 settings) were tuned using five-fold stratified CV on training data only, with macro F1 scoring (`random_state=42`). Nine candidate configurations, including a majority dummy, weighted/unweighted Logistic Regression, Random Forest, XGBoost, and the tuned tree references, were compared using shared five-fold stratified training-only out-of-fold folds (`random_state=2026`). Preprocessing is fitted inside each pipeline/fold. The test set was not used for tuning or selection.

## Candidate model comparison (training-only validation)

The candidate table reports the mean fold macro F1 used by the one-standard-error rule. The other metrics and class-wise results are pooled out-of-fold predictions over the training partition.

| Candidate | Mean CV macro F1 | OOF accuracy | OOF macro precision | OOF macro recall | OOF weighted F1 |
|---|---|---|---|---|---|
| XGBoost balanced sample weights | 0.6849 | 0.7321 | 0.6887 | 0.6878 | 0.7379 |
| Logistic Regression balanced | 0.6803 | 0.7222 | 0.6830 | 0.6875 | 0.7311 |
| Random Forest tuned reference | 0.6779 | 0.7423 | 0.6879 | 0.6714 | 0.7368 |
| XGBoost tuned reference | 0.6754 | 0.7511 | 0.6934 | 0.6672 | 0.7401 |
| Logistic Regression unweighted | 0.6693 | 0.7522 | 0.6982 | 0.6613 | 0.7365 |
| XGBoost unweighted | 0.6652 | 0.7482 | 0.6870 | 0.6573 | 0.7345 |
| Random Forest balanced | 0.6344 | 0.7437 | 0.6948 | 0.6307 | 0.7148 |
| Random Forest unweighted | 0.6298 | 0.7443 | 0.7002 | 0.6288 | 0.7125 |
| Majority dummy | 0.2220 | 0.4993 | 0.1664 | 0.3333 | 0.3325 |

### Training out-of-fold per-class results

| Candidate | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Majority dummy | Dropout | 0.0000 | 0.0000 | 0.0000 | 1137 |
| Majority dummy | Enrolled | 0.0000 | 0.0000 | 0.0000 | 635 |
| Majority dummy | Graduate | 0.4993 | 1.0000 | 0.6660 | 1767 |
| Logistic Regression unweighted | Dropout | 0.7790 | 0.7564 | 0.7675 | 1137 |
| Logistic Regression unweighted | Enrolled | 0.5379 | 0.3244 | 0.4047 | 635 |
| Logistic Regression unweighted | Graduate | 0.7778 | 0.9032 | 0.8358 | 1767 |
| Logistic Regression balanced | Dropout | 0.7958 | 0.7062 | 0.7484 | 1137 |
| Logistic Regression balanced | Enrolled | 0.4232 | 0.5685 | 0.4852 | 635 |
| Logistic Regression balanced | Graduate | 0.8301 | 0.7878 | 0.8084 | 1767 |
| Random Forest unweighted | Dropout | 0.7832 | 0.7564 | 0.7696 | 1137 |
| Random Forest unweighted | Enrolled | 0.5760 | 0.1969 | 0.2934 | 635 |
| Random Forest unweighted | Graduate | 0.7415 | 0.9332 | 0.8264 | 1767 |
| Random Forest balanced | Dropout | 0.7951 | 0.7441 | 0.7687 | 1137 |
| Random Forest balanced | Enrolled | 0.5484 | 0.2142 | 0.3080 | 635 |
| Random Forest balanced | Graduate | 0.7409 | 0.9338 | 0.8262 | 1767 |
| Random Forest tuned reference | Dropout | 0.7918 | 0.7326 | 0.7611 | 1137 |
| Random Forest tuned reference | Enrolled | 0.4835 | 0.4157 | 0.4471 | 635 |
| Random Forest tuned reference | Graduate | 0.7883 | 0.8659 | 0.8252 | 1767 |
| XGBoost unweighted | Dropout | 0.7907 | 0.7309 | 0.7596 | 1137 |
| XGBoost unweighted | Enrolled | 0.4896 | 0.3323 | 0.3959 | 635 |
| XGBoost unweighted | Graduate | 0.7807 | 0.9089 | 0.8400 | 1767 |
| XGBoost balanced sample weights | Dropout | 0.8055 | 0.6957 | 0.7466 | 1137 |
| XGBoost balanced sample weights | Enrolled | 0.4374 | 0.5449 | 0.4853 | 635 |
| XGBoost balanced sample weights | Graduate | 0.8233 | 0.8229 | 0.8231 | 1767 |
| XGBoost tuned reference | Dropout | 0.7872 | 0.7353 | 0.7603 | 1137 |
| XGBoost tuned reference | Enrolled | 0.5043 | 0.3669 | 0.4248 | 635 |
| XGBoost tuned reference | Graduate | 0.7886 | 0.8993 | 0.8403 | 1767 |

## Model selection rationale

The highest training-CV mean macro F1 was XGBoost balanced sample weights (0.6849). Its one-standard-error allowance was 0.0095, producing a macro-F1 eligibility floor of 0.6754. Among candidates within that range, the existing notebook rule selected the candidate with the highest mean Enrolled recall; mean balanced accuracy and then mean macro F1 break ties. The selected candidate was **Logistic Regression balanced**. Accuracy was not the primary selection metric. The rule uses training folds only.

## Final selected-model test evaluation

This evaluation was run once after selection was locked.

| Metric | Held-out test result |
|---|---|
| Accuracy | 0.6994 |
| Macro precision | 0.6652 |
| Macro recall | 0.6677 |
| Macro F1 | 0.6593 |
| Weighted F1 | 0.7121 |
| Balanced accuracy | 0.6677 |

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Dropout | 0.7823 | 0.6831 | 0.7293 | 284 |
| Enrolled | 0.3836 | 0.5597 | 0.4552 | 159 |
| Graduate | 0.8296 | 0.7602 | 0.7934 | 442 |

### Confusion matrix

Rows are actual classes; columns are predicted classes in `[Dropout, Enrolled, Graduate]` order.

| Actual \ Predicted | Dropout | Enrolled | Graduate |
|---|---|---|---|
| Dropout | 194 | 63 | 27 |
| Enrolled | 28 | 89 | 42 |
| Graduate | 26 | 80 | 336 |

## Saved pipeline consistency

The saved pipeline input names match the freshly selected 30-feature schema: **True**. Its predictions on this held-out split match the fresh fit: **True**. Logistic Regression coefficient match at absolute tolerance 1e-12: **True**. The active artifact was loaded for comparison and was not modified.

## Limitations

Enrolled remains the weakest class: held-out precision 0.384, recall 0.560, F1 0.455 (support 159). Its errors include both false negatives and false positives; the confusion matrix shows the counts. The dataset contains historical final outcomes, so this is a retrospective evaluation and demonstration, not evidence of prospective performance on future students. The tuned-reference candidate CV is performed after hyperparameters were chosen using the same training partition in a separate CV seed; the untouched test evaluation is the final generalization estimate.

## Reproduce

From PowerShell in the project root, activate the environment with compatible notebook dependencies (including XGBoost) and run:

```powershell
cd D:\ML_project
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt xgboost==3.4.1
python evaluate_model.py
```

The script regenerates this report, reads but does not change the dataset or saved pipeline, and does not modify the Flask app or notebook. For a Jupyter execution attempt, the existing notebook kernel launcher did not start reliably in this environment; this script is the deterministic non-Jupyter fallback.
