"""Reproduce the Student Outcome model-selection and final evaluation workflow.

This script mirrors the split, preprocessing, tuning, training-only candidate CV,
one-standard-error selection, and one-time held-out test evaluation in
notbooks/ML_project.ipynb. It never writes or replaces a model artifact.
"""

from __future__ import annotations

from pathlib import Path
import importlib.metadata
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "students_with_id.csv"
MODEL_PATH = ROOT / "models" / "student_outcome_pipeline.joblib"
RESULTS_PATH = ROOT / "MODEL_EVALUATION_RESULTS.md"

ID_COLUMN = "Student_ID"
TARGET_COLUMN = "Target"
EXPECTED_CLASSES = {"Dropout", "Enrolled", "Graduate"}
LABELS = ["Dropout", "Enrolled", "Graduate"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], *, scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler() if scale_numeric else "passthrough", numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def run() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Current Flask pipeline not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df or ID_COLUMN not in df:
        raise ValueError("Dataset must contain Student_ID and Target columns.")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Student_ID is not unique; resolve lookup ambiguity before evaluation.")
    if df[TARGET_COLUMN].isna().any() or set(df[TARGET_COLUMN].unique()) != EXPECTED_CLASSES:
        raise ValueError("Target labels are missing or differ from the expected three classes.")

    second_semester = [c for c in df.columns if c.startswith("Curricular units 2nd sem (")]
    excluded = {ID_COLUMN, TARGET_COLUMN, *second_semester}
    features = [c for c in df.columns if c not in excluded]
    if len(df) != 4424 or len(features) != 30 or len(second_semester) != 6:
        raise ValueError(
            f"Unexpected data shape/schema: rows={len(df)}, features={len(features)}, "
            f"second-semester fields={len(second_semester)}. Review before evaluation."
        )
    if df[features].isna().any().any():
        raise ValueError("Required feature values are missing; resolve before model evaluation.")

    X = df.loc[:, features].copy()
    y = df[TARGET_COLUMN].copy()
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    numeric_features = [c for c in X.columns if c not in categorical_features]
    if set(categorical_features + numeric_features) != set(features):
        raise AssertionError("Preprocessing columns do not cover the model feature set.")

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_test = X.loc[train_indices].copy(), X.loc[test_indices].copy()
    y_train, y_test = y.loc[train_indices].copy(), y.loc[test_indices].copy()
    if not set(X_train.index).isdisjoint(set(X_test.index)):
        raise AssertionError("Train/test rows overlap.")

    # STEP 4 tree tuning: only the training partition enters GridSearchCV.
    tune_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_grid = GridSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1)),
            ]
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
        },
        cv=tune_cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    rf_grid.fit(X_train, y_train)

    label_encoder = LabelEncoder().fit(sorted(EXPECTED_CLASSES))
    y_train_encoded = label_encoder.transform(y_train)
    xgb_grid = GridSearchCV(
        Pipeline(
            [
                ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                ("classifier", XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=1)),
            ]
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
        },
        cv=tune_cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    xgb_grid.fit(X_train, y_train_encoded)

    candidates = {
        "Majority dummy": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                    ("classifier", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            "encode": False,
            "sample_weight": False,
        },
        "Logistic Regression unweighted": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
                    ("classifier", LogisticRegression(max_iter=2000, solver="newton-cholesky", random_state=RANDOM_STATE)),
                ]
            ),
            "encode": False,
            "sample_weight": False,
        },
        "Logistic Regression balanced": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
                    ("classifier", LogisticRegression(max_iter=2000, solver="newton-cholesky", random_state=RANDOM_STATE, class_weight="balanced")),
                ]
            ),
            "encode": False,
            "sample_weight": False,
        },
        "Random Forest unweighted": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                    ("classifier", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight=None, n_jobs=1)),
                ]
            ),
            "encode": False,
            "sample_weight": False,
        },
        "Random Forest balanced": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                    ("classifier", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1)),
                ]
            ),
            "encode": False,
            "sample_weight": False,
        },
        "Random Forest tuned reference": {"pipeline": rf_grid.best_estimator_, "encode": False, "sample_weight": False},
        "XGBoost unweighted": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                    ("classifier", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=1)),
                ]
            ),
            "encode": True,
            "sample_weight": False,
        },
        "XGBoost balanced sample weights": {
            "pipeline": Pipeline(
                [
                    ("preprocess", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
                    ("classifier", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=1)),
                ]
            ),
            "encode": True,
            "sample_weight": True,
        },
        "XGBoost tuned reference": {"pipeline": xgb_grid.best_estimator_, "encode": True, "sample_weight": False},
    }

    selection_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_predictions: dict[str, pd.Series] = {
        name: pd.Series(index=y_train.index, dtype=object) for name in candidates
    }
    fold_rows: list[dict[str, object]] = []
    for name, candidate in candidates.items():
        for fold, (fit_positions, valid_positions) in enumerate(selection_cv.split(X_train, y_train), start=1):
            model = clone(candidate["pipeline"])
            X_fold_train, X_fold_valid = X_train.iloc[fit_positions], X_train.iloc[valid_positions]
            y_fold_train, y_fold_valid = y_train.iloc[fit_positions], y_train.iloc[valid_positions]
            if candidate["encode"]:
                y_fold_encoded = label_encoder.transform(y_fold_train)
                fit_params = {}
                if candidate["sample_weight"]:
                    fit_params["classifier__sample_weight"] = compute_sample_weight(
                        class_weight="balanced", y=y_fold_encoded
                    )
                model.fit(X_fold_train, y_fold_encoded, **fit_params)
                prediction = label_encoder.inverse_transform(model.predict(X_fold_valid).astype(int))
            else:
                model.fit(X_fold_train, y_fold_train)
                prediction = model.predict(X_fold_valid)
            oof_predictions[name].iloc[valid_positions] = prediction
            fold_rows.append(
                {
                    "Model": name,
                    "Fold": fold,
                    "Macro F1": f1_score(y_fold_valid, prediction, average="macro", zero_division=0),
                    "Balanced accuracy": balanced_accuracy_score(y_fold_valid, prediction),
                    "Dropout recall": recall_score(y_fold_valid, prediction, labels=["Dropout"], average="macro", zero_division=0),
                    "Enrolled recall": recall_score(y_fold_valid, prediction, labels=["Enrolled"], average="macro", zero_division=0),
                    "Graduate recall": recall_score(y_fold_valid, prediction, labels=["Graduate"], average="macro", zero_division=0),
                }
            )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_summary = fold_metrics.groupby("Model").mean(numeric_only=True)

    candidate_rows: list[dict[str, object]] = []
    class_rows: list[list[object]] = []
    for name, predictions in oof_predictions.items():
        if predictions.isna().any():
            raise AssertionError(f"Missing OOF predictions for {name}.")
        report = classification_report(
            y_train, predictions, labels=LABELS, output_dict=True, zero_division=0
        )
        candidate_rows.append(
            {
                "Model": name,
                "Mean CV macro F1": fold_summary.loc[name, "Macro F1"],
                "OOF accuracy": accuracy_score(y_train, predictions),
                "OOF macro precision": report["macro avg"]["precision"],
                "OOF macro recall": report["macro avg"]["recall"],
                "OOF macro F1": report["macro avg"]["f1-score"],
                "OOF weighted F1": report["weighted avg"]["f1-score"],
                "Mean CV balanced accuracy": fold_summary.loc[name, "Balanced accuracy"],
            }
        )
        for label in LABELS:
            class_rows.append(
                [
                    name,
                    label,
                    f"{report[label]['precision']:.4f}",
                    f"{report[label]['recall']:.4f}",
                    f"{report[label]['f1-score']:.4f}",
                    int(report[label]["support"]),
                ]
            )
    candidate_summary = pd.DataFrame(candidate_rows).sort_values(
        "Mean CV macro F1", ascending=False
    ).reset_index(drop=True)

    # Training-only one-standard-error selection, matching notebook cell 37.
    top_name = str(candidate_summary.iloc[0]["Model"])
    top_scores = fold_metrics.loc[fold_metrics["Model"] == top_name, "Macro F1"]
    top_se = top_scores.std(ddof=1) / np.sqrt(len(top_scores))
    selection_floor = float(candidate_summary.iloc[0]["Mean CV macro F1"] - top_se)
    eligible = candidate_summary[
        candidate_summary["Mean CV macro F1"] >= selection_floor
    ].copy()
    enrolled_recall = fold_summary["Enrolled recall"]
    eligible["CV Enrolled recall mean"] = eligible["Model"].map(enrolled_recall)
    eligible["CV balanced accuracy mean"] = eligible["Model"].map(fold_summary["Balanced accuracy"])
    selected_name = str(
        eligible.sort_values(
            ["CV Enrolled recall mean", "CV balanced accuracy mean", "Mean CV macro F1"],
            ascending=False,
        ).iloc[0]["Model"]
    )

    # This is the only held-out test scoring call; selection has already been locked.
    selected_candidate = candidates[selected_name]
    final_pipeline = clone(selected_candidate["pipeline"])
    final_pipeline.fit(X_train, y_train)
    if list(final_pipeline.feature_names_in_) != features:
        raise AssertionError("Final pipeline input schema differs from the 30 selected features.")
    test_predictions = final_pipeline.predict(X_test)
    test_report = classification_report(
        y_test, test_predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    test_confusion = confusion_matrix(y_test, test_predictions, labels=LABELS)

    # Check the existing saved artifact against the selected fresh fit without saving.
    saved_pipeline = joblib.load(MODEL_PATH)
    artifact_features_match = list(saved_pipeline.feature_names_in_) == features
    artifact_test_predictions_match = np.array_equal(saved_pipeline.predict(X_test), test_predictions)
    artifact_coef_match = (
        selected_name == "Logistic Regression balanced"
        and np.allclose(
            saved_pipeline.named_steps["classifier"].coef_,
            final_pipeline.named_steps["classifier"].coef_,
            rtol=0,
            atol=1e-12,
        )
    )

    selected_test_metrics = {
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Macro precision": precision_score(y_test, test_predictions, average="macro", zero_division=0),
        "Macro recall": recall_score(y_test, test_predictions, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, test_predictions, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, test_predictions, average="weighted", zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(y_test, test_predictions),
    }

    target_counts = y.value_counts().reindex(LABELS)
    train_counts = y_train.value_counts().reindex(LABELS)
    test_counts = y_test.value_counts().reindex(LABELS)
    confusion_rows = [
        [LABELS[i], *[int(value) for value in test_confusion[i]]]
        for i in range(len(LABELS))
    ]
    report_lines = [
        "# Model Evaluation Results",
        "",
        f"**Run date:** {pd.Timestamp.now().date()}  ",
        f"**Evaluation script:** `{Path(__file__).name}`  ",
        f"**Dataset:** `{DATA_PATH.relative_to(ROOT)}` (read-only)  ",
        f"**Selected model:** {selected_name}  ",
        f"**Saved Flask pipeline:** `{MODEL_PATH.relative_to(ROOT)}` (read-only; not replaced)",
        "",
        "## Execution environment",
        "",
        f"Reported results were computed with Python {sys.version.split()[0]} and "
        + ", ".join(
            f"{package} {importlib.metadata.version(package)}"
            for package in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost"]
        )
        + ".",
        "",
        "## Data and features",
        "",
        f"The dataset has {len(df):,} rows and {df.shape[1]} columns. The model uses {len(features)} admission/background and first-semester features. `Student_ID`, `Target`, and all six second-semester fields are excluded from inputs. Student IDs are unique. Full target counts: "
        + "; ".join(f"{label}: {int(target_counts[label])}" for label in LABELS) + ".",
        "",
        "## Split and validation method",
        "",
        f"A seeded stratified 80/20 split (`random_state={RANDOM_STATE}`) produced {len(X_train):,} training rows and {len(X_test):,} held-out test rows. Training counts: "
        + "; ".join(f"{label}: {int(train_counts[label])}" for label in LABELS) + ". Test counts: "
        + "; ".join(f"{label}: {int(test_counts[label])}" for label in LABELS) + ".",
        "",
        "Random Forest (36 settings) and XGBoost (27 settings) were tuned using five-fold stratified CV on training data only, with macro F1 scoring (`random_state=42`). Nine candidate configurations, including a majority dummy, weighted/unweighted Logistic Regression, Random Forest, XGBoost, and the tuned tree references, were compared using shared five-fold stratified training-only out-of-fold folds (`random_state=2026`). Preprocessing is fitted inside each pipeline/fold. The test set was not used for tuning or selection.",
        "",
        "## Candidate model comparison (training-only validation)",
        "",
        "The candidate table reports the mean fold macro F1 used by the one-standard-error rule. The other metrics and class-wise results are pooled out-of-fold predictions over the training partition.",
        "",
    ]
    candidate_table_rows = []
    for _, row in candidate_summary.iterrows():
        candidate_table_rows.append(
            [
                row["Model"],
                f"{row['Mean CV macro F1']:.4f}",
                f"{row['OOF accuracy']:.4f}",
                f"{row['OOF macro precision']:.4f}",
                f"{row['OOF macro recall']:.4f}",
                f"{row['OOF weighted F1']:.4f}",
            ]
        )
    report_lines.append(
        md_table(
            ["Candidate", "Mean CV macro F1", "OOF accuracy", "OOF macro precision", "OOF macro recall", "OOF weighted F1"],
            candidate_table_rows,
        )
    )
    report_lines.extend(["", "### Training out-of-fold per-class results", ""])
    report_lines.append(md_table(["Candidate", "Class", "Precision", "Recall", "F1", "Support"], class_rows))
    report_lines.extend(
        [
            "",
            "## Model selection rationale",
            "",
            f"The highest training-CV mean macro F1 was {top_name} ({float(candidate_summary.iloc[0]['Mean CV macro F1']):.4f}). Its one-standard-error allowance was {top_se:.4f}, producing a macro-F1 eligibility floor of {selection_floor:.4f}. Among candidates within that range, the existing notebook rule selected the candidate with the highest mean Enrolled recall; mean balanced accuracy and then mean macro F1 break ties. The selected candidate was **{selected_name}**. Accuracy was not the primary selection metric. The rule uses training folds only.",
            "",
            "## Final selected-model test evaluation",
            "",
            "This evaluation was run once after selection was locked.",
            "",
            md_table(
                ["Metric", "Held-out test result"],
                [[name, f"{value:.4f}"] for name, value in selected_test_metrics.items()],
            ),
            "",
            "### Classification report",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        metrics = test_report[label]
        report_lines.append(
            f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |"
        )
    report_lines.extend(
        [
            "",
            "### Confusion matrix",
            "",
            "Rows are actual classes; columns are predicted classes in `[Dropout, Enrolled, Graduate]` order.",
            "",
            md_table(["Actual \\ Predicted", *LABELS], confusion_rows),
            "",
            "## Saved pipeline consistency",
            "",
            f"The saved pipeline input names match the freshly selected 30-feature schema: **{artifact_features_match}**. Its predictions on this held-out split match the fresh fit: **{artifact_test_predictions_match}**. Logistic Regression coefficient match at absolute tolerance 1e-12: **{artifact_coef_match}**. The active artifact was loaded for comparison and was not modified.",
            "",
            "## Limitations",
            "",
            f"Enrolled remains the weakest class: held-out precision {test_report['Enrolled']['precision']:.3f}, recall {test_report['Enrolled']['recall']:.3f}, F1 {test_report['Enrolled']['f1-score']:.3f} (support {int(test_report['Enrolled']['support'])}). Its errors include both false negatives and false positives; the confusion matrix shows the counts. The dataset contains historical final outcomes, so this is a retrospective evaluation and demonstration, not evidence of prospective performance on future students. The tuned-reference candidate CV is performed after hyperparameters were chosen using the same training partition in a separate CV seed; the untouched test evaluation is the final generalization estimate.",
            "",
            "## Reproduce",
            "",
            "From PowerShell in the project root, activate the environment with compatible notebook dependencies (including XGBoost) and run:",
            "",
            "```powershell",
            "cd D:\\ML_project",
            ".\\.venv\\Scripts\\Activate.ps1",
            "python -m pip install -r requirements.txt xgboost==3.4.1",
            "python evaluate_model.py",
            "```",
            "",
            "The script regenerates this report, reads but does not change the dataset or saved pipeline, and does not modify the Flask app or notebook. For a Jupyter execution attempt, the existing notebook kernel launcher did not start reliably in this environment; this script is the deterministic non-Jupyter fallback.",
            "",
        ]
    )
    RESULTS_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print("Selected candidate:", selected_name)
    print("Training CV candidates:\n", candidate_summary.round(4).to_string(index=False))
    print("Final test metrics:", {k: round(v, 4) for k, v in selected_test_metrics.items()})
    print("Final test report:\n", classification_report(y_test, test_predictions, labels=LABELS, target_names=LABELS, zero_division=0))
    print("Confusion matrix (Dropout, Enrolled, Graduate):\n", test_confusion)
    print("Saved artifact match:", artifact_features_match, artifact_test_predictions_match, artifact_coef_match)
    print("Wrote:", RESULTS_PATH)


if __name__ == "__main__":
    run()
