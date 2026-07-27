"""Milestone 4: tune the strongest Milestone 3 models with leakage-safe CV.

This script tunes Random Forest, linear SVM, and RBF SVM using a raw-feature
pipeline that imputes missing values inside the model pipeline. The goal is to
optimize macro-F1 with leakage-safe cross-validation, then evaluate the best
configurations on a held-out split for a qualitative check.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from milestone_02_split_and_impute import (
    load_arrhythmia_data,
    merge_rare_classes,
    split_features_target,
    split_train_test,
)


def build_raw_feature_pipeline(estimator: object) -> Pipeline:
    """Create a leakage-safe imputation pipeline for raw features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ]
    )


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the arrhythmia dataset and merge rare classes."""
    df = load_arrhythmia_data()
    X, y = split_features_target(df)
    rare_labels = [7, 8, 9, 14, 15]
    y = merge_rare_classes(y, rare_labels, 99)
    return X, y


def compute_baseline_macro_f1(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Reproduce the Milestone 3 baseline macro-F1 for the tuned candidates."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_specs = [
        (
            "Random Forest",
            RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced"),
        ),
        (
            "Linear SVM",
            SVC(kernel="linear", random_state=42, class_weight="balanced"),
        ),
        (
            "Kernel SVM (RBF)",
            SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for display_name, estimator in model_specs:
        pipeline = build_raw_feature_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=None,
        )
        rows.append(
            {
                "model": display_name,
                "baseline_cv_f1_macro_mean": float(scores["test_score"].mean()),
                "baseline_cv_f1_macro_std": float(scores["test_score"].std()),
            }
        )

    return pd.DataFrame(rows)


def tune_model(
    display_name: str,
    estimator: object,
    param_grid: dict[str, list[object]],
    X: pd.DataFrame,
    y: pd.Series,
) -> GridSearchCV:
    """Run GridSearchCV for one candidate model and return the fitted search object."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = build_raw_feature_pipeline(estimator)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=None,
    )
    search.fit(X, y)

    print(f"\n=== GridSearchCV results: {display_name} ===")
    print("Best hyperparameters:")
    print(search.best_params_)
    print(f"Best cross-validated macro-F1: {search.best_score_:.6f}")
    return search


def evaluate_best_model_on_holdout(
    display_name: str,
    best_params: dict[str, object],
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Fit the tuned configuration once on a held-out split and print test metrics."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    pipeline = build_raw_feature_pipeline(estimator)
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n=== Held-out test evaluation: {display_name} ===")
    print("Best hyperparameters used:")
    print(best_params)
    print("Accuracy:", (y_pred == y_test).mean())
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))


def main() -> None:
    X, y = build_dataset()

    baseline_summary = compute_baseline_macro_f1(X, y)
    print("=== Milestone 3 baseline macro-F1 for tuned candidates ===")
    print(baseline_summary.to_string(index=False))

    tuned_results: list[dict[str, object]] = []

    rf_search = tune_model(
        "Random Forest",
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        X,
        y,
    )
    tuned_results.append(
        {
            "model": "Random Forest",
            "tuned_cv_f1_macro_mean": rf_search.best_score_,
        }
    )
    evaluate_best_model_on_holdout(
        "Random Forest",
        rf_search.best_params_,
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        X,
        y,
    )

    linear_svm_search = tune_model(
        "Linear SVM",
        SVC(kernel="linear", random_state=42, class_weight="balanced"),
        {
            "classifier__C": [0.1, 1, 10, 100],
        },
        X,
        y,
    )
    tuned_results.append(
        {
            "model": "Linear SVM",
            "tuned_cv_f1_macro_mean": linear_svm_search.best_score_,
        }
    )
    evaluate_best_model_on_holdout(
        "Linear SVM",
        linear_svm_search.best_params_,
        SVC(kernel="linear", random_state=42, class_weight="balanced"),
        X,
        y,
    )

    rbf_svm_search = tune_model(
        "Kernel SVM (RBF)",
        SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": ["scale", "auto", 0.01, 0.001],
        },
        X,
        y,
    )
    tuned_results.append(
        {
            "model": "Kernel SVM (RBF)",
            "tuned_cv_f1_macro_mean": rbf_svm_search.best_score_,
        }
    )
    evaluate_best_model_on_holdout(
        "Kernel SVM (RBF)",
        rbf_svm_search.best_params_,
        SVC(kernel="rbf", random_state=42, class_weight="balanced"),
        X,
        y,
    )

    tuned_summary = pd.DataFrame(tuned_results)
    comparison = baseline_summary.merge(tuned_summary, on="model", how="inner")
    comparison["delta_macro_f1"] = (
        comparison["tuned_cv_f1_macro_mean"] - comparison["baseline_cv_f1_macro_mean"]
    )

    print("\n=== Baseline vs tuned macro-F1 comparison ===")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()