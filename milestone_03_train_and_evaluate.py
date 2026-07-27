"""Milestone 3: compare multiple baseline classifiers using stratified k-fold CV.

Goal: train and compare Logistic Regression, KNN, Decision Tree, Random Forest,
Linear SVM, and Kernel SVM (RBF) on the arrhythmia dataset, using stratified
k-fold cross-validation rather than a single train/test split.
"""

from __future__ import annotations

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from milestone_02_split_and_impute import (
    load_arrhythmia_data,
    merge_rare_classes,
    split_features_target,
    split_train_test,
)
from milestone_02c_full_pipeline import main as build_preprocessed_data
from milestone_02c_full_pipeline import main_pca_only as build_pca_preprocessed_data


def build_raw_feature_pipeline(estimator: object) -> Pipeline:
    """Create an imputation-enabled pipeline for the raw-feature comparison."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ]
    )


def run_cv_comparison(X: pd.DataFrame | pd.Series, y: pd.Series) -> pd.DataFrame:
    """Run stratified 5-fold CV for each model and return a summary table."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_specs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("KNN", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")),
        ("Linear SVM", SVC(kernel="linear", random_state=42, class_weight="balanced")),
        ("Kernel SVM (RBF)", SVC(kernel="rbf", random_state=42, class_weight="balanced")),
    ]

    results: list[dict[str, object]] = []
    for display_name, estimator in model_specs:
        pipeline = build_raw_feature_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=None,
        )
        results.append(
            {
                "model": display_name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
                "cv_f1_macro_std": scores["test_f1_macro"].std(),
            }
        )

    return pd.DataFrame(results)


def print_single_split_confusion_matrices(X: pd.DataFrame, y: pd.Series) -> None:
    """Fit each model once on a single stratified split and print a confusion matrix."""
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model_specs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("KNN", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")),
        ("Linear SVM", SVC(kernel="linear", random_state=42, class_weight="balanced")),
        ("Kernel SVM (RBF)", SVC(kernel="rbf", random_state=42, class_weight="balanced")),
    ]

    for display_name, estimator in model_specs:
        pipeline = build_raw_feature_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n=== Single-split confusion matrix: {display_name} ===")
        print(confusion_matrix(y_test, y_pred))


def build_smote_in_cv_pipeline(estimator: object, y_train: pd.Series) -> ImbPipeline:
    """Build an imbalanced-learn pipeline that applies SMOTE inside each CV fold."""
    class_counts = y_train.value_counts().sort_index()
    smallest_class_count = int(class_counts.min())
    k_neighbors = min(5, max(1, smallest_class_count - 1))
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ("classifier", estimator),
        ]
    )


def print_pca_smote_results(X_train: pd.DataFrame | pd.Series, y_train: pd.Series, X_test: pd.DataFrame | pd.Series, y_test: pd.Series) -> None:
    """Compare the same six models on the PCA-transformed data with SMOTE inside each CV fold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_specs = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("KNN", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")),
        ("Linear SVM", SVC(kernel="linear", random_state=42, class_weight="balanced")),
        ("Kernel SVM (RBF)", SVC(kernel="rbf", random_state=42, class_weight="balanced")),
    ]

    results: list[dict[str, object]] = []
    for display_name, estimator in model_specs:
        pipeline = build_smote_in_cv_pipeline(estimator, y_train)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=None,
        )
        results.append(
            {
                "model": display_name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
                "cv_f1_macro_std": scores["test_f1_macro"].std(),
            }
        )

    print("\n=== PCA + SMOTE-in-CV (leakage-fixed) results ===")
    print(pd.DataFrame(results).to_string(index=False))

    for display_name, estimator in model_specs:
        pipeline = build_smote_in_cv_pipeline(estimator, y_train)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f"\n=== PCA + SMOTE single-split evaluation: {display_name} ===")
        print("Accuracy:", (y_pred == y_test).mean())
        print(classification_report(y_test, y_pred, zero_division=0))
        print(confusion_matrix(y_test, y_pred))


def main() -> None:
    df = load_arrhythmia_data()
    X, y = split_features_target(df)

    rare_labels = [7, 8, 9, 14, 15]
    y = merge_rare_classes(y, rare_labels, 99)

    summary = run_cv_comparison(X, y)
    print("=== Stratified 5-fold CV comparison (raw imputed features) ===")
    print(summary.to_string(index=False))

    print("\n=== Single-split confusion matrices (qualitative view only) ===")
    print_single_split_confusion_matrices(X, y)

    X_train_pca, y_train, X_test_pca, y_test = build_pca_preprocessed_data()
    print_pca_smote_results(X_train_pca, y_train, X_test_pca, y_test)


if __name__ == "__main__":
    main()
