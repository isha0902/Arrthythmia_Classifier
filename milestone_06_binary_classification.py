"""Milestone 6: binary arrhythmia classification.

This stretch goal reframes the UCI Arrhythmia dataset as a binary problem:
arrhythmia absent vs present. It provides a sanity-check comparison against the
previous 9-class model.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from milestone_02_split_and_impute import (
    load_arrhythmia_data,
    split_features_target,
    fit_and_apply_imputer,
    split_train_test,
)


def build_raw_feature_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ]
    )


def to_binary_target(y: pd.Series) -> pd.Series:
    print("Value counts before binary relabeling:")
    print(y.value_counts().sort_index())

    y_binary = y.copy()
    y_binary = y_binary.where(y_binary != 1, 0)
    y_binary = y_binary.where(y_binary == 0, 1)
    y_binary = y_binary.astype("int64")

    print("Value counts after binary relabeling:")
    print(y_binary.value_counts().sort_index())
    return y_binary


def main() -> None:
    print("## Data Preparation")
    print(
        "Class 1 is treated as 0 (arrhythmia absent) and every other original class "
        "is treated as 1 (arrhythmia present) before splitting."
    )

    df = load_arrhythmia_data()
    X, y = split_features_target(df)
    y = to_binary_target(y)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_imputed, X_test_imputed = fit_and_apply_imputer(X_train, X_test)

    print("X_train shape:", X_train_imputed.shape)
    print("X_test shape:", X_test_imputed.shape)

    print("\n## Final Model")
    print(
        "The binary comparison reuses the tuned Random Forest configuration from "
        "Milestone 4: max_depth=10, min_samples_leaf=4, n_estimators=100, "
        'class_weight="balanced", random_state=42.'
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = build_raw_feature_pipeline(model)
    cv_scores = cross_validate(
        pipeline,
        X_train_imputed,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=None,
    )

    pipeline.fit(X_train_imputed, y_train)
    y_pred = pipeline.predict(X_test_imputed)

    print("\n## Results")
    print(
        "Binary CV accuracy: {:.3f} +/- {:.3f}".format(
            cv_scores["test_accuracy"].mean(), cv_scores["test_accuracy"].std()
        )
    )
    print(
        "Binary CV macro-F1: {:.3f} +/- {:.3f}".format(
            cv_scores["test_f1_macro"].mean(), cv_scores["test_f1_macro"].std()
        )
    )
    print("Held-out test accuracy:", (y_pred == y_test).mean())
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    print(
        "Comparison: 9-class tuned Random Forest CV macro-F1 = 0.637 vs binary "
        "CV macro-F1 = {:.3f}".format(cv_scores["test_f1_macro"].mean())
    )


if __name__ == "__main__":
    main()
