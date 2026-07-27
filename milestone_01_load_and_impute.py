"""Milestone 1: load the UCI Arrhythmia dataset and impute missing values.

Why median imputation?
- The dataset is small, so we want a simple, low-variance baseline.
- The features are numeric-coded and "?" marks missing entries.
- Median is robust to skew and outliers, which is useful before any scaling or PCA.

Keep this step separate from later model evaluation so the imputer can be fit only on training data.
"""

from __future__ import annotations

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"


def load_arrhythmia_data(source: str = DATA_URL) -> pd.DataFrame:
    """Load the raw Arrhythmia dataset and convert '?' to missing values."""
    return pd.read_csv(source, header=None, na_values="?")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the last column as the target label."""
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype("int64").copy()
    return X, y


def impute_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with the column median."""
    return X.fillna(X.median(numeric_only=True))


def main() -> None:
    df = load_arrhythmia_data()
    X, y = split_features_target(df)
    X_imputed = impute_missing_values(X)

    print(f"Loaded shape: {df.shape}")
    print(f"Missing values in X before imputation: {int(X.isna().sum().sum())}")
    print(f"Missing values in X after imputation: {int(X_imputed.isna().sum().sum())}")
    print(f"Target classes: {y.nunique()} total")


if __name__ == "__main__":
    main()
