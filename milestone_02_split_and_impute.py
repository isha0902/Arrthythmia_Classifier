"""Milestone 2: split the data before preprocessing, then impute safely.

Why split before imputation?
- This avoids data leakage. The imputer should learn from training data only.
- With a small, imbalanced dataset, stratified splitting is important so minority
  classes remain represented in both train and test folds.
- Median imputation is still a good baseline here because it is robust to outliers
  and avoids introducing extra complexity before PCA or model comparison.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"


def load_arrhythmia_data(source: str = DATA_URL) -> pd.DataFrame:
    """Load the raw Arrhythmia dataset and convert '?' to missing values."""
    return pd.read_csv(source, header=None, na_values="?")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the last column as the target label."""
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype("int64").copy()
    return X, y


def merge_rare_classes(
    y: pd.Series,
    rare_labels: list[int],
    new_label: int,
) -> pd.Series:
    """Replace rare classes with a single combined label and return a new series."""
    y_merged = y.copy()
    y_merged = y_merged.where(~y_merged.isin(rare_labels), new_label)
    return y_merged.astype("int64")


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test sets with stratification."""
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def fit_and_apply_imputer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the imputer on the training set and apply it to both train and test."""
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_imputed, X_test_imputed


def main() -> None:
    df = load_arrhythmia_data()
    X, y = split_features_target(df)

    print("Original target value counts:")
    print(y.value_counts().sort_index())

    rare_labels = [7, 8, 9, 14, 15]
    y = merge_rare_classes(y, rare_labels, 99)

    print("Merged target value counts:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_imputed, X_test_imputed = fit_and_apply_imputer(X_train, X_test)

    print(f"Loaded shape: {df.shape}")
    print(f"X_train shape: {X_train_imputed.shape}")
    print(f"X_test shape: {X_test_imputed.shape}")
    print(f"Missing values in X_train after imputation: {int(X_train_imputed.isna().sum().sum())}")
    print(f"Missing values in X_test after imputation: {int(X_test_imputed.isna().sum().sum())}")
    print(f"Training target classes: {y_train.nunique()} total")
    print(f"Test target classes: {y_test.nunique()} total")


if __name__ == "__main__":
    main()
