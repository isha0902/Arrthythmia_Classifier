"""Milestone 2b: scale -> PCA -> SMOTE preprocessing chain.

This module wraps the later preprocessing steps so Milestone 3 can import the
final train/test artifacts directly without recomputing the pipeline.

RESULT: Rejected. Once SMOTE-in-CV leakage was fixed, this pipeline
underperformed the plain imputed + class_weight='balanced' baseline
across all 6 models tested (e.g. Random Forest macro-F1: 0.611
baseline vs. 0.404 with PCA+SMOTE). Kept here as a documented
negative result - the final pipeline used going forward is the
raw imputed one in milestone_02_split_and_impute.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from milestone_02_split_and_impute import (
    fit_and_apply_imputer,
    load_arrhythmia_data,
    merge_rare_classes,
    split_features_target,
    split_train_test,
)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on the training set and apply it to both train and test."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled


def explore_pca_variance(X_train_scaled: pd.DataFrame) -> None:
    """Fit PCA on the scaled training data and print cumulative variance information."""
    pca = PCA(n_components=None)
    pca.fit(X_train_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print("PCA cumulative explained variance (first 50 components):")
    for idx, variance in enumerate(cumulative_variance[:50], start=1):
        print(f"Component {idx}: {variance:.4f}")

    n_components_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
    n_components_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)
    print(f"Components needed for 90% variance: {n_components_90}")
    print(f"Components needed for 95% variance: {n_components_95}")


def apply_pca(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    n_components: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training data only and transform both train and test sets."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca


def apply_smote(
    X_train_pca: np.ndarray,
    y_train: pd.Series,
) -> tuple[np.ndarray, pd.Series]:
    """Apply SMOTE to the PCA-transformed training data using a dynamic k_neighbors value."""
    class_counts = y_train.value_counts().sort_index()
    smallest_class_count = int(class_counts.min())

    if smallest_class_count <= 1:
        print("Warning: at least one class is too small for SMOTE even with k_neighbors=1; returning the original training data.")
        return X_train_pca, y_train.copy()

    k_neighbors = min(5, smallest_class_count - 1)
    print(f"SMOTE k_neighbors: {k_neighbors}")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_smote, y_smote = smote.fit_resample(X_train_pca, y_train)
    y_smote_series = pd.Series(y_smote, name=y_train.name)
    return X_smote, y_smote_series


def build_pca_preprocessed_data() -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    """Run scaling and PCA only, returning the PCA-transformed train/test artifacts."""
    df = load_arrhythmia_data()
    X, y = split_features_target(df)

    rare_labels = [7, 8, 9, 14, 15]
    y = merge_rare_classes(y, rare_labels, 99)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_imputed, X_test_imputed = fit_and_apply_imputer(X_train, X_test)
    X_train_scaled, X_test_scaled = scale_features(X_train_imputed, X_test_imputed)

    explore_pca_variance(X_train_scaled)
    X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_components=70)

    print("Training target value counts before SMOTE:")
    print(y_train.value_counts().sort_index())
    print(f"X_train after PCA shape: {X_train_pca.shape}")
    print(f"X_test after PCA shape: {X_test_pca.shape}")

    return X_train_pca, y_train, X_test_pca, y_test


def build_preprocessed_data() -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    """Run the full scaling -> PCA -> SMOTE pipeline and return final train/test artifacts."""
    X_train_pca, y_train, X_test_pca, y_test = build_pca_preprocessed_data()
    X_train_final, y_train_final = apply_smote(X_train_pca, y_train)

    print("Training target value counts after SMOTE:")
    print(y_train_final.value_counts().sort_index())
    print(f"X_train after SMOTE shape: {X_train_final.shape}")

    return X_train_final, y_train_final, X_test_pca, y_test


if __name__ == "__main__":
    build_preprocessed_data()
