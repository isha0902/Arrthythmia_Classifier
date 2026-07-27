"""Milestone 2c: full preprocessing pipeline returning the final train/test artifacts.

This module exposes the final PCA+SMOTE outputs so Milestone 3 can import them directly.

RESULT: Rejected after leakage-fixed evaluation. See
milestone_02b_pca_and_resampling.py docstring for the comparison
numbers. Not used in the final pipeline.
"""

from __future__ import annotations

from milestone_02b_pca_and_resampling import build_pca_preprocessed_data, build_preprocessed_data


def main() -> tuple:
    """Return the final PCA+SMOTE train/test artifacts."""
    return build_preprocessed_data()


def main_pca_only() -> tuple:
    """Return the PCA-transformed train/test artifacts before SMOTE is applied."""
    return build_pca_preprocessed_data()


if __name__ == "__main__":
    X_train_final, y_train_final, X_test_pca, y_test = main()
    print("X_train_final shape:", X_train_final.shape)
    print("y_train_final shape:", y_train_final.shape)
    print("X_test_pca shape:", X_test_pca.shape)
    print("y_test shape:", y_test.shape)
