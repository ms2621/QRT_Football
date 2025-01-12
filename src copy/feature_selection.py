from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def baseline_residual_filtering(X, y, baseline_model):
    """
    Select features based on residuals from a baseline model.

    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
        baseline_model (object): A pre-trained model with a `predict` method.

    Returns:
        DataFrame: Filtered feature set based on residual importance.
    """
    residuals = y - baseline_model.predict(X)
    residual_threshold = np.mean(np.abs(residuals))  # Mean absolute residual threshold
    important_features = residuals[residuals > residual_threshold].index
    return X.iloc[important_features]


def lasso_feature_selection(X, y, cv=5):
    """
    Perform Lasso-based feature selection.

    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
        cv (int): Number of cross-validation splits.

    Returns:
        DataFrame: Selected features based on non-zero Lasso coefficients.
    """
    # Standardize the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit LassoCV
    lasso = LassoCV(cv=cv, random_state=42).fit(X_scaled, y)

    # Get selected features
    selected_features = X.columns[(lasso.coef_ != 0)]
    print(f"Selected {len(selected_features)} features via Lasso.")

    return X[selected_features], lasso


def residual_lasso_pipeline(X, y, baseline_model):
    """
    Combined residual filtering and Lasso feature selection.

    Args:
        X (DataFrame): Features for training.
        y (Series): Target variable.
        baseline_model (object): A pre-trained baseline model.

    Returns:
        DataFrame: Final selected features after residual filtering and Lasso selection.
    """
    print("Starting baseline residual filtering...")
    filtered_X = baseline_residual_filtering(X, y, baseline_model)
    print(f"Filtered down to {filtered_X.shape[1]} features based on residuals.")

    print("Starting Lasso feature selection...")
    selected_X, lasso_model = lasso_feature_selection(filtered_X, y)
    print(f"Final feature count after Lasso: {selected_X.shape[1]}")

    return selected_X, lasso_model
