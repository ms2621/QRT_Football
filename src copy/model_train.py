from xgboost import XGBRegressor, XGBClassifier
import optuna

def train_regressor(X_train, y_train, params):
    """
    Train an XGBoost regressor with the given parameters.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        params (dict): Hyperparameters for the regressor.

    Returns:
        Trained regressor model.
    """
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_classifier(X_train, y_train, params):
    """
    Train an XGBoost classifier with the given parameters.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        params (dict): Hyperparameters for the classifier.

    Returns:
        Trained classifier model.
    """
    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

import optuna
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def optuna_regressor_tuning(X_train, y_train, n_trials=50):
    """
    Perform Optuna hyperparameter tuning for XGBoost regression.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        n_trials (int): Number of Optuna trials.

    Returns:
        dict: Best parameters found by Optuna.
    """
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
        }

        model = XGBRegressor(**params, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
            y_pred = model.predict(X_cv_val)
            mse_scores.append(mean_squared_error(y_cv_val, y_pred))

        return sum(mse_scores) / len(mse_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optuna_classifier_tuning(X_train, y_train, n_trials=50):
    """
    Perform Optuna hyperparameter tuning for XGBoost classification.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        n_trials (int): Number of Optuna trials.

    Returns:
        dict: Best parameters found by Optuna.
    """
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        }

        model = XGBClassifier(**params, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_scores = []

        for train_idx, val_idx in kf.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
            y_pred = model.predict(X_cv_val)
            accuracy_scores.append(accuracy_score(y_cv_val, y_pred))

        return 1 - (sum(accuracy_scores) / len(accuracy_scores))  # Minimize error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
