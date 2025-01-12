from feature_selection import residual_lasso_pipeline
from data_preprocessing import preprocess_data, aggregate_features_by_position
from model_train import optuna_regressor_tuning, train_regressor
from evaluation import evaluate_regression


def main():
    # Load and preprocess data
    home_player, away_player, home_team, away_team, train_output = load_data()
    processed_home = preprocess_data(home_player, "home")
    processed_away = preprocess_data(away_player, "away")
    positions = ["goalkeeper", "defender", "midfielder", "attacker"]

    # Aggregate features by position
    aggregated_home = aggregate_features_by_position(processed_home, positions, "home")
    aggregated_away = aggregate_features_by_position(processed_away, positions, "away")

    # Combine home and away features
    X = aggregated_home.merge(aggregated_away, on="ID", how="inner").drop(columns=["ID"])
    y = train_output["GOAL_DIFF"]

    # Train a baseline model for residual filtering
    from xgboost import XGBRegressor
    baseline_model = XGBRegressor(n_estimators=100, random_state=42)
    baseline_model.fit(X, y)

    # Perform feature selection
    print("Starting feature selection...")
    selected_X, lasso_model = residual_lasso_pipeline(X, y, baseline_model)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning and model training
    print("Starting hyperparameter tuning...")
    best_params = optuna_regressor_tuning(X_train, y_train)
    final_model = train_regressor(X_train, y_train, best_params)

    # Evaluate the model
    print("Evaluating final model...")
    mse = evaluate_regression(final_model, X_test, y_test)
    print(f"Final MSE: {mse}")


if __name__ == "__main__":
    main()
