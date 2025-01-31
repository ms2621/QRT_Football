# QRT_Football
Combination of XGB regression and classification to predict Win Draw or Loss of home team in football games. Achieves classification accuracy of 0.4899 on unseen-test data.
Data can be found at: https://challengedata.ens.fr/participants/challenges/143/

# Football Match Outcome Prediction

## Overview
This project predicts football match outcomes using player statistics, team-level data, and advanced machine learning techniques. The solution uses XGBoost models for regression and classification tasks. Additionally, Bayesian hyperparameter tuning with Optuna ensures optimal model performance.


## Key Features
- **Feature Engineering**:
  - Aggregated player statistics by position (e.g., defender, midfielder, attacker).
  - Imputed missing values and scaled numerical features.
  - Deleted columns with 100% missing values and rows with 40% or more missing data.
  - Generated position-based summary statistics (mean, sum, max, std), simulating starting lineup (only players with 45 mins played per game on average).

- **Dimensionality Reduction**:
  - Utilized Elastic Net for feature selection to reduce dimensionality.

- **Modeling**:
  - Regression: Predicts goal differences using XGBRegressor.
  - Classification: Predicts match outcomes (home win, draw, away win) using XGBClassifier.
  - Selected features by regressing against residuals from a baseline XGBoost model, which is computationally efficient as residuals are calculated only once.

- **Hyperparameter Tuning**:
  - Used Optuna to optimize hyperparameters for regression and classification models.

- **Evaluation**:
  - Cross-validated mean squared error (MSE) and R-squared for regression.
  - Cross-validated accuracy and log loss for classification.


## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Place your data files in the `data/` directory.

3. Run the scripts from the `scripts/` folder to execute various stages of the pipeline.

## Usage
- **Run QRT_raw.py**:


## Results
The pipeline generates predictions for match outcomes and goal differences. The final outputs are saved in the `results/` directory. Key metrics include:
- Regression:
  - Test MSE: <value>
  - Test R²: <value>
- Classification:
  - Test Accuracy: <value>
  - Test Log Loss: <value>

##Other Techniques
Techniques such as RFE for feature selection could not be employed due to lack of local RAM.
PCA, autoencoders, ridge and lasso regression were also tried and tested but experienced a significant drop in performance particularly with PCA. Condition number was always <30 when elastic net (Lasso + Ridge) was used, so I determined the drop in explanation of variance was not worth the added stability.

## Acknowledgments
This project leverages:
- **XGBoost** for gradient-boosted decision trees.
- **Optuna** for efficient hyperparameter optimization.
- Acknowledgment to **Elsa Doukhan** for her guidance and support in this project.

## License
This project is licensed under the [MIT License](LICENSE).

