# import pydevd
# pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True, suspend=False)



import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, log_loss
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
import optuna

away_player1 = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/train_away_player_statistics_df.csv')
home_player1 = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/train_home_player_statistics_df.csv')
away_playertest1 = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/test_away_player_statistics_df.csv')
home_playertest1 = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/test_home_player_statistics_df.csv')


away_playertest = away_playertest1.add_suffix("_away")
home_playertest = home_playertest1.add_suffix("_home")
home_player = home_player1.add_suffix("_home")
away_player = away_player1.add_suffix("_away")




home_playertest = home_playertest.rename(columns={"ID_home": "ID"})
away_playertest = away_playertest.rename(columns={"ID_away": "ID"})
position_data_home_test = home_playertest[["ID", "POSITION_home"]].copy()
position_data_away_test = away_playertest[["ID", "POSITION_away"]].copy()
position_data_home_test.rename(columns={"POSITION_home": "POSITION"}, inplace=True)
position_data_away_test.rename(columns={"POSITION_away": "POSITION"}, inplace=True)

home_player = home_player.rename(columns={"ID_home": "ID"})
away_player = away_player.rename(columns={"ID_away": "ID"})
position_data_home = home_player[["ID", "POSITION_home"]].copy()
position_data_away = away_player[["ID", "POSITION_away"]].copy()
position_data_home.rename(columns={"POSITION_home": "POSITION"}, inplace=True)
position_data_away.rename(columns={"POSITION_away": "POSITION"}, inplace=True)

position_data = pd.concat([position_data_home, position_data_away], ignore_index=True)
position_data_test = pd.concat([position_data_home_test, position_data_away_test], ignore_index=True)

# Add the condition for minimum playing time (45 minutes average per season)
min_minutes_played = 45

# Normalize the POSITION column (lowercase for consistency)
home_player["POSITION_home"] = home_player["POSITION_home"].str.lower()
away_player["POSITION_away"] = away_player["POSITION_away"].str.lower()

valid_positions = ["defender", "midfielder", "attacker"]
# Filter valid positions and minimum playing time for home players
home_position_data = home_player[
    (home_player["POSITION_home"].isin(valid_positions)) &
    (home_player["PLAYER_MINUTES_PLAYED_season_average_home"] >= min_minutes_played)
].copy()

# Filter valid positions and minimum playing time for away players
away_position_data = away_player[
    (away_player["POSITION_away"].isin(valid_positions)) &
    (away_player["PLAYER_MINUTES_PLAYED_season_average_away"] >= min_minutes_played)
].copy()

# Count positions for home players
home_position_counts = (
    home_position_data.groupby("ID")["POSITION_home"]
    .value_counts()
    .unstack(fill_value=0)  # Convert to wide format with one column per position
    .reset_index()
)

# Rename columns for clarity
home_position_counts.rename(
    columns={"defender": "num_defenders_home", "midfielder": "num_midfielders_home", "attacker": "num_attackers_home"},
    inplace=True
)

# Count positions for away players
away_position_counts = (
    away_position_data.groupby("ID")["POSITION_away"]
    .value_counts()
    .unstack(fill_value=0)  # Convert to wide format with one column per position
    .reset_index()
)

# Rename columns for clarity
away_position_counts.rename(
    columns={"defender": "num_defenders_away", "midfielder": "num_midfielders_away", "attacker": "num_attackers_away"},
    inplace=True
)

# Repeat the process for test data (home and away)
home_playertest["POSITION_home"] = home_playertest["POSITION_home"].str.lower()
away_playertest["POSITION_away"] = away_playertest["POSITION_away"].str.lower()

# Filter valid positions and minimum playing time for home players in test set
home_position_data_test = home_playertest[
    (home_playertest["POSITION_home"].isin(valid_positions)) &
    (home_playertest["PLAYER_MINUTES_PLAYED_season_average_home"] >= min_minutes_played)
].copy()

# Filter valid positions and minimum playing time for away players in test set
away_position_data_test = away_playertest[
    (away_playertest["POSITION_away"].isin(valid_positions)) &
    (away_playertest["PLAYER_MINUTES_PLAYED_season_average_away"] >= min_minutes_played)
].copy()

# Count positions for home players in test set
home_position_counts_test = (
    home_position_data_test.groupby("ID")["POSITION_home"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns for clarity
home_position_counts_test.rename(
    columns={"defender": "num_defenders_home", "midfielder": "num_midfielders_home", "attacker": "num_attackers_home"},
    inplace=True
)

# Count positions for away players in test set
away_position_counts_test = (
    away_position_data_test.groupby("ID")["POSITION_away"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns for clarity
away_position_counts_test.rename(
    columns={"defender": "num_defenders_away", "midfielder": "num_midfielders_away", "attacker": "num_attackers_away"},
    inplace=True
)



#Preprocessing of the data.

home_player["POSITION"] = home_player["POSITION_home"].str.lower()
away_player["POSITION"] = away_player["POSITION_away"].str.lower()
home_playertest["POSITION"] = home_playertest["POSITION_home"].str.lower()
away_playertest["POSITION"] = away_playertest["POSITION_away"].str.lower()
columns_to_drop = [
    "LEAGUE_home", "TEAM_NAME_home", "PLAYER_NAME_home",
    "LEAGUE_away", "TEAM_NAME_away", "PLAYER_NAME_away",
]


#Handle NaN values:
nan_cutoff = 40
def filter_rows_by_nan(dataframe, nan_cutoff):
    row_nan_percentage = dataframe.isnull().mean(axis=1) * 100
    return dataframe[row_nan_percentage <= nan_cutoff]





def remove_columns_with_all_nans(df, threshold=0.3):
    """
    Removes columns from a DataFrame where the proportion of NaN values exceeds a specified threshold.
    Handles mixed data types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Proportion of NaNs to use as the cutoff.

    Returns:
        pd.DataFrame: DataFrame with columns removed based on the threshold.
    """
    # Identify numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculate the proportion of NaNs along columns in the numeric subset
    data_array = numeric_df.to_numpy()  # Convert only numeric columns to a NumPy array
    nan_proportion = np.isnan(data_array).mean(axis=0)

    # Identify numeric columns to keep based on the threshold
    numeric_columns_to_keep = numeric_df.columns[nan_proportion < threshold]

    # Combine with non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    final_columns_to_keep = numeric_columns_to_keep.tolist() + non_numeric_columns.tolist()

    # Return the filtered DataFrame
    return df[final_columns_to_keep]







# Apply the filters to each dataset
home_player2 = filter_rows_by_nan(home_player, nan_cutoff)
home_player2 = remove_columns_with_all_nans(home_player2)
away_player2 = filter_rows_by_nan(away_player, nan_cutoff)
away_player2 = remove_columns_with_all_nans(away_player)
home_playertest2 = filter_rows_by_nan(home_playertest, nan_cutoff)
home_playertest2 = remove_columns_with_all_nans(home_playertest)
away_playertest2 = filter_rows_by_nan(away_playertest, nan_cutoff)
away_playertest2 = remove_columns_with_all_nans(away_playertest)



# Step 1: Imputation and Scaling for Player Data
imputer = SimpleImputer(strategy= 'mean')  # Use median to handle outliers robustly
scaler = StandardScaler()  # Standardize features

# Impute and scale numeric columns for home and away player datasets
def preprocess_player_data(player_df, suffix):
    """
    Preprocess player data by handling missing values and scaling numeric features.
    Args:
        player_df (DataFrame): Raw player data.
        suffix (str): Suffix for home/away to differentiate columns.

    Returns:
        DataFrame: Processed player data with imputed and scaled features.
    """
    # Identify numeric columns
    numeric_columns = player_df.select_dtypes(include=["float64", "int64"]).columns

    # Identify columns with all missing values and exclude them
    all_missing_columns = numeric_columns[player_df[numeric_columns].isnull().all()]
    valid_columns = numeric_columns.difference(all_missing_columns)

    # Impute missing values for valid columns
    imputer = SimpleImputer(strategy="median")  # Impute only for partially missing columns
    imputed_data = pd.DataFrame(
        imputer.fit_transform(player_df[valid_columns]),
        columns=valid_columns,
        index=player_df.index
    )

    # Scale the numeric columns
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(imputed_data),
        columns=valid_columns,
        index=player_df.index
    )

    # Add back non-numeric columns (like ID and POSITION)
    scaled_data["ID"] = player_df["ID"]
    scaled_data[f"position_{suffix}"] = player_df[f"POSITION_{suffix}"]

    return scaled_data



home_player_scaled = preprocess_player_data(home_player2, "home")
away_player_scaled = preprocess_player_data(away_player2, "away")
home_player_test_scaled = preprocess_player_data(home_playertest2, "home")
away_player_test_scaled = preprocess_player_data(away_playertest2, "away")



# Step 2: Aggregation by Position
def aggregate_features_by_position(player_df, positions, suffix):
    features = [col for col in player_df.columns if col not in ["ID", f"position_{suffix}"]]
    valid_player_df = player_df[player_df[f"position_{suffix}"].isin(positions)].copy()
    aggregated_features = pd.DataFrame()
    aggregated_features["ID"] = valid_player_df["ID"].unique()

    for position in positions:
        position_df = valid_player_df[valid_player_df[f"position_{suffix}"] == position]

        for agg_method in ["mean", "sum", "max", "std"]:
            agg_df = position_df.groupby("ID")[features].agg(agg_method).reset_index()
            agg_df = agg_df.rename(
                columns={feature: f"{feature}_{agg_method}_{position}_{suffix}" for feature in features}
            )
            aggregated_features = aggregated_features.merge(agg_df, on="ID", how="left")

    return aggregated_features

# Valid positions
positions = ["goalkeeper", "defender", "midfielder", "attacker"]


# Aggregate features for home and away players
home_player_aggregated = aggregate_features_by_position(home_player_scaled, positions, "home")


home_player_aggregated = home_player_aggregated.merge(home_position_counts, on = "ID", how = "left")

away_player_aggregated = aggregate_features_by_position(away_player_scaled, positions, "away")
away_player_aggregated = away_player_aggregated.merge(away_position_counts, on = "ID", how = "left")
home_player_test_aggregated = aggregate_features_by_position(home_player_test_scaled, positions, "home")
home_player_test_aggregated = home_player_test_aggregated.merge(home_position_counts_test, on = "ID", how = "left")
away_player_test_aggregated = aggregate_features_by_position(away_player_test_scaled, positions, "away")
away_player_test_aggregated = away_player_test_aggregated.merge(away_position_counts_test, on = "ID", how = "left")


# Step 3: Impute Missing Aggregated Features
def impute_aggregated_features(aggregated_df):
    non_id_columns = [col for col in aggregated_df.columns if col != "ID"]
    aggregated_imputed = pd.DataFrame(
        imputer.fit_transform(aggregated_df[non_id_columns]),
        columns=non_id_columns
    )
    aggregated_imputed["ID"] = aggregated_df["ID"].values
    return aggregated_imputed

home_player_aggregated_imputed = impute_aggregated_features(home_player_aggregated)
away_player_aggregated_imputed = impute_aggregated_features(away_player_aggregated)
home_player_test_aggregated_imputed = impute_aggregated_features(home_player_test_aggregated)
away_player_test_aggregated_imputed = impute_aggregated_features(away_player_test_aggregated)



# Step 4: Scale Aggregated Features
def scale_aggregated_features(aggregated_imputed):
    non_id_columns = [col for col in aggregated_imputed.columns if col != "ID"]
    aggregated_scaled = pd.DataFrame(
        scaler.fit_transform(aggregated_imputed[non_id_columns]),
        columns=non_id_columns
    )
    aggregated_scaled["ID"] = aggregated_imputed["ID"].values
    return aggregated_scaled

home_player_scaled_final = scale_aggregated_features(home_player_aggregated_imputed)
away_player_scaled_final = scale_aggregated_features(away_player_aggregated_imputed)
home_player_test_scaled_final = scale_aggregated_features(home_player_test_aggregated_imputed)
away_player_test_scaled_final = scale_aggregated_features(away_player_test_aggregated_imputed)

# Step 5: Combine Home and Away Features for Final Dataset
train_player_features = home_player_scaled_final.merge(
    away_player_scaled_final, on="ID", how="inner"
)
test_player_features = home_player_test_scaled_final.merge(
    away_player_test_scaled_final, on="ID", how="inner"
)

# Ensure no missing values remain in the final datasets
train_player_features = pd.DataFrame(
    imputer.fit_transform(train_player_features),
    columns=train_player_features.columns
)


test_player_features = pd.DataFrame(
    imputer.fit_transform(test_player_features),
    columns=test_player_features.columns
)

home_features = [col for col in train_player_features.columns if col.endswith('_home')]
away_features = [col for col in train_player_features.columns if col.endswith('_away')]

# Identify matching features (ignoring '_home' and '_away' suffix)
common_feature_names = set(
    col.replace('_home', '').replace('_away', '') for col in home_features + away_features
)

# Initialize a DataFrame for differences
difference_features = pd.DataFrame()
difference_features_test = pd.DataFrame()
difference_features["ID"] = train_player_features["ID"]
difference_features_test["ID"] = test_player_features["ID"]
# Calculate differences for matching features
for feature_name in common_feature_names:
    home_feature = f"{feature_name}_home"
    away_feature = f"{feature_name}_away"
    if home_feature in train_player_features.columns and away_feature in train_player_features.columns:
        diff_feature = f"{feature_name}_diff"
        difference_features[diff_feature] = (
            train_player_features[home_feature] - train_player_features[away_feature]
        )
    if home_feature in test_player_features.columns and away_feature in train_player_features.columns:
        diff_feature = f"{feature_name}_diff"
        difference_features_test[diff_feature] = (test_player_features[home_feature] -test_player_features[away_feature])


train_player_features = train_player_features.merge(difference_features, on = "ID", how = "left")
test_player_features = test_player_features.merge(difference_features_test, on = "ID", how = "left")

away_teamtest = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/test_away_team_statistics_df.csv')
home_teamtest = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/test_home_team_statistics_df.csv')
away_team = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/train_away_team_statistics_df.csv')
home_team = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/train_home_team_statistics_df.csv')
train_output = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/train_output.csv')
goal_diff = pd.read_csv('/Users/miguelsanchez/Desktop/PycharmProjects/QRT/src/data/Y_train_supp.csv')



merged_data = home_team.merge(away_team, on='ID', how='inner', suffixes=('', '_away_team'))
merged_data = merged_data.merge(goal_diff, on='ID', how='inner')

merged_datatest = home_teamtest.merge(away_teamtest, on = 'ID', how = 'inner',suffixes=('', '_away_team') )
X_ids = merged_data["ID"]
X_ids_test = merged_datatest["ID"]
# Prepare team-level data for the baseline model
team_features = [col for col in merged_data.columns if col not in ["ID","LEAGUE", "LEAGUE_away_team", "TEAM_NAME",
                                                                   "TEAM_NAME_away_team","GOAL_DIFF_HOME_AWAY"]]
X_team = merged_data[team_features]
X_team_test = merged_datatest[team_features]



# Handle missing values
team_imputer = KNNImputer(n_neighbors=5)
X_team_imputed = pd.DataFrame(team_imputer.fit_transform(X_team), columns=team_features)
X_team_test_imputed = pd.DataFrame(team_imputer.fit_transform(X_team_test), columns = team_features)



# Scale features
team_scaler = StandardScaler()
X_team_scaled = pd.DataFrame(team_scaler.fit_transform(X_team_imputed), columns=team_features)
X_team_test_scaled = pd.DataFrame(team_scaler.fit_transform(X_team_test_imputed), columns = team_features)
X_team_scaled["ID"] = X_ids.values
X_team_test_scaled["ID"] = X_ids_test.values


'''
Selected features in  utils.py

Final Selected Features:
['PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_mean_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_season_average_home_mean_goalkeeper_home', 'PLAYER_REDCARDS_5_last_match_average_home_mean_goalkeeper_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_sum_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_home_sum_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_season_average_home_sum_goalkeeper_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_max_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_season_sum_home_max_goalkeeper_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_5_last_match_average_home_mean_defender_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_mean_defender_home', 'PLAYER_ACCURATE_PASSES_season_sum_home_mean_defender_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_5_last_match_average_home_sum_defender_home', 'PLAYER_ACCURATE_PASSES_season_sum_home_sum_defender_home', 'PLAYER_FOULS_DRAWN_5_last_match_std_home_sum_defender_home', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_home_sum_defender_home', 'PLAYER_GOALS_CONCEDED_season_average_home_sum_defender_home', 'PLAYER_KEY_PASSES_5_last_match_std_home_sum_defender_home', 'PLAYER_ACCURATE_PASSES_season_average_home_max_defender_home', 'PLAYER_ASSISTS_5_last_match_std_home_mean_midfielder_home', 'PLAYER_FOULS_season_average_home_mean_midfielder_home', 'PLAYER_GOALS_season_sum_home_mean_midfielder_home', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_home_sum_midfielder_home', 'PLAYER_REDCARDS_5_last_match_std_home_sum_midfielder_home', 'PLAYER_BIG_CHANCES_CREATED_season_average_home_max_midfielder_home', 'PLAYER_BIG_CHANCES_CREATED_season_sum_home_max_midfielder_home', 'PLAYER_GOALS_CONCEDED_season_average_home_max_midfielder_home', 'PLAYER_SHOTS_ON_TARGET_season_average_home_max_midfielder_home', 'PLAYER_BIG_CHANCES_CREATED_season_average_home_std_midfielder_home', 'PLAYER_DUELS_LOST_season_std_home_std_midfielder_home', 'PLAYER_GOALS_CONCEDED_season_sum_home_std_midfielder_home', 'PLAYER_TOTAL_DUELS_season_std_home_std_midfielder_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_5_last_match_average_away_mean_goalkeeper_away', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_away_mean_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_away_mean_goalkeeper_away', 'PLAYER_GOALKEEPER_GOALS_CONCEDED_5_last_match_sum_away_sum_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_5_last_match_average_away_sum_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_sum_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_season_average_away_sum_goalkeeper_away', 'PLAYER_KEY_PASSES_season_std_away_sum_goalkeeper_away', 'PLAYER_GOALKEEPER_GOALS_CONCEDED_season_average_away_max_goalkeeper_away', 'PLAYER_ACCURATE_PASSES_5_last_match_average_away_mean_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_sum_away_mean_defender_away', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_away_mean_defender_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_mean_defender_away', 'PLAYER_GOALS_CONCEDED_season_sum_away_mean_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_sum_away_sum_defender_away', 'PLAYER_ACCURATE_PASSES_season_sum_away_sum_defender_away', 'PLAYER_GOALS_CONCEDED_5_last_match_average_away_sum_defender_away', 'PLAYER_GOALS_CONCEDED_season_average_away_sum_defender_away', 'PLAYER_SHOTS_BLOCKED_season_average_away_sum_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_average_away_max_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_sum_away_max_defender_away', 'PLAYER_GOALS_CONCEDED_season_sum_away_max_defender_away', 'PLAYER_PASSES_5_last_match_sum_away_max_defender_away', 'PLAYER_ACCURATE_PASSES_season_average_away_mean_midfielder_away', 'PLAYER_GOALS_CONCEDED_season_average_away_mean_midfielder_away', 'PLAYER_PASSES_season_average_away_mean_midfielder_away', 'PLAYER_ACCURATE_PASSES_5_last_match_average_away_sum_midfielder_away', 'PLAYER_ACCURATE_PASSES_season_average_away_sum_midfielder_away', 'PLAYER_ACCURATE_PASSES_season_average_away_max_midfielder_away', 'PLAYER_GOALS_CONCEDED_season_std_away_max_midfielder_away', 'PLAYER_GOALS_CONCEDED_season_sum_away_max_midfielder_away', 'PLAYER_ASSISTS_season_average_away_mean_attacker_away', 'PLAYER_BIG_CHANCES_MISSED_season_average_away_mean_attacker_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_sum_attacker_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_max_attacker_away']


'''

# List of final selected features
final_selected_features = ['PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_mean_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_season_average_home_mean_goalkeeper_home', 'PLAYER_REDCARDS_5_last_match_average_home_mean_goalkeeper_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_sum_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_home_sum_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_season_average_home_sum_goalkeeper_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_max_goalkeeper_home', 'PLAYER_GOALS_CONCEDED_season_sum_home_max_goalkeeper_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_5_last_match_average_home_mean_defender_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_home_mean_defender_home', 'PLAYER_ACCURATE_PASSES_season_sum_home_mean_defender_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_5_last_match_average_home_sum_defender_home', 'PLAYER_ACCURATE_PASSES_season_sum_home_sum_defender_home', 'PLAYER_FOULS_DRAWN_5_last_match_std_home_sum_defender_home', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_home_sum_defender_home', 'PLAYER_GOALS_CONCEDED_season_average_home_sum_defender_home', 'PLAYER_KEY_PASSES_5_last_match_std_home_sum_defender_home', 'PLAYER_ACCURATE_PASSES_season_average_home_max_defender_home', 'PLAYER_ASSISTS_5_last_match_std_home_mean_midfielder_home', 'PLAYER_FOULS_season_average_home_mean_midfielder_home', 'PLAYER_GOALS_season_sum_home_mean_midfielder_home', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_home_sum_midfielder_home', 'PLAYER_REDCARDS_5_last_match_std_home_sum_midfielder_home', 'PLAYER_BIG_CHANCES_CREATED_season_average_home_max_midfielder_home', 'PLAYER_BIG_CHANCES_CREATED_season_sum_home_max_midfielder_home', 'PLAYER_GOALS_CONCEDED_season_average_home_max_midfielder_home', 'PLAYER_SHOTS_ON_TARGET_season_average_home_max_midfielder_home', 'PLAYER_BIG_CHANCES_CREATED_season_average_home_std_midfielder_home', 'PLAYER_DUELS_LOST_season_std_home_std_midfielder_home', 'PLAYER_GOALS_CONCEDED_season_sum_home_std_midfielder_home', 'PLAYER_TOTAL_DUELS_season_std_home_std_midfielder_home', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_5_last_match_average_away_mean_goalkeeper_away', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_away_mean_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_5_last_match_sum_away_mean_goalkeeper_away', 'PLAYER_GOALKEEPER_GOALS_CONCEDED_5_last_match_sum_away_sum_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_5_last_match_average_away_sum_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_sum_goalkeeper_away', 'PLAYER_GOALS_CONCEDED_season_average_away_sum_goalkeeper_away', 'PLAYER_KEY_PASSES_season_std_away_sum_goalkeeper_away', 'PLAYER_GOALKEEPER_GOALS_CONCEDED_season_average_away_max_goalkeeper_away', 'PLAYER_ACCURATE_PASSES_5_last_match_average_away_mean_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_sum_away_mean_defender_away', 'PLAYER_ACCURATE_PASSES_PERCENTAGE_season_average_away_mean_defender_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_mean_defender_away', 'PLAYER_GOALS_CONCEDED_season_sum_away_mean_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_sum_away_sum_defender_away', 'PLAYER_ACCURATE_PASSES_season_sum_away_sum_defender_away', 'PLAYER_GOALS_CONCEDED_5_last_match_average_away_sum_defender_away', 'PLAYER_GOALS_CONCEDED_season_average_away_sum_defender_away', 'PLAYER_SHOTS_BLOCKED_season_average_away_sum_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_average_away_max_defender_away', 'PLAYER_ACCURATE_PASSES_5_last_match_sum_away_max_defender_away', 'PLAYER_GOALS_CONCEDED_season_sum_away_max_defender_away', 'PLAYER_PASSES_5_last_match_sum_away_max_defender_away', 'PLAYER_ACCURATE_PASSES_season_average_away_mean_midfielder_away', 'PLAYER_GOALS_CONCEDED_season_average_away_mean_midfielder_away', 'PLAYER_PASSES_season_average_away_mean_midfielder_away', 'PLAYER_ACCURATE_PASSES_5_last_match_average_away_sum_midfielder_away', 'PLAYER_ACCURATE_PASSES_season_average_away_sum_midfielder_away', 'PLAYER_ACCURATE_PASSES_season_average_away_max_midfielder_away', 'PLAYER_GOALS_CONCEDED_season_std_away_max_midfielder_away', 'PLAYER_GOALS_CONCEDED_season_sum_away_max_midfielder_away', 'PLAYER_ASSISTS_season_average_away_mean_attacker_away', 'PLAYER_BIG_CHANCES_MISSED_season_average_away_mean_attacker_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_sum_attacker_away', 'PLAYER_GOALS_CONCEDED_5_last_match_std_away_max_attacker_away']


# Ensure 'ID' is always included in the selected features
selected_features1 = ['ID']

# Loop through the final_selected_features and check if they exist in train_player_features
for feature in final_selected_features:
    if feature in train_player_features.columns:
        selected_features1.append(feature)

# Filter the selected features from train_player_features
selected_features_df = train_player_features[selected_features1]
selected_features_test_df = test_player_features[selected_features1]

# Display the selected features for verification
print(f"Number of selected features: {len(selected_features1)}")
print(f"Selected features: {selected_features1}")



# Merge selected features into X_team_scaled
X_team_scaled_with_selected = pd.merge(
    X_team_scaled, selected_features_df, on="ID", how="left"
)

X_team_test_scaled_with_selected = pd.merge(
    X_team_test_scaled, selected_features_test_df, on = "ID", how = "left"
)

X_team_scaled_with_selected_ids = X_team_scaled_with_selected["ID"]
X_team_scaled_with_selected.drop(columns = ["ID"])
X_team_scaled_with_selected = pd.DataFrame(team_imputer.fit_transform(X_team_scaled_with_selected), columns = X_team_scaled_with_selected.columns)
X_team_scaled_with_selected["ID"] = X_team_scaled_with_selected_ids.values

X_team_test_scaled_with_selected_ids = X_team_test_scaled_with_selected["ID"]
X_team_test_scaled_with_selected.drop(columns = ["ID"])
X_team_test_scaled_with_selected = pd.DataFrame(team_imputer.fit_transform(X_team_test_scaled_with_selected), columns = X_team_test_scaled_with_selected.columns)
X_team_test_scaled_with_selected["ID"] = X_team_test_scaled_with_selected_ids.values

X_scaled = X_team_scaled_with_selected.copy()
X_scaled_test = X_team_test_scaled_with_selected.copy()


y = merged_data["GOAL_DIFF_HOME_AWAY"]
# Scale the target variable
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
y_classify = pd.DataFrame()
y_classify['Target'] = np.argmax(train_output[['HOME_WINS', 'DRAW', 'AWAY_WINS']].values, axis=1)


lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y_scaled)

# Select important features
selected_features = X_scaled.columns[(lasso.coef_ != 0)]
X_scaled_selected = X_scaled[selected_features]
X_scaled_test_selected = X_scaled_test[selected_features]

# Step 2: Regression with XGBRegressor and Optuna for Hyperparameter Tuning
from sklearn.model_selection import train_test_split

# Step 1: Split data into training+validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_selected, y_scaled, test_size=0.2, random_state=42)

def regression_objective(trial):
    # Define hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 100),
        "eval_metric": "rmse"
    }

    regressor = XGBRegressor(**params, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        regressor.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False,
        )

        y_val_pred = regressor.predict(X_val_cv)
        fold_mse = mean_squared_error(y_val_cv, y_val_pred)
        cv_mse_scores.append(fold_mse)

    return np.mean(cv_mse_scores)

# Step 2: Hyperparameter tuning with Optuna
study = optuna.create_study(direction="minimize")
study.optimize(regression_objective, n_trials=100)

# Step 3: Train final model with optimal hyperparameters
best_params = study.best_params
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Use the test set as a validation set during training
    verbose=True  # Enable verbose output to monitor progress
)


# Step 4: Evaluate on the reserved test set
y_test_pred = final_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("Final Test MSE:", test_mse)
print("Final Test r2:", test_r2)



# Generate predictions for the training set using out-of-fold predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_goal_diff_predictions = np.zeros(len(X_train))  # Out-of-fold predictions
test_goal_diff_predictions = np.zeros(len(X_test))    # Averaged across folds for test set
full_predictions = np.zeros(len(X_scaled_selected))
test_selected_predictions = np.zeros(len(X_scaled_test_selected))
for train_index, val_index in kf.split(X_train):
    X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

    # Train the model on training fold
    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train_cv, y_train_cv,eval_set=[(X_val_cv, y_val_cv)],)

    # Predict on validation fold (out-of-fold predictions)
    train_goal_diff_predictions[val_index] = model.predict(X_val_cv)

    # Predict on the test set (averaging predictions across folds)
    test_goal_diff_predictions += model.predict(X_test) / kf.n_splits
    test_selected_predictions += model.predict(X_scaled_test_selected) / kf.n_splits

# Add predictions back to the dataset
full_predictions[X_train.index] = train_goal_diff_predictions

# Assign averaged predictions for test data
full_predictions[X_test.index] = test_goal_diff_predictions

# Add predictions to X_scaled_selected
X_scaled_selected["Predicted_Goal_Diff"] = full_predictions
X_scaled_test_selected["Predicted_Goal_Diff"] = test_selected_predictions


# Step 3: Classification with XGBClassifier and Optuna for Hyperparameter Tuning
y_classify = y_classify["Target"]

def classification_objective(trial):
    # Define hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5.0),  # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5.0),  # L2 regularization
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 10, 100),
        "eval_metric": "mlogloss"

    }

    # Initialize the classifier
    classifier = XGBClassifier(**params, random_state=42)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy_scores = []

    for train_index, val_index in skf.split(X_scaled_selected, y_classify):
        X_train_cv, X_val_cv = X_scaled_selected.iloc[train_index], X_scaled_selected.iloc[val_index]
        y_train_cv, y_val_cv = y_classify[train_index], y_classify[val_index]

        # Train the model
        classifier.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False,)

        # Predict on the validation fold
        y_val_pred = classifier.predict(X_val_cv)

        # Calculate accuracy for the fold
        fold_accuracy = accuracy_score(y_val_cv, y_val_pred)
        cv_accuracy_scores.append(fold_accuracy)

    # Return the mean CV accuracy as the optimization target
    return 1 - np.mean(cv_accuracy_scores)  # Minimize the error

# Split data for tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled_selected, y_classify, test_size=0.2, random_state=42
)

# Run Optuna for classification
study_cls = optuna.create_study(direction="minimize")
study_cls.optimize(classification_objective, n_trials=50, callbacks=[lambda study, trial: print(f"Trial {trial.number} complete with value: {trial.value}")])

# Train the final classifier on the full training dataset with the best parameters
best_params_cls = study_cls.best_params
classifier = XGBClassifier(**best_params_cls, random_state=42)
classifier.fit(X_train, y_train,
               eval_set = [(X_val, y_val)],
               verbose =True)


# Generate out-of-fold predictions for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays for predictions
train_classification_predictions = np.zeros(len(X_scaled_selected))  # Out-of-fold predictions
train_classification_probabilities = np.zeros((len(X_scaled_selected), 3))  # Probabilities for all rows
test_classification_probabilities = np.zeros((len(X_scaled_test_selected), 3))  # For probabilities of 3 classes

fold_accuracies = []
fold_log_losses = []

for train_index, val_index in skf.split(X_scaled_selected, y_classify):
    X_train_cv, X_val_cv = X_scaled_selected.iloc[train_index], X_scaled_selected.iloc[val_index]
    y_train_cv, y_val_cv = y_classify[train_index], y_classify[val_index]

    # Train the model for this fold
    fold_model = XGBClassifier(**best_params_cls, random_state=42)
    fold_model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)

    # Predict on validation fold (out-of-fold predictions)
    y_val_pred = fold_model.predict(X_val_cv)
    y_val_proba = fold_model.predict_proba(X_val_cv)
    train_classification_predictions[val_index] = y_val_pred
    train_classification_probabilities[val_index] = y_val_proba

    # Evaluate metrics for this fold
    fold_accuracy = accuracy_score(y_val_cv, y_val_pred)
    fold_log_loss = log_loss(y_val_cv, y_val_proba)
    fold_accuracies.append(fold_accuracy)
    fold_log_losses.append(fold_log_loss)

    # Predict on the test set (averaging across folds)
    test_classification_probabilities += fold_model.predict_proba(X_scaled_test_selected) / skf.n_splits

# Log and print metrics for cross-validation

test_classification_predictions = np.argmax(test_classification_probabilities, axis=1)
print("Cross-validation metrics:")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Mean Log Loss: {np.mean(fold_log_losses):.4f}")

# Final Training Metrics
train_accuracy = accuracy_score(y_classify, train_classification_predictions)
train_log_loss = log_loss(y_classify, train_classification_probabilities)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Training Log Loss: {train_log_loss:.4f}")

# Assign predictions to test dataset
X_scaled_test_selected["Predicted_Class"] = test_classification_predictions
X_scaled_test_selected["Class_Probabilities"] = list(test_classification_probabilities)



# Add out-of-fold predictions to the dataset
X_scaled_selected["Predicted_Class"] = train_classification_predictions
X_scaled_test_selected["Predicted_Class"] = test_classification_predictions
y_test_pred_final = test_classification_predictions.astype(int)
X_scaled_test_selected["Predicted_Prob_Home_Wins"] = test_classification_probabilities[:, 0]
X_scaled_test_selected["Predicted_Prob_Draw"] = test_classification_probabilities[:, 1]
X_scaled_test_selected["Predicted_Prob_Away_Wins"] = test_classification_probabilities[:, 2]

one_hot_predictions = pd.DataFrame(
    np.eye(3, dtype=int)[y_test_pred_final],  # Ensure one-hot encoding
    columns=["HOME_WINS", "DRAW", "AWAY_WINS"]
)

# Combine IDs with predictions
submission_df = pd.concat([X_ids_test.reset_index(drop=True), one_hot_predictions], axis=1)

# Save to CSV in the required format
submission_file_path = "/Users/miguelsanchez/Desktop/PycharmProjects/QRT/submission2.csv"
submission_df.to_csv(submission_file_path, index=False)

