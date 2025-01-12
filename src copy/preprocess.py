import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(player_df, suffix):
    """
    Preprocess player data: impute missing values and scale numeric features.

    Args:
        player_df (DataFrame): Input player statistics DataFrame.
        suffix (str): Indicates whether the data is for home/away players.

    Returns:
        Processed DataFrame with scaled features.
    """
    numeric_columns = player_df.select_dtypes(include=["float64", "int64"]).columns
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    imputed_data = pd.DataFrame(
        imputer.fit_transform(player_df[numeric_columns]),
        columns=numeric_columns,
        index=player_df.index
    )
    scaled_data = pd.DataFrame(
        scaler.fit_transform(imputed_data),
        columns=numeric_columns,
        index=player_df.index
    )
    scaled_data["ID"] = player_df["ID"]
    scaled_data[f"position_{suffix}"] = player_df[f"POSITION_{suffix}"]
    return scaled_data
