import pandas as pd

def load_data():
    """
    Load raw data files into pandas DataFrames.

    Returns:
        DataFrames for home_player, away_player, home_team, away_team, train_output.
    """
    home_player = pd.read_csv("data/train_home_player_statistics_df.csv")
    away_player = pd.read_csv("data/train_away_player_statistics_df.csv")
    home_team = pd.read_csv("data/train_home_team_statistics_df.csv")
    away_team = pd.read_csv("data/train_away_team_statistics_df.csv")
    train_output = pd.read_csv("data/train_output.csv")
    return home_player, away_player, home_team, away_team, train_output
