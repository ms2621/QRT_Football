import pandas as pd

def aggregate_features_by_position(player_df, positions, suffix):
    """
    Aggregate player features by position.

    Args:
        player_df (DataFrame): Processed player DataFrame.
        positions (list): List of valid positions to aggregate.
        suffix (str): Home/Away suffix for column differentiation.

    Returns:
        Aggregated DataFrame.
    """
    features = [col for col in player_df.columns if col not in ["ID", f"position_{suffix}"]]
    valid_player_df = player_df[player_df[f"position_{suffix}"].isin(positions)].copy()
    aggregated_features = pd.DataFrame({"ID": valid_player_df["ID"].unique()})

    for position in positions:
        position_df = valid_player_df[valid_player_df[f"position_{suffix}"] == position]
        for agg_method in ["mean", "sum", "max"]:
            agg_df = position_df.groupby("ID")[features].agg(agg_method).reset_index()
            agg_df = agg_df.rename(columns={col: f"{col}_{agg_method}_{position}_{suffix}" for col in features})
            aggregated_features = aggregated_features.merge(agg_df, on="ID", how="left")
    return aggregated_features
