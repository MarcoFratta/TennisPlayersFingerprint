# -*- coding: utf-8 -*-
"""
Data Loading Module for Tennis Match Charting Project

This module contains all functions related to loading and processing
tennis match charting data from the Match Charting Project dataset.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_categorized_stats_files(gender='m', base_path='tennis_MatchChartingProject-master'):
    """
    Load pre-aggregated stats files with proper row categorization handling

    Args:
        gender: 'm' for men, 'w' for women
        base_path: Base path to the tennis dataset files

    Returns:
        Dictionary with processed DataFrames for each stats category
    """
    stats_data = {}

    # Overview: Only "Total" rows (no set-by-set breakdown)
    # We do not want set-by-set info
    try:
        overview_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-Overview.csv')
        overview_df['row'] = overview_df['set']
        overview_df.drop(columns=['set'], inplace=True)
        stats_data['overview'] = overview_df[overview_df['row'] == 'Total'].copy()
        print(f"Overview: {len(stats_data['overview'])} total records loaded")
    except FileNotFoundError:
        print(f"Overview file not found for gender {gender}")
        stats_data['overview'] = pd.DataFrame()

    # ServeBasics: Differentiate 1st and 2nd serve (exclude "Total")
    try:
        serve_basics_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ServeBasics.csv')
        stats_data['serve_basics'] = serve_basics_df[serve_basics_df['row'].isin(['1', '2'])].copy()
        print(f"ServeBasics: {len(stats_data['serve_basics'])} records (1st/2nd serve split)")
    except FileNotFoundError:
        print(f"ServeBasics file not found for gender {gender}")
        stats_data['serve_basics'] = pd.DataFrame()

    # Rally: Keep rally length differentiation (exclude "Total")
    try:
        rally_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-Rally.csv')
        rally_categories = ['1-3', '4-6', '7-9', '10']
        rally_df['player'] = rally_df['server']
        rally_df.drop(columns=['server'], inplace=True)
        stats_data['rally'] = rally_df[rally_df['row'].isin(rally_categories)].copy()
        print(f"Rally: {len(stats_data['rally'])} records across rally lengths {rally_categories}")
    except FileNotFoundError:
        print(f"Rally file not found for gender {gender}")
        stats_data['rally'] = pd.DataFrame()

    # ShotDirection: Keep shot type differentiation (f, b, s, etc.)
    try:
        shot_dir_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ShotDirection.csv')
        # Exclude "Total" to keep shot type breakdown
        shot_types = shot_dir_df['row'].unique()
        shot_types = [t for t in shot_types if t != 'Total']
        stats_data['shot_direction'] = shot_dir_df[shot_dir_df['row'].isin(shot_types)].copy()
        print(f"ShotDirection: {len(stats_data['shot_direction'])} records across shot types {shot_types}")
    except FileNotFoundError:
        print(f"ShotDirection file not found for gender {gender}")
        stats_data['shot_direction'] = pd.DataFrame()

    # ReturnOutcomes: Keep all differentiation (v1st/v2nd, fh/bh, court positions, game scores)
    try:
        return_outcomes_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ReturnOutcomes.csv')
        # Exclude "Total" to keep detailed breakdown
        return_categories = return_outcomes_df['row'].unique()
        stats_data['return_outcomes'] = return_outcomes_df[return_outcomes_df['row'].isin(return_categories)].copy()
        print(f"ReturnOutcomes: {len(stats_data['return_outcomes'])} records across categories {return_categories[:10]}...")
    except FileNotFoundError:
        print(f"ReturnOutcomes file not found for gender {gender}")
        stats_data['return_outcomes'] = pd.DataFrame()

    # ReturnDepth: Keep all differentiation (return depth categories)
    try:
        return_depth_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ReturnDepth.csv')
        # Exclude "Total" to keep detailed breakdown
        return_depth_categories = return_depth_df['row'].unique()
        return_depth_categories = [t for t in return_depth_categories if t != 'Total']
        stats_data['return_depth'] = return_depth_df[return_depth_df['row'].isin(return_depth_categories)].copy()
        print(f"ReturnDepth: {len(stats_data['return_depth'])} records across categories {return_depth_categories}")
    except FileNotFoundError:
        print(f"ReturnDepth file not found for gender {gender}")
        stats_data['return_depth'] = pd.DataFrame()

    # ShotTypes: Keep differentiation
    try:
        shot_types_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ShotTypes.csv')
        shot_type_categories = shot_types_df['row'].unique()
        stats_data['shot_types'] = shot_types_df[shot_types_df['row'].isin(shot_type_categories)].copy()
        print(f"ShotTypes: {len(stats_data['shot_types'])} records across types {shot_type_categories}")
    except FileNotFoundError:
        print(f"ShotTypes file not found for gender {gender}")
        stats_data['shot_types'] = pd.DataFrame()

    # SnV: Keep serve & volley differentiation
    try:
        snv_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-SnV.csv')
        snv_categories = snv_df['row'].unique()
        stats_data['snv'] = snv_df.copy()
        print(f"SnV: {len(stats_data['snv'])} records across categories {snv_categories}")
    except FileNotFoundError:
        print(f"SnV file not found for gender {gender}")
        stats_data['snv'] = pd.DataFrame()

    # ServeDirection: Keep 1st/2nd serve differentiation
    try:
        serve_dir_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ServeDirection.csv')
        serve_dir_categories = serve_dir_df['row'].unique()
        serve_dir_categories = [t for t in serve_dir_categories if t != 'Total']
        stats_data['serve_direction'] = serve_dir_df[serve_dir_df['row'].isin(serve_dir_categories)].copy()
        print(f"ServeDirection: {len(stats_data['serve_direction'])} records for 1st/2nd serves")
        print("ServeDirection row values: '1' = first serve, '2' = second serve")
    except FileNotFoundError:
        print(f"ServeDirection file not found for gender {gender}")
        stats_data['serve_direction'] = pd.DataFrame()

    # ServeInfluence: Keep 1st/2nd serve differentiation
    try:
        serve_inf_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ServeInfluence.csv')
        serve_inf_categories = serve_inf_df['row'].unique()
        stats_data['serve_influence'] = serve_inf_df.copy()
        print(f"ServeInfluence: {len(stats_data['serve_influence'])} records for 1st/2nd serves")
        print("ServeInfluence columns: won_1+ to won_10+ = % points won when rally reaches 1+ to 10+ shots")
        print("ServeInfluence row values: '1' = first serve influence, '2' = second serve influence")
    except FileNotFoundError:
        print(f"ServeInfluence file not found for gender {gender}")
        stats_data['serve_influence'] = pd.DataFrame()

    # ShotDirOutcome: Keep differentiation
    try:
        shot_dir_outcome_df = pd.read_csv(f'{base_path}/charting-{gender}-stats-ShotDirOutcomes.csv')
        if not shot_dir_outcome_df.empty:
            shot_dir_outcome_categories = shot_dir_outcome_df['row'].unique()
            stats_data['shot_dir_outcome'] = shot_dir_outcome_df.copy()
            print(f"ShotDirOutcome: {len(stats_data['shot_dir_outcome'])} records")
            print("ShotDirOutcome contains shot direction vs outcome correlations")
        else:
            stats_data['shot_dir_outcome'] = pd.DataFrame()
    except FileNotFoundError:
        print(f"ShotDirOutcome file not found for gender {gender}")
        stats_data['shot_dir_outcome'] = pd.DataFrame()

    return stats_data


def aggregate_overview(df):
    """
    Aggregate overview statistics with percentage calculations
    Overview contains total match statistics
    Percentages calculated as: (stat / pts) * 100
    """
    results = []

    for player in df['player'].unique():
        player_data = df[df['player'] == player]

        if len(player_data) == 0:
            continue

        percentages = []
        for _, ma in player_data.iterrows():
            pts = ma['serve_pts'] + ma['return_pts']
            if pts > 0:
                match_percentages = {
                    'dfs_pct': (ma['dfs'] / ma['serve_pts']) * 100,
                    'first_in_pct': (ma['first_in'] / ma['serve_pts']) * 100,
                    'second_in_pct': (ma['second_in'] / ma['serve_pts']) * 100,
                }
                percentages.append(match_percentages)

        if percentages:
            pct_df = pd.DataFrame(percentages)
            result = {
                'player': player,
                'matches_count': len(player_data),
                'type': 'overview'
            }

            for col in pct_df.columns:
                result[f'{col}_mean'] = pct_df[col].mean()
                result[f'{col}_std'] = pct_df[col].std()

            results.append(result)

    return pd.DataFrame(results)


def aggregate_serve_basics(df):
    """
    Aggregate serve basics statistics with percentage calculations
    Row '1' = first serve, Row '2' = second serve
    Percentages calculated as: (stat / pts) * 100
    """
    results = []

    for player in df['player'].unique():
        player_data = df[df['player'] == player]

        for row_type in ['1', '2']:
            row_data = player_data[player_data['row'] == row_type]

            if len(row_data) == 0:
                continue

            serve_type = 'first_serve' if row_type == '1' else 'second_serve'

            # Calculate percentages for each match
            percentages = []
            for _, match in row_data.iterrows():
                pts = match['pts']
                if pts > 0:
                    match_percentages = {
                        'aces_pct': (match['aces'] / pts) * 100,
                        'unret_pct': (match['unret'] / pts) * 100,
                        'forced_err_pct': (match['forced_err'] / pts) * 100,
                        'wide_pct': (match['wide'] / pts) * 100,
                        'body_pct': (match['body'] / pts) * 100,
                        't_pct': (match['t'] / pts) * 100
                    }
                    percentages.append(match_percentages)

            if percentages:
                # Calculate mean and std for each percentage
                pct_df = pd.DataFrame(percentages)
                result = {
                    'player': player,
                    'type': serve_type,
                    'matches_count': len(row_data)
                }

                for col in pct_df.columns:
                    result[f'{col}_mean'] = pct_df[col].mean()
                    result[f'{col}_std'] = pct_df[col].std()

                results.append(result)

    return pd.DataFrame(results)


def aggregate_rally(df):
    """
    Aggregate rally statistics with percentage calculations
    Row categories: '1-3', '4-6', '7-9', '10' (rally lengths)
    Each row contains stats for both server (pl1) and returner (pl2)
    Server and returner statistics are merged into single statistics per player
    Percentages calculated as: (stat / pts) * 100
    """
    results = []

    # Get all unique players (both servers and returners)
    all_players = set(df['player'].unique())
    all_returners = set(df['returner'].unique())
    all_players.update(all_returners)

    for player in all_players:
        for rally_length in ['1-3', '4-6', '7-9', '10']:
            # Get all matches where this player was either server or returner
            server_matches = df[(df['player'] == player) & (df['row'] == rally_length)]
            returner_matches = df[(df['returner'] == player) & (df['row'] == rally_length)]

            if len(server_matches) == 0 and len(returner_matches) == 0:
                continue

            # Collect all statistics for this player in this rally length
            all_percentages = []

            # Add server statistics (pl1 stats)
            for _, match in server_matches.iterrows():
                pts = match['pts']
                if pts > 0:
                    match_percentages = {
                        'winners_pct': (match['pl1_winners'] / pts) * 100,
                        'forced_pct': (match['pl1_forced'] / pts) * 100,
                        'unforced_pct': (match['pl1_unforced'] / pts) * 100
                    }
                    all_percentages.append(match_percentages)

            # Add returner statistics (pl2 stats)
            for _, match in returner_matches.iterrows():
                pts = match['pts']
                if pts > 0:
                    match_percentages = {
                        'winners_pct': (match['pl2_winners'] / pts) * 100,
                        'forced_pct': (match['pl2_forced'] / pts) * 100,
                        'unforced_pct': (match['pl2_unforced'] / pts) * 100
                    }
                    all_percentages.append(match_percentages)

            # Create result for this player and rally length
            if all_percentages:
                pct_df = pd.DataFrame(all_percentages)
                result = {
                    'player': player,
                    'type': rally_length,
                    'matches_count': len(server_matches) + len(returner_matches)
                }

                # Calculate mean and std for each percentage
                for col in pct_df.columns:
                    result[f'{col}_mean'] = pct_df[col].mean()
                    result[f'{col}_std'] = pct_df[col].std()

                results.append(result)

    return pd.DataFrame(results)


def aggregate_shot_direction(df):
    """
    Aggregate shot direction statistics with percentage calculations
    Row categories: 'F' (forehand), 'B' (backhand), 'S' (serve)
    Percentages calculated as: (stat / pts) * 100
    """
    results = []

    for player in df['player'].unique():
        player_data = df[df['player'] == player]

        for shot_type in ['F', 'B']:
            shot_data = player_data[player_data['row'] == shot_type]

            if len(shot_data) == 0:
                continue

            percentages = []
            for _, ma in shot_data.iterrows():
                pts = ma['crosscourt'] + ma['down_middle'] + ma['down_the_line'] + ma['inside_out'] + ma['inside_in']
                if pts > 0:
                    match_percentages = {
                        'crosscourt_pct': (ma['crosscourt'] / pts) * 100,
                        'down_middle_pct': (ma['down_middle'] / pts) * 100,
                        'down_the_line_pct': (ma['down_the_line'] / pts) * 100,
                        'inside_out_pct': (ma['inside_out'] / pts) * 100,
                        'inside_in_pct': (ma['inside_in'] / pts) * 100
                    }
                    percentages.append(match_percentages)

            if percentages:
                pct_df = pd.DataFrame(percentages)
                result = {
                    'player': player,
                    'type': shot_type,
                    'matches_count': len(shot_data)
                }

                for col in pct_df.columns:
                    result[f'{col}_mean'] = pct_df[col].mean()
                    result[f'{col}_std'] = pct_df[col].std()

                results.append(result)

    return pd.DataFrame(results)


def aggregate_return_depth(df):
    """
    Aggregate return depth statistics with percentage calculations
    Row categories: 'fh' (forehand), 'bh' (backhand)
    Percentages calculated as: (stat / pts) * 100
    """
    results = []

    for player in df['player'].unique():
        player_data = df[df['player'] == player]

        for shot_type in ['fh', 'bh']:
            shot_data = player_data[player_data['row'] == shot_type]

            if len(shot_data) == 0:
                continue

            percentages = []
            for _, ma in shot_data.iterrows():
                # number of in-play returns
                pts = ma['shallow'] + ma['deep'] + ma['very_deep']
                if pts > 0:
                    match_percentages = {
                        'shallow_pct': (ma['shallow'] / pts) * 100,
                        'deep_pct': (ma['deep'] / pts) * 100,
                        'very_deep_pct': (ma['very_deep'] / pts) * 100,
                        'unforced_pct': (ma['unforced'] / (pts + ma['unforced'])) * 100,
                        'err_net_pct': (ma['err_net'] / ma['unforced']) * 100 if ma['unforced'] else 0,
                        'err_deep_pct': (ma['err_deep'] / ma['unforced']) * 100 if ma['unforced'] else 0,
                        'err_wide_pct': (ma['err_wide'] / ma['unforced']) * 100 if ma['unforced'] else 0,
                        'err_wide_deep_pct': (ma['err_wide_deep'] / ma['unforced']) * 100 if ma['unforced'] else 0
                    }
                    percentages.append(match_percentages)

            if percentages:
                pct_df = pd.DataFrame(percentages)
                result = {
                    'player': player,
                    'type': shot_type,
                    'matches_count': len(shot_data)
                }

                for col in pct_df.columns:
                    result[f'{col}_mean'] = pct_df[col].mean()
                    result[f'{col}_std'] = pct_df[col].std()

                results.append(result)

    return pd.DataFrame(results)


def create_total_shots_dict(shot_types_df):
    """
    Creates a dictionary with total shots per match and player by filtering for 'Total' row.

    Args:
        shot_types_df: DataFrame containing shot types data

    Returns:
        Dictionary where keys are (match_id, player) tuples and values are total_shots.
    """
    # Filter for 'Total' row
    total_shots_df = shot_types_df[shot_types_df['row'] == 'Total'].copy()

    # Create dictionary from match_id, player, and shots
    total_shots_dict = total_shots_df.set_index(['match_id', 'player'])['shots'].to_dict()

    return total_shots_dict


def aggregate_shot_types(df):
    """
    Aggregate shot types statistics with percentage calculations
    Percentages calculated as: (stat / total_shots) * 100
    """
    results = []
    total_shots_dict = create_total_shots_dict(df)

    for player in df['player'].unique():
        player_data = df[df['player'] == player].copy()

        for shot_type in ['Gs', 'Sl', 'Dr', 'Vo','Ov','Hv','Lo', 'Sw', 'F', 'B', 'R', 'S']: #'V', 'Z', 'O', 'P', 'U', 'Y',
                          #'L', 'M', 'H', 'U', 'I', 'J', 'K']:
            shot_data = player_data[player_data['row'] == shot_type].copy()

            if shot_data.empty:
                continue

            percentages = []
            for _, ma in shot_data.iterrows():
                # Get the total shots for the current match and player using dictionary lookup
                total_shots= total_shots_dict.get((ma['match_id'], ma['player']), 0)
                if total_shots > 0:
                    match_percentages = {
                        'shots_pct': (ma['shots'] / total_shots) * 100,
                        'winners_pct': (ma['winners'] / ma['shots']) * 100 if ma['shots'] > 0 else 0,
                        'unforced_pct': (ma['unforced'] / ma['shots']) * 100 if ma['shots'] > 0 else 0,
                        'serve_return_pct': (ma['serve_return'] / total_shots) * 100 if total_shots > 0 else 0,
                    }
                    percentages.append(match_percentages)

            if percentages:
                pct_df = pd.DataFrame(percentages)
                result = {
                    'player': player,
                    'type': shot_type,
                    'matches_count': len(shot_data)
                }

                for col in pct_df.columns:
                    result[f'{col}_mean'] = pct_df[col].mean()
                    result[f'{col}_std'] = pct_df[col].std()

                results.append(result)

    return pd.DataFrame(results)


def merge_dataset_by_shot_type(df, matches_threshold=None):
    """
    Merge all rows for the same player in a dataset by prefixing column names with type values.
    Optionally filter out shot types with matches_count below a threshold.

    Args:
        df: DataFrame with player, type, and other columns
        matches_threshold: minimum matches_count for a row to be included (None = no filter)

    Returns:
        DataFrame with one row per player, columns prefixed with type
    """
    if df.empty:
        return pd.DataFrame()

    if 'type' not in df.columns:
        print(f"Warning: No 'type' column found in dataset. Available columns: {df.columns.tolist()}")
        return df

    # Apply matches_count filtering if threshold is given
    if matches_threshold is not None and 'matches_count' in df.columns:
        before_count = len(df)
        df = df[df['matches_count'] >= matches_threshold]
        after_count = len(df)
        print(f"Filtered rows by matches_count >= {matches_threshold}: {before_count} → {after_count}")

    players = df['player'].unique()
    merged_data = []

    for player in players:
        player_data = df[df['player'] == player]

        if player_data.empty:
            continue

        result = {'player': player}

        # Add total matches_count across all kept shot types
        if 'matches_count' in player_data.columns:
            result['matches_count'] = player_data['matches_count'].sum()

        for _, row in player_data.iterrows():
            type_value = row['type']
            for col in row.index:
                if col not in ['player', 'matches_count', 'type']:
                    new_col_name = f"{type_value}_{col}"
                    result[new_col_name] = row[col]

        merged_data.append(result)

    return pd.DataFrame(merged_data)


def merge_all_datasets(*datasets, matches_threshold = 10):
    """
    Merge all datasets together by player, creating a single row per player.
    Only keeps players that have data in ALL provided datasets.

    Args:
        *datasets: Variable number of DataFrames to merge
        matches_threshold: minimum matches_count for filtering

    Returns:
        DataFrame with one row per player containing all statistics
    """
    if not datasets:
        return pd.DataFrame()

    # Auto-generate dataset names for logging
    dataset_names = [f'dataset_{i}' for i in range(len(datasets))]

    # Create datasets dictionary for processing
    datasets_dict = dict(zip(dataset_names, datasets))

    # First, merge each dataset by shot type
    merged_datasets = {}

    for dataset_name, df in datasets_dict.items():
        print(f"Processing {dataset_name}...")
        if not df.empty:
            merged_df = merge_dataset_by_shot_type(df, matches_threshold=matches_threshold)
            if not merged_df.empty:
                merged_datasets[dataset_name] = merged_df
                print(f"  {dataset_name}: {len(merged_df)} players")

    if not merged_datasets:
        print("No datasets to merge")
        return pd.DataFrame()

    # Find players present in ALL datasets
    all_players_sets = [set(df['player'].unique()) for df in merged_datasets.values()]
    common_players = set.intersection(*all_players_sets)

    print(f"\nPlayers in each dataset:")
    for i, (dataset_name, df) in enumerate(merged_datasets.items()):
        players_in_dataset = set(df['player'].unique())
        print(f"  {dataset_name}: {len(players_in_dataset)} players")

    print(f"\nPlayers present in ALL {len(merged_datasets)} datasets: {len(common_players)}")

    # Filter each dataset to only include common players
    filtered_datasets = {}
    for dataset_name, df in merged_datasets.items():
        filtered_df = df[df['player'].isin(common_players)].copy()
        filtered_datasets[dataset_name] = filtered_df
        print(f"  {dataset_name} after filtering: {len(filtered_df)} players")

    # Merge all datasets by player
    print("\nMerging all datasets...")

    # Start with the first dataset (drop matches_count)
    final_df = list(filtered_datasets.values())[0].drop(columns=['matches_count'], errors='ignore')

    # Merge with remaining datasets
    for i, (dataset_name, df) in enumerate(list(filtered_datasets.items())[1:], 1):
        print(f"Merging {dataset_name}...")

        # Drop matches_count from all datasets
        df_to_merge = df.drop(columns=['matches_count'], errors='ignore')

        # Use suffixes to handle any other potential column conflicts
        final_df = pd.merge(final_df, df_to_merge, on='player', how='inner', suffixes=('', f'_{dataset_name}'))
        print(f"  After merging {dataset_name}: {len(final_df)} players")

    print(f"\nFinal dataset: {len(final_df)} players, {len(final_df.columns)} features")

    return final_df


def analyze_missing_values(df, dataset_name="Dataset"):
    """
    Analyze missing values in the merged dataset to understand data completeness.

    Args:
        df: Merged DataFrame
        dataset_name: Name for display purposes
    """
    print(f"\n=== {dataset_name} Missing Values Analysis ===")

    # Count missing values per column
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    # Show columns with missing values
    columns_with_missing = missing_counts[missing_counts > 0]

    if len(columns_with_missing) > 0:
        print(f"Columns with missing values: {len(columns_with_missing)}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Players with complete data: {len(df.dropna())}")

        # Show top 10 columns with most missing values
        print("\nTop 10 columns with most missing values:")
        missing_df = pd.DataFrame({
            'missing_count': missing_counts,
            'missing_percentage': missing_percentages
        }).sort_values('missing_count', ascending=False)

        print(missing_df.head(10))

        # Show which shot types have the most missing data
        print("\nMissing values by shot type (first part of column name):")
        shot_type_missing = {}
        for col in df.columns:
            if '_' in col:
                shot_type = col.split('_')[0]
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    if shot_type not in shot_type_missing:
                        shot_type_missing[shot_type] = 0
                    shot_type_missing[shot_type] += missing_count

        for shot_type, count in sorted(shot_type_missing.items(), key=lambda x: x[1], reverse=True):
            print(f"  {shot_type}: {count} missing values")
    else:
        print("No missing values found!")


def fix_missing_values(df, dataset_name="Dataset", strategy='median'):
    """
    Return two datasets:
      - one with only players with complete data (no missing values)
      - one with all players with missing values imputed using the specified strategy.

    Args:
        df: DataFrame including 'player' column
        dataset_name: Name for display purposes
        strategy: Imputation strategy ('median' recommended)

    Returns:
        complete_df: DataFrame with only complete rows (no NaNs)
        imputed_df: DataFrame with missing values imputed
    """
    print(f"\n=== {dataset_name} Players Imputation Analysis ===")
    print(f"Original dataset: {len(df)} players, {len(df.columns)} features")

    total_missing = df.isna().sum().sum()
    print(f"Total missing values in dataset: {total_missing}")

    missing_per_col = df.isna().sum()
    print("Missing values per feature (top 10):")
    print(missing_per_col[missing_per_col > 0].sort_values(ascending=False).head(10))

    if 'player' not in df.columns:
        raise ValueError("DataFrame must contain a 'player' column")

    # Dataset with complete data (no missing values)
    complete_df = df.dropna()
    print(f"Players with complete data: {len(complete_df)}")

    # Separate player column from features for imputation
    player_col = df['player']
    feature_df = df.drop(columns=['player'])

    # Impute missing values
    imputer = SimpleImputer(strategy=strategy)
    imputed_array = imputer.fit_transform(feature_df)

    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=feature_df.columns, index=feature_df.index)
    imputed_df['player'] = player_col.values

    print(f"After imputation: dataset has {len(imputed_df)} players, {len(imputed_df.columns)} features")
    print("Sample players (imputed dataset):")
    print(imputed_df['player'].head(10).tolist())

    return complete_df, imputed_df


def encode_player_names(df, player_column='player'):
    """
    Encode player names to numeric IDs using a new LabelEncoder.

    Args:
        df: DataFrame with player names
        player_column: Name of the column containing player names

    Returns:
        DataFrame with player names replaced by numeric IDs, and the fitted LabelEncoder
    """
    print(f"\n=== Encoding Player Names ===")
    print(f"Original dataset: {len(df)} players")

    # Create a copy to avoid modifying the original
    df_encoded = df.copy()

    # Get unique players in this dataset
    unique_players = df_encoded[player_column].unique()
    print(f"Unique players in dataset: {len(unique_players)}")

    # Create and fit a new LabelEncoder
    new_le = LabelEncoder()
    new_le.fit(unique_players)

    # Encode player names
    df_encoded[player_column] = new_le.transform(df_encoded[player_column])

    # Create reverse mapping (id -> name)
    id_to_name = dict(zip(new_le.transform(new_le.classes_), new_le.classes_))

    print(f"Encoded dataset: {len(df_encoded)} players")
    print(f"LabelEncoder fitted on {len(new_le.classes_)} unique players")

    return df_encoded, new_le, id_to_name


def normalize_tennis_data_minmax(df: pd.DataFrame, exclude_cols=None):
    """
    Normalize all numerical features with MinMaxScaler to [0,1] range,
    excluding specified columns (e.g., IDs).

    Args:
        df: Input DataFrame with features and possibly ID columns.
        exclude_cols: List of columns to exclude from normalization (e.g. ['player']).

    Returns:
        DataFrame with normalized features (same columns, same order).
    """
    if exclude_cols is None:
        exclude_cols = []

    df_norm = df.copy()

    # Select columns to scale (numeric and not excluded)
    cols_to_scale = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_norm[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df_norm


def split_data(X, test_size=0.2, random_state=42):
    """
    Splits data into training and test sets.

    Args:
        X (np.ndarray): Input data matrix.
        test_size (float): Fraction of data for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test: Training and test data.
    """
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    return X_train, X_test
