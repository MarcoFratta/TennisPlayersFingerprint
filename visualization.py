# -*- coding: utf-8 -*-
"""
Visualization Module for Tennis Match Charting Project

This module contains all functions related to data visualization,
plotting, and interactive charts for tennis player analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model
from IPython.display import Image, display
import os
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm


def show_side_by_side(plot_functions, titles=None, figsize=(5, 4)):
    """
    Display multiple plots side by side in a single figure.

    Args:
        plot_functions: List of functions that take an axes object and plot on it
        titles: List of titles for each subplot
        figsize: Tuple of (width, height) for the figure
    """
    n = len(plot_functions)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0]*n, figsize[1]), squeeze=False)

    for i, plot_fn in enumerate(plot_functions):
        plot_fn(axes[0, i])
        if titles:
            axes[0, i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


def plot_tournament_pie(df, title, threshold=50):
    """
    Plot a pie chart showing tournament distribution with small categories grouped as "Other".

    Args:
        df: DataFrame containing tournament data
        title: Title for the plot
        threshold: Minimum number of matches for a tournament to be shown separately
    """
    # Count matches per tournament
    counts = df['Tournament'].value_counts()

    # Group small categories into "Other"
    main = counts[counts >= threshold]
    other_sum = counts[counts < threshold].sum()
    if other_sum > 0:
        main['Other'] = other_sum

    plt.figure(figsize=(6, 6))
    plt.pie(main.values, labels=main.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.show()


def plot_pca_3d(df_scaled, player_names_map, dataset_name="Dataset"):
    """
    Performs PCA to reduce dimensionality to 3 and plots the results in a 3D scatter plot.

    Args:
        df_scaled: DataFrame with scaled features and 'player' ID column.
        player_names_map: Dictionary mapping player IDs back to names.
        dataset_name: Name of the dataset for plot title.
    """
    print(f"\n=== Performing PCA and Plotting for {dataset_name} ===")

    # Separate features and player IDs
    player_ids = df_scaled['player']
    features_scaled = df_scaled.drop(columns=['player'])

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_scaled)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

    # Add player names back using the mapping
    pca_df['player_id'] = player_ids
    pca_df['player_name'] = pca_df['player_id'].map(player_names_map)

    # Create the 3D scatter plot using Plotly
    fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3',
                        hover_name='player_name', # Show player name on hover
                        title=f'{dataset_name} Player Style Clustering (PCA 3D)',
                        labels={'PC1': 'Principal Component 1',
                                'PC2': 'Principal Component 2',
                                'PC3': 'Principal Component 3'},
                        width=1200,
                        height=600)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        hovermode='closest'
    )

    fig.show()

    print(f"Explained variance ratio by each component: {pca.explained_variance_ratio_}")
    print(f"Total explained variance ratio: {pca.explained_variance_ratio_.sum()}")


def plot_training_history(history):
    """
    Plots training and validation loss curves.

    Args:
        history: Keras History object returned by model.fit()
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final validation loss
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final Validation Loss: {final_val_loss:.4f}")


def plot_latent_space(latent_df, player_names_map, dataset_name="Dataset"):
    """
    Plots the latent space using plot_pca_3d if needed.

    Args:
        latent_df: dataset to plot.
        player_names_map: Dictionary mapping player IDs back to names.
        dataset_name: Name of the dataset for plot title.
    """
    print(f"\n=== Encoding and Plotting Latent Space for {dataset_name} ===")

    if latent_df.shape[1] - 1 == 3: # Check if latent space is 3D
      # Rename latent columns for consistent plotting function input
      latent_df_renamed = latent_df.rename(columns={
          'latent_1': 'PC1',
          'latent_2': 'PC2',
          'latent_3': 'PC3'
          })
      # Now plot using Plotly directly from the latent space DataFrame
      fig = px.scatter_3d(latent_df_renamed, x='PC1', y='PC2', z='PC3',
                        hover_name=latent_df_renamed['player'].map(player_names_map), # Map player IDs back to names for hover
                        title=f'{dataset_name} Player Style Clustering (Latent Space 3D)',
                        labels={'PC1': 'Latent Dimension 1',
                                'PC2': 'Latent Dimension 2',
                                'PC3': 'Latent Dimension 3'})

      fig.update_layout(
          margin=dict(l=0, r=0, b=0, t=40),
          hovermode='closest'
      )

      fig.show()

    else:
      print(f"Latent space dimension is {latent_df.shape[1]}. 3D plotting requires 3 dimensions.")
      plot_pca_3d(latent_df, player_names_map, dataset_name)


def plot_clustering_metrics(metrics_df, title="Clustering Metrics"):
    """
    Plot clustering metrics from a pre-computed metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with columns 'k', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'
        title: Title for the plot
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].plot(metrics_df['k'], metrics_df['Silhouette'], marker='o')
    axs[0].set_title('Silhouette (Higher Better)')
    axs[0].set_xlabel('Number of Clusters (k)')
    axs[0].set_ylabel('Score')

    axs[1].plot(metrics_df['k'], metrics_df['Davies-Bouldin'], marker='o', color='orange')
    axs[1].set_title('Davies-Bouldin (Lower Better)')
    axs[1].set_xlabel('Number of Clusters (k)')
    axs[1].set_ylabel('Score')

    axs[2].plot(metrics_df['k'], metrics_df['Calinski-Harabasz'], marker='o', color='green')
    axs[2].set_title('Calinski-Harabasz (Higher Better)')
    axs[2].set_xlabel('Number of Clusters (k)')
    axs[2].set_ylabel('Score')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_clustering_with_models(
    latent_df,
    fitted_models_dict,
    player_names_map=None,
    player_col='player',
    reducer='umap',  # 'pca', 'tsne', or 'umap'
    random_state=42,
    dataset_name="Dataset",
    show_2d=True,
    show_3d=True
):
    """
    Visualize clustering using pre-fitted models (no fitting allowed).
    Shows both 2D and 3D plots by default.
    
    Args:
        latent_df (pd.DataFrame): Latent features with optional player column.
        fitted_models_dict (dict): Dictionary with k as keys and fitted models as values.
        player_names_map (dict): Dictionary mapping player IDs back to names (optional).
        player_col (str): Column containing player IDs/names in latent_df.
        reducer (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        random_state (int): Random seed for reproducibility.
        dataset_name (str): Name of the dataset for plot titles.
        show_2d (bool): Whether to show 2D plots.
        show_3d (bool): Whether to show 3D plots.

    Returns:
        None
    """
    print(f"\n=== Visualizing clusters for {dataset_name} using {reducer.upper()} ===")

    # Separate features and players
    if player_col in latent_df.columns:
        players = latent_df[player_col]
        features = latent_df.drop(columns=[player_col])
    else:
        players = None
        features = latent_df

    # Dimensionality reduction to 3D (for both 2D and 3D plots)
    if reducer == 'pca':
        reducer_model_3d = PCA(n_components=3, random_state=random_state)
        reduced_features_3d = reducer_model_3d.fit_transform(features)
        if show_2d:
            reducer_model_2d = PCA(n_components=2, random_state=random_state)
            reduced_features_2d = reducer_model_2d.fit_transform(features)
    elif reducer == 'tsne':
        reducer_model_3d = TSNE(n_components=3, random_state=random_state, perplexity=30, n_iter=300)
        reduced_features_3d = reducer_model_3d.fit_transform(features)
        if show_2d:
            reducer_model_2d = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=300)
            reduced_features_2d = reducer_model_2d.fit_transform(features)
    elif reducer == 'umap':
        reducer_model_3d = umap.UMAP(n_components=3, random_state=random_state, n_neighbors=15, min_dist=0.1)
        reduced_features_3d = reducer_model_3d.fit_transform(features)
        if show_2d:
            reducer_model_2d = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            reduced_features_2d = reducer_model_2d.fit_transform(features)
    else:
        raise ValueError("Reducer must be 'pca', 'tsne', or 'umap'.")

    # Prepare DataFrames for plotly
    plot_df_3d = pd.DataFrame({
        'Dim1': reduced_features_3d[:, 0],
        'Dim2': reduced_features_3d[:, 1],
        'Dim3': reduced_features_3d[:, 2],
    })
    
    if show_2d:
        plot_df_2d = pd.DataFrame({
            'Dim1': reduced_features_2d[:, 0],
            'Dim2': reduced_features_2d[:, 1],
        })

    if players is not None:
        plot_df_3d[player_col] = players
        if show_2d:
            plot_df_2d[player_col] = players
        if player_names_map is not None:
            plot_df_3d['player_name'] = plot_df_3d[player_col].map(player_names_map)
            if show_2d:
                plot_df_2d['player_name'] = plot_df_2d[player_col].map(player_names_map)
        else:
            plot_df_3d['player_name'] = plot_df_3d[player_col].astype(str)
            if show_2d:
                plot_df_2d['player_name'] = plot_df_2d[player_col].astype(str)
    else:
        plot_df_3d['player_name'] = "Unknown"
        if show_2d:
            plot_df_2d['player_name'] = "Unknown"

    for k, model in fitted_models_dict.items():
        print(f"\n--- Visualizing clusters with k={k} ---")
        
        # Get cluster labels from pre-fitted model
        if hasattr(model, 'labels_'):
            cluster_labels = model.labels_
        elif hasattr(model, 'predict'):
            cluster_labels = model.predict(features)
        else:
            raise ValueError(f"Model for k={k} does not have 'labels_' or 'predict' method")
        
        plot_df_3d['cluster'] = cluster_labels.astype(str)
        if show_2d:
            plot_df_2d['cluster'] = cluster_labels.astype(str)

        # Create 2D plot
        if show_2d:
            fig_2d = px.scatter(
                plot_df_2d,
                x='Dim1', y='Dim2',
                color='cluster',
                hover_name='player_name',
                title=f"2D Clusters (k={k}) using {reducer.upper()} for {dataset_name}",
                color_discrete_sequence=px.colors.qualitative.Safe,
                opacity=0.8,
                width=800,
                height=600
            )

            fig_2d.update_layout(
                legend_title_text='Cluster',
                margin=dict(l=0, r=0, b=0, t=40)
            )

            fig_2d.show()

        # Create 3D plot
        if show_3d:
            fig_3d = px.scatter_3d(
                plot_df_3d,
                x='Dim1', y='Dim2', z='Dim3',
                color='cluster',
                hover_name='player_name',
                title=f"3D Clusters (k={k}) using {reducer.upper()} for {dataset_name}",
                color_discrete_sequence=px.colors.qualitative.Safe,
                opacity=0.8,
                width=1200,
                height=600
            )

            fig_3d.update_layout(
                legend_title_text='Cluster',
                margin=dict(l=0, r=0, b=0, t=40)
            )

            fig_3d.show()

        # Print cluster stats
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("Cluster counts:")
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count}")

        if players is not None and player_names_map is not None:
            print("Players per cluster:")
            for c in unique_labels:
                sample_players_ids = players[cluster_labels == c].tolist()
                sample_players_names = [player_names_map.get(pid, f"ID_{pid}") for pid in sample_players_ids]
                print(f"Cluster {c}: {sample_players_names}")


def visualize_clustering_2d_only(
    latent_df,
    fitted_models_dict,
    player_names_map=None,
    player_col='player',
    reducer='umap',
    random_state=42,
    dataset_name="Dataset"
):
    """
    Visualize clustering using pre-fitted models in 2D only.
    
    Args:
        latent_df (pd.DataFrame): Latent features with optional player column.
        fitted_models_dict (dict): Dictionary with k as keys and fitted models as values.
        player_names_map (dict): Dictionary mapping player IDs back to names (optional).
        player_col (str): Column containing player IDs/names in latent_df.
        reducer (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        random_state (int): Random seed for reproducibility.
        dataset_name (str): Name of the dataset for plot titles.

    Returns:
        None
    """
    visualize_clustering_with_models(
        latent_df=latent_df,
        fitted_models_dict=fitted_models_dict,
        player_names_map=player_names_map,
        player_col=player_col,
        reducer=reducer,
        random_state=random_state,
        dataset_name=dataset_name,
        show_2d=True,
        show_3d=False
    )


def visualize_clustering_3d_only(
    latent_df,
    fitted_models_dict,
    player_names_map=None,
    player_col='player',
    reducer='umap',
    random_state=42,
    dataset_name="Dataset"
):
    """
    Visualize clustering using pre-fitted models in 3D only.
    
    Args:
        latent_df (pd.DataFrame): Latent features with optional player column.
        fitted_models_dict (dict): Dictionary with k as keys and fitted models as values.
        player_names_map (dict): Dictionary mapping player IDs back to names (optional).
        player_col (str): Column containing player IDs/names in latent_df.
        reducer (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        random_state (int): Random seed for reproducibility.
        dataset_name (str): Name of the dataset for plot titles.

    Returns:
        None
    """
    visualize_clustering_with_models(
        latent_df=latent_df,
        fitted_models_dict=fitted_models_dict,
        player_names_map=player_names_map,
        player_col=player_col,
        reducer=reducer,
        random_state=random_state,
        dataset_name=dataset_name,
        show_2d=False,
        show_3d=True
    )


def visualize_cluster_regions(model, features_df, id_to_player, plot_3d=True, random_state=42):
    """
    Visualizes cluster regions for any fitted clustering model using convex hulls.

    Args:
        model: Fitted clustering model (GMM, KMeans, etc.)
        features_df (pd.DataFrame): DataFrame with features
        id_to_player (dict): Mapping from player ID to player name
        plot_3d (bool): Whether to plot in 3D (True) or 2D (False)
        random_state (int): Random seed for UMAP

    Returns:
        plotly.graph_objects.Figure: Interactive plot showing cluster regions
    """
    if 'player' in features_df.columns:
        player_ids = features_df['player'].values
        X_features = features_df.drop(columns=['player'])
        X = X_features.values
    else:
        player_ids = features_df.index.values
        X_features = features_df
        X = features_df.values

    if hasattr(model, 'predict'):
        clusters = model.predict(X_features)
    elif hasattr(model, 'fit_predict'):
        clusters = model.fit_predict(X_features)
    elif hasattr(model, 'labels_'):
        clusters = model.labels_
    else:
        raise ValueError("Model must have 'predict', 'fit_predict', or 'labels_' attribute")

    n_components = len(np.unique(clusters))

    if plot_3d:
        reducer = umap.UMAP(n_components=3, random_state=random_state)
    else:
        reducer = umap.UMAP(n_components=2, random_state=random_state)

    embedding = reducer.fit_transform(X)

    if plot_3d:
        cols = ['x', 'y', 'z']
    else:
        cols = ['x', 'y']

    plot_df = pd.DataFrame(embedding, columns=cols)
    plot_df['Cluster'] = clusters
    plot_df['Player'] = [id_to_player.get(pid, "Unknown") for pid in player_ids]

    fig = go.Figure()
    colors = px.colors.qualitative.Set3[:n_components]

    for cluster_id in range(n_components):
        cluster_mask = plot_df['Cluster'] == cluster_id
        cluster_data = plot_df[cluster_mask]

        if len(cluster_data) > 0:
            if plot_3d and len(cluster_data) >= 4:
                try:
                    hull = ConvexHull(cluster_data[['x', 'y', 'z']].values)
                    vertices = cluster_data[['x', 'y', 'z']].values[hull.vertices]

                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        color=colors[cluster_id],
                        opacity=0.3,
                        name=f'Cluster {cluster_id} Boundary',
                        showlegend=True
                    ))
                except:
                    pass
            elif not plot_3d and len(cluster_data) >= 3:
                try:
                    hull = ConvexHull(cluster_data[['x', 'y']].values)
                    vertices = cluster_data[['x', 'y']].values[hull.vertices]

                    fig.add_trace(go.Scatter(
                        x=np.append(vertices[:, 0], vertices[0, 0]),
                        y=np.append(vertices[:, 1], vertices[0, 1]),
                        mode='lines',
                        line=dict(color=colors[cluster_id], width=2),
                        fill='toself',
                        fillcolor=colors[cluster_id],
                        opacity=0.3,
                        name=f'Cluster {cluster_id} Boundary',
                        showlegend=True
                    ))
                except:
                    pass

            if plot_3d:
                fig.add_trace(go.Scatter3d(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    z=cluster_data['z'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors[cluster_id],
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name=f'Cluster {cluster_id} Points',
                    showlegend=False,
                    customdata=cluster_data[['Player', 'Cluster']].values,
                    hovertemplate='<b>%{customdata[0]}</b><br>' +
                                'Cluster: %{customdata[1]}<br>' +
                                'X: %{x:.2f}<br>' +
                                'Y: %{y:.2f}<br>' +
                                'Z: %{z:.2f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[cluster_id],
                        opacity=0.8,
                        symbol='circle'
                    ),
                    name=f'Cluster {cluster_id} Points',
                    showlegend=False,
                    customdata=cluster_data[['Player', 'Cluster']].values,
                    hovertemplate='<b>%{customdata[0]}</b><br>' +
                                'Cluster: %{customdata[1]}<br>' +
                                'X: %{x:.2f}<br>' +
                                'Y: %{y:.2f}<extra></extra>'
                ))

    if plot_3d:
        fig.update_layout(
            title=f'Cluster Regions Visualization (k={n_components})',
            scene=dict(
                xaxis_title='UMAP-1',
                yaxis_title='UMAP-2',
                zaxis_title='UMAP-3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
    else:
        fig.update_layout(
            title=f'Cluster Regions Visualization (k={n_components})',
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            width=1000,
            height=800
        )

    return fig


def compare_clustering_models(latent_df, fitted_models_dict, mapper, random_state=42):
    """
    Compare different clustering models using pre-fitted models (no fitting allowed).

    Args:
        latent_df: DataFrame with latent features
        fitted_models_dict: Dictionary with model names as keys and fitted models as values
        mapper: Dictionary mapping player IDs to names
        random_state: Random seed for reproducibility
    """
    for model_name, model in fitted_models_dict.items():
        print(f"\n--- Visualizing {model_name} model ---")
        fig = visualize_cluster_regions(model, latent_df, mapper, plot_3d=True, random_state=random_state)
        fig.show()



def _cluster_scores_from_centroids(x_scaled, centroids_scaled, eps=1e-8):
    """
    Helper function to calculate cluster scores from centroids using improved distance normalization.
    This approach handles small clusters better and provides more balanced radar plot visualization.

    Args:
        x_scaled: Scaled feature vector
        centroids_scaled: Scaled centroids
        eps: Small value to avoid division by zero

    Returns:
        numpy array of cluster scores (10-90 range)
    """
    dists = np.linalg.norm(centroids_scaled - x_scaled[None, :], axis=1)
    
    # Use exponential decay for better score distribution
    # This prevents extreme compression when one cluster is very far away
    max_dist = np.max(dists)
    
    if max_dist > eps:
        # Use exponential decay: scores = base_score * exp(-decay_rate * normalized_distance)
        # This ensures all clusters get reasonable scores even if one is very far
        normalized_dists = dists / max_dist
        decay_rate = 2.0  # Controls how quickly scores decrease with distance
        
        # Calculate scores using exponential decay
        scores = 80 * np.exp(-decay_rate * normalized_dists) + 10
        
        # Ensure scores are in the 10-90 range
        scores = np.clip(scores, 10, 90)
    else:
        # If all distances are very small, give equal scores
        scores = np.full(len(dists), 50.0)
    
    return scores


def _cluster_scores_from_centroids_robust(x_scaled, centroids_scaled, eps=1e-8, min_score=15, max_score=85):
    """
    Robust helper function to calculate cluster scores from centroids.
    Uses percentile-based normalization to handle outliers and small clusters better.

    Args:
        x_scaled: Scaled feature vector
        centroids_scaled: Scaled centroids
        eps: Small value to avoid division by zero
        min_score: Minimum score to assign (default 15)
        max_score: Maximum score to assign (default 85)

    Returns:
        numpy array of cluster scores (min_score to max_score range)
    """
    dists = np.linalg.norm(centroids_scaled - x_scaled[None, :], axis=1)
    
    # Use percentile-based normalization to handle outliers
    # This is more robust than min-max normalization
    p25, p75 = np.percentile(dists, [25, 75])
    iqr = p75 - p25
    
    if iqr > eps:
        # Use IQR-based normalization with outlier handling
        # Cap extreme distances at p75 + 1.5*IQR (like in box plots)
        outlier_threshold = p75 + 1.5 * iqr
        capped_dists = np.minimum(dists, outlier_threshold)
        
        # Normalize using the capped distances
        min_dist = np.min(capped_dists)
        max_dist = np.max(capped_dists)
        
        if max_dist > min_dist:
            # Invert distances: closer = higher score
            normalized = (max_dist - capped_dists) / (max_dist - min_dist)
            scores = min_score + normalized * (max_score - min_score)
        else:
            scores = np.full(len(dists), (min_score + max_score) / 2)
    else:
        # If distances are very similar, use softmax-like approach
        # This gives more balanced scores when clusters are close
        softmax_scores = np.exp(-dists / (np.std(dists) + eps))
        softmax_scores = softmax_scores / np.sum(softmax_scores)
        scores = min_score + softmax_scores * (max_score - min_score)
    
    return scores


def _cluster_scores_from_centroids_weighted(x_scaled, centroids_scaled, feature_std, eps=1e-8, temperature=1.0):
    """
    Calculate cluster scores using improved distance normalization with minimum threshold.
    This prevents extreme values while maintaining relative relationships.

    Args:
        x_scaled: Scaled feature vector
        centroids_scaled: Scaled centroids
        feature_std: Standard deviations for feature weighting
        eps: Small value to avoid division by zero
        temperature: Temperature parameter for scaling

    Returns:
        numpy array of cluster scores (10-90 range)
    """
    dists = np.linalg.norm(centroids_scaled - x_scaled[None, :], axis=1)
    
    # Apply minimum threshold to avoid extreme values
    min_threshold = 10.0  # Minimum score to ensure all clusters are visible
    max_score = 90.0      # Maximum score to leave room for scale rings
    
    min_dist, max_dist = np.min(dists), np.max(dists)
    
    if max_dist > min_dist:
        # Normalize to 10-90 range (closer = higher score)
        scores = max_score - ((dists - min_dist) / (max_dist - min_dist)) * (max_score - min_threshold)
        scores = np.maximum(scores, min_threshold)  # Ensure minimum threshold
    else:
        # If all distances are the same, give equal scores
        scores = np.full(len(dists), (min_threshold + max_score) / 2)
    
    return scores


def plot_player_style_radar(player, df, model, scale_values=[0, 20, 40, 60, 80, 100], id_to_name=None, cluster_to_style=None, styles_order=None, title=None, feature_std=None, temperature=0.5, scoring_method='robust'):
    """
    Create a hexagonal radar chart showing a player's style profile.

    Args:
        player: Player name or ID
        df: DataFrame containing player data
        model: Fitted clustering model
        scale_values: List of scale values for the radar chart grid
        id_to_name: Dictionary mapping player IDs to names
        cluster_to_style: Dictionary mapping cluster IDs to style names
        styles_order: List defining the order of styles on the radar
        title: Custom title for the plot
        feature_std: Standard deviations for feature weighting
        temperature: Temperature parameter for soft clustering
        scoring_method: Method for calculating cluster scores ('robust', 'exponential', 'weighted', 'simple')
    """
    STYLE_LABELS = ["Big Server","Serve and Volley","All Court Player","Attacking Baseliner","Solid Baseliner","Counter Puncher"]
    
    Xf = df.drop(columns=['player'], errors='ignore')
    if feature_std is None:
        feature_std = Xf.values.std(axis=0)
    if isinstance(player, str):
        if id_to_name is None:
            raise ValueError("id_to_name required when passing a player name")
        name_to_id = {v:k for k,v in id_to_name.items()}
        if player not in name_to_id:
            raise ValueError("player name not found")
        player_id = name_to_id[player]
    else:
        player_id = player
    player_label = id_to_name.get(player_id, str(player_id)) if id_to_name is not None else str(player_id)
    x = _player_vector(df, player_id)
    xs = x
    if hasattr(model, "predict_proba"):
        p = model.predict_proba([xs])[0]
        # Convert probabilities to radar plot values (10-90 range)
        p = p * 80 + 10  # Scale from [0,1] to [10,90]
    elif hasattr(model, "cluster_centers_"):
        # Choose scoring method based on parameter
        if scoring_method == 'robust':
            p = _cluster_scores_from_centroids_robust(xs, model.cluster_centers_)
        elif scoring_method == 'exponential':
            p = _cluster_scores_from_centroids(xs, model.cluster_centers_)
        elif scoring_method == 'weighted':
            p = _cluster_scores_from_centroids_weighted(xs, model.cluster_centers_, feature_std, temperature=temperature)
        elif scoring_method == 'simple':
            # Original simple method (for backward compatibility)
            dists = np.linalg.norm(model.cluster_centers_ - xs[None, :], axis=1)
            min_dist, max_dist = np.min(dists), np.max(dists)
            if max_dist > min_dist:
                p = 90 - ((dists - min_dist) / (max_dist - min_dist)) * 80
                p = np.maximum(p, 10)
            else:
                p = np.full(len(dists), 50.0)
        else:
            raise ValueError(f"Unknown scoring_method: {scoring_method}. Choose from 'robust', 'exponential', 'weighted', 'simple'")
    else:
        raise ValueError("Unsupported model")
    k = len(p)
    if cluster_to_style is None:
        cluster_to_style = {i: STYLE_LABELS[i] if i < len(STYLE_LABELS) else f"Cluster {i}" for i in range(k)}
    elif isinstance(cluster_to_style, list):
        if len(cluster_to_style) != k:
            raise ValueError("cluster_to_style list length must equal number of clusters")
        cluster_to_style = {i: cluster_to_style[i] for i in range(k)}
    elif isinstance(cluster_to_style, dict):
        missing = [i for i in range(k) if i not in cluster_to_style]
        if missing:
            raise ValueError(f"cluster_to_style dict missing keys: {missing}")
    else:
        raise ValueError("cluster_to_style must be None, list, or dict")
    if styles_order is None:
        styles_order = list(dict.fromkeys(cluster_to_style.values()))
    style_scores = {s:0.0 for s in styles_order}
    for i, score in enumerate(p):
        style_scores[cluster_to_style[i]] = style_scores.get(cluster_to_style[i], 0.0) + float(score)
    values = [style_scores[s] for s in styles_order]  # Remove *100 since scores are already in 10-90 range

    # Calculate hexagonal coordinates
    n_vertices = len(styles_order)
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False) + np.pi/2

    # Convert to cartesian coordinates for hexagon
    x_coords = []
    y_coords = []
    for angle, value in zip(angles, values):
        x_coords.append(value * np.cos(angle))
        y_coords.append(value * np.sin(angle))

    # Close the hexagon
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    fig = go.Figure()

    for scale in scale_values:
        x_scale = []
        y_scale = []
        for angle in angles:
            x_scale.append(scale * np.cos(angle))
            y_scale.append(scale * np.sin(angle))
        x_scale.append(x_scale[0])
        y_scale.append(y_scale[0])

        fig.add_trace(go.Scatter(
            x=x_scale,
            y=y_scale,
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add the main hexagon
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='lines+markers',
        fill='toself',
        line=dict(color='#3b6cff', width=2),
        fillcolor='rgba(59, 108, 255, 0.25)',
        marker=dict(size=4, color='#3b6cff'),
        name=player_label,
        showlegend=False
    ))

    # Add labels at hexagon vertices
    for i, (angle, label) in enumerate(zip(angles, styles_order)):
        label_x = 50 * np.cos(angle)
        label_y = 50 * np.sin(angle)
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=label,
            showarrow=False,
            font=dict(size=11),
            xanchor='center',
            yanchor='middle'
        )

    # Add scale labels
    for scale in scale_values:
        fig.add_annotation(
            x=scale + 8,
            y=0,
            text=str(scale),
            showarrow=False,
            font=dict(size=8),
            xanchor='left',
            yanchor='middle'
        )

    fig.update_layout(
        title=dict(
            text=title or f"{player_label} Profile",
            font=dict(size=16),
            x=0.5
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        plot_bgcolor='white',
        width=800,
        height=600
    )

    fig.show()



def _player_vector(df, player_id):
    """
    Helper function to extract player vector from DataFrame.

    Args:
        df: DataFrame containing player data
        player_id: Player ID to extract

    Returns:
        numpy array of player features
    """
    if 'player' in df.columns:
        row = df.loc[df['player'] == player_id]
        if row.empty:
            raise ValueError("player_id not found")
        return row.drop(columns=['player']).iloc[0].values
    else:
        if player_id not in df.index:
            raise ValueError("player_id not found")
        return df.loc[player_id].values

def get_cluster_members(df, model, id_to_name, scaler=None):
    """
    Get cluster members for each cluster.

    Args:
        df: DataFrame with player data
        model: Fitted clustering model
        id_to_name: Dictionary mapping player IDs to names
        scaler: Optional scaler for features

    Returns:
        Dictionary mapping cluster IDs to lists of player names
    """
    Xf = df.drop(columns=['player'], errors='ignore')
    Xs = Xf.values if scaler is None else scaler.transform(Xf)
    labels = model.predict(Xs)
    pids = df['player'].values if 'player' in df.columns else df.index.values
    names = [id_to_name.get(pid, str(pid)) for pid in pids]
    out = {}
    for c in range(getattr(model, 'n_clusters', len(np.unique(labels)))):
        out[c] = [n for n, lab in zip(names, labels) if lab == c]
    return out


def print_cluster_members(df, model, id_to_name, scaler=None):
    """
    Print cluster members for each cluster.

    Args:
        df: DataFrame with player data
        model: Fitted clustering model
        id_to_name: Dictionary mapping player IDs to names
        scaler: Optional scaler for features
    """
    members = get_cluster_members(df, model, id_to_name, scaler)
    for c in sorted(members):
        print(f"Cluster {c}: {members[c]}")


def get_cluster_centroids(model, feature_names):
    """
    Get cluster centroids from a fitted model.
    
    Args:
        model: Fitted clustering model
        feature_names: List of feature names
        
    Returns:
        DataFrame with centroids
    """
    if hasattr(model, 'cluster_centers_'):
        centroids = pd.DataFrame(
            model.cluster_centers_, 
            columns=feature_names,
            index=[f"cluster_{i}" for i in range(len(model.cluster_centers_))]
        )
        return centroids
    else:
        raise ValueError("Model does not have cluster_centers_ attribute")


def plot_silhouette_analysis(model, features, labels=None, dataset_name="Dataset", figsize=(15, 6)):
    """
    Create a silhouette analysis plot for a given clustering model.
    
    Args:
        model: Fitted clustering model
        features: Feature matrix used for clustering (DataFrame or array)
        labels: Cluster labels (if None, will be predicted from model)
        dataset_name: Name for the plot title
        figsize: Figure size tuple (width, height)
        
    Returns:
        Dictionary with silhouette analysis results
    """
    
    # Handle DataFrame input - remove 'player' column if present
    if hasattr(features, 'columns'):
        if 'player' in features.columns:
            features_array = features.drop(columns=['player']).values
        else:
            features_array = features.values
    else:
        features_array = features
    
    # Get cluster labels if not provided
    if labels is None:
        if hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            try:
                labels = model.predict(features_array)
            except ValueError as e:
                if "features" in str(e) and "expecting" in str(e):
                    print(f"Warning: Feature dimension mismatch. Model expects {getattr(model, 'n_features_in_', 'unknown')} features, but got {features_array.shape[1]}.")
                    print("Using model.labels_ if available, otherwise cannot proceed.")
                    if hasattr(model, 'labels_'):
                        labels = model.labels_
                    else:
                        raise ValueError(f"Cannot predict labels due to feature mismatch: {e}")
                else:
                    raise e
        else:
            raise ValueError("Model must have 'labels_' or 'predict' method")
    
    # Get number of clusters
    n_clusters = len(np.unique(labels))
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(features_array, labels)
    sample_silhouette_values = silhouette_samples(features_array, labels)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Silhouette plot
    y_lower = 10
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # Plot 2: The clustered data
    # Reduce to 2D for visualization
    if features_array.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features_array)
        x_label = f"1st principal component (explained variance: {pca.explained_variance_ratio_[0]:.2f})"
        y_label = f"2nd principal component (explained variance: {pca.explained_variance_ratio_[1]:.2f})"
    else:
        features_2d = features_array
        x_label = "Feature space for the 1st feature"
        y_label = "Feature space for the 2nd feature"
    
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    ax2.scatter(features_2d[:, 0], features_2d[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    
    # Mark the centers
    if hasattr(model, 'cluster_centers_'):
        if features_array.shape[1] > 2:
            # Check if cluster centers have the same number of features as our current data
            if model.cluster_centers_.shape[1] == features_array.shape[1]:
                centers_2d = pca.transform(model.cluster_centers_)
            else:
                print(f"Warning: Cluster centers have {model.cluster_centers_.shape[1]} features, but current data has {features_array.shape[1]} features. Skipping center visualization.")
                centers_2d = None
        else:
            centers_2d = model.cluster_centers_
        
        if centers_2d is not None:
            ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='o',
                       c="white", alpha=1, s=200, edgecolor='k')
            
            for i, c in enumerate(centers_2d):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                           s=50, edgecolor='k')
    
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    
    plt.suptitle(f"Silhouette analysis for KMeans clustering on {dataset_name} with n_clusters = {n_clusters}",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print silhouette analysis summary
    print(f"\nSilhouette Analysis Summary for {dataset_name}:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Average silhouette score: {silhouette_avg:.3f}")
    
    # Print silhouette scores per cluster
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_avg = np.mean(cluster_silhouette_values)
        cluster_size = len(cluster_silhouette_values)
        print(f"Cluster {i}: {cluster_size} samples, avg silhouette score: {cluster_avg:.3f}")
    
    return {
        'silhouette_avg': silhouette_avg,
        'sample_silhouette_values': sample_silhouette_values,
        'n_clusters': n_clusters,
        'cluster_sizes': [np.sum(labels == i) for i in range(n_clusters)]
    }


def plot_silhouette_comparison(models_dict, features, dataset_name="Dataset", figsize=(20, 12)):
    """
    Create silhouette analysis plots for multiple clustering models for comparison.
    
    Args:
        models_dict: Dictionary with model names as keys and fitted models as values
        features: Feature matrix used for clustering
        dataset_name: Name for the plot title
        figsize: Figure size tuple (width, height)
        
    Returns:
        Dictionary with silhouette analysis results for each model
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    results = {}
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        # Get cluster labels
        if hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            labels = model.predict(features)
        else:
            raise ValueError(f"Model {model_name} must have 'labels_' or 'predict' method")
        
        n_clusters = len(np.unique(labels))
        silhouette_avg = silhouette_score(features, labels)
        sample_silhouette_values = silhouette_samples(features, labels)
        
        # Plot silhouette analysis
        ax1 = axes[0, idx]
        y_lower = 10
        
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_title(f"{model_name}\n(n_clusters={n_clusters}, avg_score={silhouette_avg:.3f})")
        
        # Plot 2D visualization
        ax2 = axes[1, idx]
        
        # Reduce to 2D for visualization
        if features.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(features)
        else:
            features_2d = features
        
        colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
        ax2.scatter(features_2d[:, 0], features_2d[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        
        # Mark the centers
        if hasattr(model, 'cluster_centers_'):
            if features.shape[1] > 2:
                centers_2d = pca.transform(model.cluster_centers_)
            else:
                centers_2d = model.cluster_centers_
            
            ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='o',
                       c="white", alpha=1, s=200, edgecolor='k')
            
            for i, c in enumerate(centers_2d):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                           s=50, edgecolor='k')
        
        ax2.set_xlabel("1st principal component")
        ax2.set_ylabel("2nd principal component")
        ax2.set_title(f"{model_name} - 2D Visualization")
        
        results[model_name] = {
            'silhouette_avg': silhouette_avg,
            'n_clusters': n_clusters,
            'cluster_sizes': [np.sum(labels == i) for i in range(n_clusters)]
        }
    
    plt.suptitle(f"Silhouette Analysis Comparison for {dataset_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print(f"\nSilhouette Analysis Comparison for {dataset_name}:")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:20s}: {result['n_clusters']:2d} clusters, avg score: {result['silhouette_avg']:.3f}")
    
    return results
   
    
def create_dynamic_cluster_to_style_mapping(df, model, id_to_name, scaler=None, reference_players=None):
    """
    Create cluster-to-style mapping based on reference players.
    Finds which cluster contains each reference player and assigns the appropriate style.
    
    Args:
        df: DataFrame with player data
        model: Fitted clustering model
        id_to_name: Dictionary mapping player IDs to names
        scaler: Optional scaler for features
        reference_players: Dictionary mapping style names to reference player names.
            If None, uses default reference players.
    
    Returns:
        Dictionary mapping cluster IDs to style names
    """
    # Default reference players if none provided
    if reference_players is None:
        reference_players = {
            "Serve and Volley": "John Mcenroe",
            "Big Server": "John Isner", 
            "All Court Player": "Stan Wawrinka",
            "Attacking Baseliner": "Carlos Alcaraz",
            "Solid Baseliner": "Novak Djokovic",
            "Counter Puncher": "Lleyton Hewitt"
        }
    
    # Get cluster members
    members = get_cluster_members(df, model, id_to_name, scaler)
    
    # Create reverse mapping: player name -> cluster number
    player_to_cluster = {}
    for cluster_id, player_list in members.items():
        for player in player_list:
            player_to_cluster[player] = cluster_id
    
    # Map each style to its cluster based on reference player
    cluster_to_style = {}
    unmapped_styles = []
    
    for style, ref_player in reference_players.items():
        if ref_player in player_to_cluster:
            cluster_id = player_to_cluster[ref_player]
            cluster_to_style[cluster_id] = style
            print(f"Style '{style}' assigned to Cluster {cluster_id} (reference: {ref_player})")
        else:
            unmapped_styles.append(style)
            print(f"Warning: Reference player '{ref_player}' for style '{style}' not found in clusters")
    
    # Handle any clusters that weren't mapped
    all_clusters = set(members.keys())
    mapped_clusters = set(cluster_to_style.keys())
    unmapped_clusters = all_clusters - mapped_clusters
    
    for cluster_id in unmapped_clusters:
        cluster_to_style[cluster_id] = f"Unknown Style {cluster_id}"
        print(f"Warning: Cluster {cluster_id} not mapped to any style")
    
    if unmapped_styles:
        print(f"Warning: Styles not mapped: {unmapped_styles}")
    
    return cluster_to_style

    
def visualize_model(model, filename="model.png", show_shapes=True, expand_nested=True, dpi=96, inline=True):
  """
  Visualizes a Keras model architecture.

  Args:
      model: Keras model instance to visualize.
      filename (str): Output file name (supports .png, .svg, .pdf).
      show_shapes (bool): Whether to display layer shapes in the diagram.
      expand_nested (bool): Whether to expand nested models.
      dpi (int): Dots per inch (resolution) for the saved diagram.
      inline (bool): If True, displays diagram inline (works in Jupyter).
  """
  # Create the diagram file
  plot_model(
      model,
      to_file=filename,
      show_shapes=show_shapes,
      show_layer_names=True,
      expand_nested=expand_nested,
      dpi=dpi
  )

  # Display inline if requested and running in Jupyter
  if inline and os.path.exists(filename):
      display(Image(filename=filename))
