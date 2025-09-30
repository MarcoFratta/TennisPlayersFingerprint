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


def cluster_with_metrics(latent_df, k_range=range(2, 50), random_state=42):
    """
    Cluster latent space features with K-means and evaluate using multiple metrics.

    Args:
        latent_df: DataFrame with latent space features (numeric columns only)
        k_range: range of k values to test
        random_state: random seed for reproducibility

    Returns:
        metrics_df: DataFrame with scores for each k
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    X = latent_df.values  # ensure it's numeric
    silhouette_scores = []
    dbi_scores = []
    ch_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)

        silhouette_scores.append(silhouette_score(X, labels))
        dbi_scores.append(davies_bouldin_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))

    # Store metrics
    metrics_df = pd.DataFrame({
        'k': list(k_range),
        'Silhouette': silhouette_scores,
        'Davies-Bouldin': dbi_scores,
        'Calinski-Harabasz': ch_scores
    })

    # Plot metrics
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

    plt.tight_layout()
    plt.show()

    return metrics_df


def visualize_clustering(
    latent_df,
    k_list,
    player_names_map=None,
    player_col='player',
    reducer='umap',  # 'pca', 'tsne', or 'umap'
    random_state=42,
    dataset_name="Dataset"
):
    """
    For each k in k_list, runs KMeans on latent_df and visualizes the clusters
    in interactive 3D plot after dimensionality reduction using PCA, t-SNE, or UMAP.

    Args:
        latent_df (pd.DataFrame): Latent features with optional player column.
        k_list (list[int]): List of cluster counts to try.
        player_names_map (dict): Dictionary mapping player IDs back to names (optional).
        player_col (str): Column containing player IDs/names in latent_df.
        reducer (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        random_state (int): Random seed for reproducibility.
        dataset_name (str): Name of the dataset for plot titles.

    Returns:
        None
    """
    print(f"\n=== Visualizing KMeans clusters for {dataset_name} using {reducer.upper()} (3D Interactive) ===")

    # Separate features and players
    if player_col in latent_df.columns:
        players = latent_df[player_col]
        features = latent_df.drop(columns=[player_col])
    else:
        players = None
        features = latent_df

    # Dimensionality reduction to 3D
    if reducer == 'pca':
        reducer_model = PCA(n_components=3, random_state=random_state)
        reduced_features = reducer_model.fit_transform(features)
    elif reducer == 'tsne':
        reducer_model = TSNE(n_components=3, random_state=random_state, perplexity=30, n_iter=300)
        reduced_features = reducer_model.fit_transform(features)
    elif reducer == 'umap':
        reducer_model = umap.UMAP(n_components=3, random_state=random_state, n_neighbors=15, min_dist=0.1)
        reduced_features = reducer_model.fit_transform(features)
    else:
        raise ValueError("Reducer must be 'pca', 'tsne', or 'umap'.")

    # Prepare DataFrame for plotly
    plot_df = pd.DataFrame({
        'Dim1': reduced_features[:, 0],
        'Dim2': reduced_features[:, 1],
        'Dim3': reduced_features[:, 2],
    })

    if players is not None:
        plot_df[player_col] = players
        if player_names_map is not None:
            plot_df['player_name'] = plot_df[player_col].map(player_names_map)
        else:
            plot_df['player_name'] = plot_df[player_col].astype(str)
    else:
        plot_df['player_name'] = "Unknown"

    for k in k_list:
        print(f"\n--- KMeans clustering with k={k} ---")
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        print(f'clustering on {features.shape} shape df' )
        cluster_labels = kmeans.fit_predict(features)
        plot_df['cluster'] = cluster_labels.astype(str)  # convert to str for categorical coloring

        fig = px.scatter_3d(
            plot_df,
            x='Dim1', y='Dim2', z='Dim3',
            color='cluster',
            hover_name='player_name',
            title=f"KMeans Clusters (k={k}) using {reducer.upper()} for {dataset_name}",
            color_discrete_sequence=px.colors.qualitative.Safe,
            opacity=0.8,
            width=1200,
            height=600
        )

        fig.update_layout(
            legend_title_text='Cluster',
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

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


def cluster_with_models(latent_df, mapper, k, random_state=42):
    """
    Cluster data using both GMM and KMeans and visualize the results.

    Args:
        latent_df: DataFrame with latent features
        mapper: Dictionary mapping player IDs to names
        k: Number of clusters
        random_state: Random seed for reproducibility
    """
    # Example with GMM
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(latent_df.drop(columns='player'))
    fig_gmm = visualize_cluster_regions(gmm, latent_df, mapper, plot_3d=True)
    fig_gmm.show()

    # Example with KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(latent_df.drop(columns='player'))
    fig_kmeans = visualize_cluster_regions(kmeans, latent_df, mapper, plot_3d=True)
    fig_kmeans.show()


def plot_player_style_radar(player, df, model, id_to_name=None, cluster_to_style=None, styles_order=None, title=None):
    """
    Create a hexagonal radar chart showing a player's style profile.

    Args:
        player: Player name or ID
        df: DataFrame containing player data
        model: Fitted clustering model
        id_to_name: Dictionary mapping player IDs to names
        cluster_to_style: Dictionary mapping cluster IDs to style names
        styles_order: List defining the order of styles on the radar
        title: Custom title for the plot
    """
    STYLE_LABELS = ["Big Server","Serve and Volley","All Court Player","Attacking Baseliner","Solid Baseliner","Counter Puncher"]
    
    Xf = df.drop(columns=['player'], errors='ignore')
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
    elif hasattr(model, "cluster_centers_"):
        p = _cluster_scores_from_centroids(xs, model.cluster_centers_)
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
    values = [style_scores[s]*100.0 for s in styles_order]

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

    # Add concentric hexagons for scale
    scale_values = [0, 10, 20, 30, 40]
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


def _cluster_scores_from_centroids(x_scaled, centroids_scaled, eps=1e-8):
    """
    Helper function to calculate cluster scores from centroids.

    Args:
        x_scaled: Scaled feature vector
        centroids_scaled: Scaled centroids
        eps: Small value to avoid division by zero

    Returns:
        numpy array of cluster scores
    """
    dists = np.linalg.norm(centroids_scaled - x_scaled[None, :], axis=1)
    w = 1.0 / (dists + eps)
    return (w / w.sum())


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


def fit_kmeans_on_means(df, n_clusters=6, random_state=42):
    """
    Fit KMeans on the means dataset and return model and centroids.

    Args:
        df: DataFrame with player data
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (kmeans_model, None, labels, centroids)
    """
    Xf = df.drop(columns=['player'], errors='ignore')
    X = Xf.values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=Xf.columns, index=[f"c{i}" for i in range(n_clusters)])
    return kmeans, None, labels, centroids
    
def create_dynamic_cluster_to_style_mapping(df, model, id_to_name, scaler=None):
    """
    Create cluster-to-style mapping based on reference players.
    Finds which cluster contains each reference player and assigns the appropriate style.
    """
    # Define reference players for each tennis style
    reference_players = {
        "Serve and Volley": "John Mcenroe",
        "Big Server": "John Isner", 
        "All Court Player": "Lleyton Hewitt",
        "Attacking Baseliner": "Carlos Alcaraz",
        "Solid Baseliner": "Jannik Sinner",
        "Counter Puncher": "Alexander Zverev"
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
