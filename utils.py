# -*- coding: utf-8 -*-
"""
Utility Functions for Tennis Match Charting Project

This module contains utility functions for model building, training,
and data processing that don't fit into data loading or visualization.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def build_autoencoder(input_dim, latent_dim, denoising=False):
    """
    Build an autoencoder model with the specified architecture.

    Args:
        input_dim: Number of input features
        latent_dim: Dimension of the latent space
        denoising: Whether to add Gaussian noise for denoising autoencoder

    Returns:
        Tuple of (autoencoder_model, encoder_model)
    """
    input_layer = layers.Input(shape=(input_dim,))
    x = input_layer

    # Add Gaussian noise if denoising
    if denoising:
        x = layers.GaussianNoise(0.1)(x)

    # Encoder
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    latent = layers.Dense(latent_dim, activation='linear')(x)

    # Decoder
    x = layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(latent)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    output_layer = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, latent)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder


def train_autoencoder_with_validation(model, X_train, X_test, batch_size=16, epochs=100, patience=10):
    """
    Trains autoencoder on training data and validates on test data.

    Args:
        model: Compiled autoencoder model
        X_train: Training data
        X_test: Test data
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience

    Returns:
        Keras History object
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return history


def encode_dataset_to_latent_space(encoder_model, df_scaled, scaler=None):
    """
    Encode a dataset to latent space using a trained encoder.

    Args:
        encoder_model: Trained encoder model
        df_scaled: Scaled DataFrame with features
        scaler: Optional scaler for the latent space

    Returns:
        Tuple of (latent_df, scaler)
    """
    player_ids = df_scaled['player'].reset_index(drop=True)
    features_scaled = df_scaled.drop(columns=['player'])
    latent_space = encoder_model.predict(features_scaled, verbose=0)
    latent_cols = [f'latent_{i+1}' for i in range(latent_space.shape[1])]
    latent_df = pd.DataFrame(latent_space, columns=latent_cols)
    if scaler is None:
        scaler = StandardScaler()
        latent_df[latent_cols] = scaler.fit_transform(latent_df[latent_cols])
    else:
        latent_df[latent_cols] = scaler.transform(latent_df[latent_cols])
    latent_df['player'] = player_ids
    return latent_df, scaler


def fit_optimized_kmeans(latent_df, n_clusters, init_method='k-means++', n_init=10, max_iter=300, random_state=42):
    """
    Fit KMeans with optimized initialization strategies.
    
    Args:
        latent_df: DataFrame with latent features and 'player' column
        n_clusters: Number of clusters
        init_method: Initialization method ('k-means++', 'k-means||', or 'random')
        n_init: Number of times the algorithm is run with different centroid seeds
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'model': Fitted KMeans model
        - 'labels': Cluster labels
        - 'centroids': Cluster centroids as DataFrame
        - 'inertia': Sum of squared distances of samples to their closest cluster center
        - 'n_iter': Number of iterations run
        - 'features': Features used for clustering (without player column)
        - 'player_ids': Player IDs
    """
    # Separate features and player IDs
    if 'player' in latent_df.columns:
        player_ids = latent_df['player'].values
        features = latent_df.drop(columns=['player']).values
    else:
        player_ids = latent_df.index.values
        features = latent_df.values
    
    # Configure KMeans with optimized parameters
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm='lloyd'  # Use Lloyd's algorithm for better performance
    )
    
    # Fit the model
    labels = kmeans.fit_predict(features)
    
    # Create centroids DataFrame
    feature_names = latent_df.drop(columns=['player']).columns if 'player' in latent_df.columns else latent_df.columns
    centroids = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=feature_names,
        index=[f"cluster_{i}" for i in range(n_clusters)]
    )
    
    return {
        'model': kmeans,
        'labels': labels,
        'centroids': centroids,
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_,
        'features': features,
        'player_ids': player_ids
    }


def evaluate_clustering_metrics(features, labels):
    """
    Evaluate clustering quality using multiple metrics.
    
    Args:
        features: Feature matrix used for clustering
        labels: Cluster labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    return {
        'silhouette_score': silhouette_score(features, labels),
        'davies_bouldin_score': davies_bouldin_score(features, labels),
        'calinski_harabasz_score': calinski_harabasz_score(features, labels)
    }


def find_optimal_k(latent_df, k_range=range(2, 21), init_method='k-means++', n_init=10, random_state=42):
    """
    Find optimal number of clusters using multiple metrics.
    
    Args:
        latent_df: DataFrame with latent features and 'player' column
        k_range: Range of k values to test
        init_method: Initialization method for KMeans
        n_init: Number of times the algorithm is run with different centroid seeds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - 'metrics_df': DataFrame with scores for each k
        - 'best_k_silhouette': Best k according to silhouette score
        - 'best_k_davies_bouldin': Best k according to Davies-Bouldin score
        - 'best_k_calinski_harabasz': Best k according to Calinski-Harabasz score
    """
    # Separate features
    if 'player' in latent_df.columns:
        features = latent_df.drop(columns=['player']).values
    else:
        features = latent_df.values
    
    silhouette_scores = []
    dbi_scores = []
    ch_scores = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init=init_method,
            n_init=n_init,
            random_state=random_state,
            algorithm='lloyd'
        )
        labels = kmeans.fit_predict(features)
        
        silhouette_scores.append(silhouette_score(features, labels))
        dbi_scores.append(davies_bouldin_score(features, labels))
        ch_scores.append(calinski_harabasz_score(features, labels))
        inertias.append(kmeans.inertia_)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'k': list(k_range),
        'silhouette_score': silhouette_scores,
        'davies_bouldin_score': dbi_scores,
        'calinski_harabasz_score': ch_scores,
        'inertia': inertias
    })
    
    # Find best k values
    best_k_silhouette = metrics_df.loc[metrics_df['silhouette_score'].idxmax(), 'k']
    best_k_davies_bouldin = metrics_df.loc[metrics_df['davies_bouldin_score'].idxmin(), 'k']
    best_k_calinski_harabasz = metrics_df.loc[metrics_df['calinski_harabasz_score'].idxmax(), 'k']
    
    return {
        'metrics_df': metrics_df,
        'best_k_silhouette': best_k_silhouette,
        'best_k_davies_bouldin': best_k_davies_bouldin,
        'best_k_calinski_harabasz': best_k_calinski_harabasz
    }
