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
        'Silhouette': silhouette_scores,
        'Davies-Bouldin': dbi_scores,
        'Calinski-Harabasz': ch_scores,
        'inertia': inertias
    })
    
    # Find best k values
    best_k_silhouette = metrics_df.loc[metrics_df['Silhouette'].idxmax(), 'k']
    best_k_davies_bouldin = metrics_df.loc[metrics_df['Davies-Bouldin'].idxmin(), 'k']
    best_k_calinski_harabasz = metrics_df.loc[metrics_df['Calinski-Harabasz'].idxmax(), 'k']
    
    return {
        'metrics_df': metrics_df,
        'best_k_silhouette': best_k_silhouette,
        'best_k_davies_bouldin': best_k_davies_bouldin,
        'best_k_calinski_harabasz': best_k_calinski_harabasz
    }
