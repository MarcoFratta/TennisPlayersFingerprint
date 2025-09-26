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
