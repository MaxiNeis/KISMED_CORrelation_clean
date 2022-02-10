# -*- coding: utf-8 -*-
"""
Beispiel Code und  Spielwiese

"""

import csv
import os
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from Skalpell import skalpell
from Preprocessing import preprocessing_training
from ecgdetectors import Detectors
from wettbewerb import load_references

# Beim Erstellen des neuronalen Netzes haben wir uns initial am Timeseries Classification Transformer
# Beispiel von Keras orientiert und von dort aus zu unserem Modell gearbeitet. Keras wiederum 
# referenziert für den Code das Paper "Attention Is All You Need" von Vaswani et al.
# https://keras.io/examples/timeseries/timeseries_transformer_classification/
# 
# Für die Implementierung des Positional Encodings haben wir uns sowohl hinsichtlich der Theorie
# als auch des Codes hieran orientiert.
# https://towardsdatascience.com/concepts-about-positional-encoding-you-might-not-know-about-1f247f4e4e23
#
# Beim Trainieren gehen wir für optimale Ergebnisse von einem ausbalancierten (Label) Datensatz aus
#  
#

### if __name__ == '__main__':  # bei multiprocessing auf Windows notwendig

ecg_leads, ecg_labels, fs, ecg_names = load_references()  # Importiere EKG-Dateien, zugehörige Diagnose, Sampling-Frequenz (Hz) und Name                                                # Sampling-Frequenz 300 Hz

ecg_array = []
afib = []
for idx, ecg_lead in enumerate(ecg_leads):
    ecg_subarrays, is_onedimensional = skalpell(ecg_array)
    if not is_onedimensional:
        for cnt_subarrays in range(len(ecg_subarrays)):
            ecg_subarrays[cnt_subarrays] = preprocessing_training(ecg_subarrays[cnt_subarrays])
            if ecg_labels[idx] == 'N':
                afib.append(0)
                ecg_array.append(ecg_subarrays[cnt_subarrays])
            if ecg_labels[idx] == 'A':
                afib.append(1)
                ecg_array.append(ecg_subarrays[cnt_subarrays])
    else:
        ecg_subarrays = preprocessing_training(ecg_subarrays)
        if ecg_labels[idx] == 'N':
            afib.append(0)
            ecg_array.append(ecg_subarrays)
        if ecg_labels[idx] == 'A':
            afib.append(1)
            ecg_array.append(ecg_subarrays)

x_train = np.asarray(ecg_array)
y_train = np.asarray(afib)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0

n_classes = len(np.unique(y_train))

# Positional Encoding
def pos(x):
  dims = x.shape.as_list()
  d_model = dims[2]
  positions = dims[1]
  matrix = np.zeros((positions, d_model))
  for pos in range(positions):
    for i in range(0,d_model-1,2):
      matrix[pos][i]=math.sin(pos/(10000**((2*i)/d_model)))
      matrix[pos][i+1]=math.cos(pos/(10000**((2*i)/d_model)))
  tensor = tf.constant([matrix], dtype=tf.float32)
  return tensor

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

     #Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0,):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Conv1D(filters=64, kernel_size=64, strides=8, activation="relu", padding='same')(x)
    x = layers.GlobalMaxPooling1D(data_format="channels_first", keepdims=True)(x)
    x = layers.Conv1D(filters=128, kernel_size=32, strides=4, activation="relu", padding='same')(x)
    x = layers.GlobalMaxPooling1D(data_format="channels_first", keepdims=True)(x)
    x = x + pos(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="linear")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(dim, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]
lr = 1e-3

model = build_model(
    input_shape,
    head_size=128,
    num_heads=4,
    ff_dim=512,
    num_transformer_blocks=4,
    mlp_units=[256, 256],
    mlp_dropout=0.5,
    dropout=0.25,
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=100000,
    decay_rate=0.985,
    staircase=True)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy",patience=50, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=800,
    batch_size=256,
    callbacks=callbacks,
)

model.save("Transformer_Encoder.h5")