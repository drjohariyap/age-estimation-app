"""
Train CNN model for Dental Age Estimation
This script:
1. Loads your labels.csv
2. Builds an EfficientNetB0 CNN (regression)
3. Trains the model
4. Saves the trained model as best_age_cnn.h5
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "labels.csv"        # Your dataset labels
IMAGE_FOLDER = "./images"      # Folder containing radiograph images
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
MODEL_OUTPUT = "best_age_cnn.h5"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["image_path", "age"])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================================================
# IMAGE LOADER
# =========================================================
def load_image(path, label):
    full_path = tf.strings.join([IMAGE_FOLDER, "/", path])
    img = tf.io.read_file(full_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.float32)

def df_to_dataset(dataframe):
    paths = dataframe["image_path"].values
    ages = dataframe["age"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, ages))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = df_to_dataset(train_df)
test_ds = df_to_dataset(test_df)

# =========================================================
# BUILD MODEL (EfficientNetB0 for Regression)
# =========================================================
def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model

model = build_model()
model.summary()

# =========================================================
# TRAIN MODEL
# =========================================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_OUTPUT,
        monitor="val_mean_absolute_error",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# SAVE FINAL MODEL
# =========================================================
model.save(MODEL_OUTPUT)
print("Model saved as:", MODEL_OUTPUT)
