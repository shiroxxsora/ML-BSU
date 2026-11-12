"""Данные, модель и общие функции лабораторной работы №10."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.datasets import mnist


IMAGE_SIZE = 28
PIXELS = IMAGE_SIZE ** 2
SEED = 42
BATCH_SIZE = 256


def load_mnist(path: Path | None):
    if path is None:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        with np.load(path) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]

    x_train = x_train.reshape(-1, PIXELS).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, PIXELS).astype(np.float32) / 255.0
    return x_train, y_train, x_test, y_test


def train_autoencoder(
    inputs, targets, validation_inputs, validation_targets,
    *, epochs, latent_dim=32, activation="relu", loss="binary_crossentropy",
):
    # Каждый эксперимент начинает обучение со свежими весами.
    tf.keras.utils.set_random_seed(SEED)
    image = layers.Input(shape=(PIXELS,))
    encoded = layers.Dense(128, activation=activation)(image)
    encoded = layers.Dense(64, activation=activation)(encoded)
    latent = layers.Dense(latent_dim, activation=activation, name="latent")(encoded)
    decoded = layers.Dense(64, activation=activation)(latent)
    decoded = layers.Dense(128, activation=activation)(decoded)
    output = layers.Dense(PIXELS, activation="sigmoid")(decoded)

    model = Model(image, output)
    model.compile(optimizer="adam", loss=loss)
    history = model.fit(
        inputs, targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
    )
    return model, history.history


def predict(model, inputs):
    return model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)


def reconstruction_mse(images, restored):
    return np.mean(np.square(images - restored), axis=1)


def save_images(path, title, rows, count=10):
    fig, axes = plt.subplots(len(rows), count, figsize=(14, 1.25 * len(rows) + 0.6))
    for axes_row, (label, images) in zip(axes, rows):
        for index, ax in enumerate(axes_row):
            ax.imshow(images[index].reshape(IMAGE_SIZE, IMAGE_SIZE),
                      cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes_row[0].set_ylabel(label, rotation=0, ha="right", va="center")
    fig.suptitle(title)
    save_plot(fig, path)


def save_curves(path, title, histories, loss_label, *, include_train=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        line, = ax.plot(epochs, history["val_loss"], label=f"{label} · проверка")
        if include_train:
            ax.plot(epochs, history["loss"], "--", color=line.get_color(),
                    label=f"{label} · обучение")
    ax.set(title=title, xlabel="Эпоха", ylabel=loss_label)
    ax.grid(alpha=0.25)
    ax.legend()
    save_plot(fig, path)


def save_plot(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def save_metrics(path, metrics):
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
