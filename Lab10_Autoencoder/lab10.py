from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from model import ANOMALY_EPOCHS, SEED, forward, train_autoencoder


OUTPUT = Path(__file__).with_name("results")


def mse(original, restored):
    return np.mean((original - restored) ** 2, axis=1)


def save_plot(name):
    plt.savefig(OUTPUT / name, dpi=150, bbox_inches="tight")
    plt.close()


def images(name, rows):
    fig, axes = plt.subplots(len(rows), 10, figsize=(14, 1.6 * len(rows)))
    for axes_row, (label, samples) in zip(axes, rows):
        for ax, sample in zip(axes_row, samples[:10]):
            ax.imshow(sample.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set(xticks=[], yticks=[])
        axes_row[0].set_ylabel(label)
    save_plot(name)


def make_plot(name, histories):
    plt.figure(figsize=(8, 4))

    for label, history in histories.items():
        epochs = range(1, len(history["loss"]) + 1)
        line, = plt.plot(epochs, history["val_loss"], label=f"{label}: val")
        plt.plot(epochs, history["loss"], "--", color=line.get_color(),
                 label=f"{label}: train")
        
    plt.xlabel("epoch")
    plt.ylabel("binary cross entropy")
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot(name)


def main():
    OUTPUT.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255

    # Задание 1. Базовый автокодировщик
    print("\n1. Реконструкция")

    model, history = train_autoencoder(x_train, x_train, x_test, x_test)
    restored = forward(model, x_test)

    print(f"MSE: {mse(x_test, restored).mean():.6f}")

    images("01_reconstruction.png", [("Оригинал", x_test), ("Результат", restored)])
    make_plot("01_learning.png", {"ReLU, 32": history})

    # Задание 2. Обучение только на нулях и поиск других цифр
    print("\n2. Аномалии")

    train_zeros, test_zeros = x_train[y_train == 0], x_test[y_test == 0]
    model, _ = train_autoencoder(train_zeros, train_zeros, test_zeros, test_zeros,
                                epochs=ANOMALY_EPOCHS, loss="mse")
    threshold = np.percentile(mse(train_zeros, forward(model, train_zeros)), 95)
    errors = mse(x_test, forward(model, x_test))
    detected, actual = errors > threshold, y_test != 0
    precision = np.sum(detected & actual) / max(np.sum(detected), 1)

    print(f"Порог: {threshold:.6f}; точность: {precision:.2%}")
    print(f"Найдено аномалий: {detected[actual].mean():.2%}; "
          f"ложных срабатываний: {detected[~actual].mean():.2%}")
    
    plt.figure(figsize=(8, 4))
    bins = np.histogram_bin_edges(errors, bins=70)
    plt.hist(errors[~actual], bins=bins, density=True, alpha=0.6, label="Цифра 0")
    plt.hist(errors[actual], bins=bins, density=True, alpha=0.6, label="Цифры 1-9")
    plt.axvline(threshold, color="black", linestyle="--", label="Порог")
    plt.xlabel("MSE")
    plt.ylabel("Плотность")
    plt.legend()
    save_plot("02_anomalies.png")

    # Задание 3. Размер латентного слоя, шум и активации
    print("\n3. Размер латентного слоя")

    results, histories = {32: restored}, {"32": history}
    for size in (2, 128):
        model, histories[str(size)] = train_autoencoder(
            x_train, x_train, x_test, x_test, latent_dim=size)
        results[size] = forward(model, x_test)

    for size in (2, 32, 128):
        print(f"{size} признаков: MSE {mse(x_test, results[size]).mean():.6f}")

    images("03_latent.png", [("Оригинал", x_test)] +
           [(str(size), results[size]) for size in (2, 32, 128)])
    make_plot("03_latent_learning.png", histories)

    print("\nШумоподавление")

    rng = np.random.default_rng(SEED)
    noisy_train, noisy_test = [
        np.clip(x + 0.4 * rng.standard_normal(x.shape, dtype=np.float32), 0, 1)
        for x in (x_train, x_test)
    ]
    model, _ = train_autoencoder(noisy_train, x_train, noisy_test, x_test)
    denoised = forward(model, noisy_test)

    print(f"MSE до: {mse(x_test, noisy_test).mean():.6f}; "
          f"после: {mse(x_test, denoised).mean():.6f}")
    
    images("03_denoising.png", [("Оригинал", x_test),
           ("С шумом", noisy_test), ("После очистки", denoised)])

    print("\nФункции активации")

    activations = {"relu": history}
    for activation in ("tanh", "elu"):
        _, activations[activation] = train_autoencoder(
            x_train, x_train, x_test, x_test, activation=activation)
        
    for label, h in activations.items():
        print(f"{label}: BCE {h['val_loss'][0]:.6f} → {h['val_loss'][-1]:.6f}")

    make_plot("03_activations.png", activations)

    print(f"\nГрафики: {OUTPUT}")


if __name__ == "__main__":
    main()
