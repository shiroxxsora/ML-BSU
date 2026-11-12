import tensorflow as tf
from tensorflow.keras import Model, layers


EPOCHS = 50
ANOMALY_EPOCHS = 100
BATCH_SIZE = 256
SEED = 42


def train_autoencoder(
    inputs, targets, validation_inputs, validation_targets,
    *, epochs=EPOCHS, latent_dim=32, activation="relu", loss="binary_crossentropy",
):
    tf.keras.utils.set_random_seed(SEED)
    image = layers.Input(shape=(784,))
    encoded1 = layers.Dense(128, activation=activation)(image)
    encoded2 = layers.Dense(64, activation=activation)(encoded1)
    latent = layers.Dense(latent_dim, activation=activation, name="latent")(encoded2)
    decoded1 = layers.Dense(64, activation=activation)(latent)
    decoded2 = layers.Dense(128, activation=activation)(decoded1)
    output = layers.Dense(784, activation="sigmoid")(decoded2)

    model = Model(image, output)
    model.compile(optimizer="adam", loss=loss)
    history = model.fit(
        inputs, targets,
        validation_data=(validation_inputs, validation_targets),
        epochs=epochs, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
    )
    return model, history.history


def forward(model, inputs):
    return model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)

