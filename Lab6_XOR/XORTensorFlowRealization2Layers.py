import numpy as np
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    print(f"TF версия {tf.__version__}")


class XORNeural:
    def __init__(self, activation='sigmoid', learning_rate=0.1, epochs=1000):
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs

        hidden_layer_count_1 = 4
        hidden_layer_count_2 = 4
        
        self.w1 = tf.Variable(tf.random.normal([2, hidden_layer_count_1], stddev=0.1), dtype=tf.float32)
        self.b1 = tf.Variable([-1] * hidden_layer_count_1, dtype=tf.float32)
        
        self.w2 = tf.Variable(tf.random.normal([hidden_layer_count_1, hidden_layer_count_2], stddev=0.1), dtype=tf.float32)
        self.b2 = tf.Variable([-1] * hidden_layer_count_2, dtype=tf.float32)
        
        self.wo = tf.Variable(tf.random.normal([hidden_layer_count_2, 1], stddev=0.1), dtype=tf.float32)
        self.bo = tf.Variable([-1], dtype=tf.float32)

    def activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + tf.exp(-x))
        elif self.activation == 'relu':
            return tf.maximum(0.0, x)
        else:
            return x

    def train(self, X, Y):
        optimizer = tf.optimizers.Adam(self.learning_rate)
        
        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                hidden_layer_1 = self.activation_function(tf.matmul(X, self.w1) + self.b1)
                hidden_layer_2 = self.activation_function(tf.matmul(hidden_layer_1, self.w2) + self.b2)
                output_layer = 1 / (1 + tf.exp(-(tf.matmul(hidden_layer_2, self.wo) + self.bo)))
                loss = self.binary_cross_entropy(Y, output_layer)
                
            gradients = tape.gradient(loss, [self.w1, self.b1, self.w2, self.b2, self.wo, self.bo])
            optimizer.apply_gradients(zip(gradients, [self.w1, self.b1, self.w2, self.b2, self.wo, self.bo]))

            if epoch % 100 == 0:
                print(f"{self.activation} Эпоха {epoch}, Потери: {loss.numpy()}")

    def binary_cross_entropy(self, y_true, y_pred):
        return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-10) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-10))

    def predict(self, X):
        hidden_layer_1 = self.activation_function(tf.matmul(X, self.w1) + self.b1)
        hidden_layer_2 = self.activation_function(tf.matmul(hidden_layer_1, self.w2) + self.b2)
        output_layer = 1 / (1 + tf.exp(-(tf.matmul(hidden_layer_2, self.wo) + self.bo)))
        
        return output_layer.numpy()
