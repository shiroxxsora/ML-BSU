import numpy as np
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
	print(f"TF version {tf.__version__}")


class XORNeural:
	def __init__(self, activation='sigmoid', learning_rate=0.1, epochs=1000):
		self.activation = activation
		self.learning_rate = learning_rate
		self.epochs = epochs

		hidden_layer_count = 4 # Количество нейронов скрытого слоя
		
		# Скрытый слой
		self.w1 = tf.Variable(tf.random.normal([2, hidden_layer_count], stddev=0.1), dtype=tf.float32)
		self.b1 = tf.Variable([-1] * hidden_layer_count, dtype=tf.float32)

		# Выходной слой 1 нейрон
		self.wo = tf.Variable(tf.random.normal([hidden_layer_count, 1], stddev=0.1), dtype=tf.float32)
		self.bo = tf.Variable([-1], dtype=tf.float32)

	def activation_function(self, x):
		if self.activation == 'sigmoid':
			return 1 / (1 + tf.exp(-x))  # Sigmoid
		elif self.activation == 'relu':
			return tf.maximum(0.0, x)  # ReLU
		else:
			return x  # Linear

	def train(self, X, Y):
		optimizer = tf.optimizers.Adam(self.learning_rate)
		
		for epoch in range(self.epochs):
			#Все операции, выполненные в блоке with, будут отслеживаться, чтобы можно было позже вычислить их производные.
			with tf.GradientTape() as tape:
				hidden_layer = self.activation_function(tf.matmul(X, self.w1) + self.b1) # Прямой проход по скрытому слою
				output_layer = 1 / (1 + tf.exp(-(tf.matmul(hidden_layer, self.wo) + self.bo))) # Прямой проход по выходному
				loss = self.binary_cross_entropy(Y, output_layer) # Потеря для бинарных функций
				
			gradients = tape.gradient(loss, [self.w1, self.b1, self.wo, self.bo])
			optimizer.apply_gradients(zip(gradients, [self.w1, self.b1, self.wo, self.bo]))

			if epoch % 100 == 0:
				print(f"{self.activation} Epoch {epoch}, Loss: {loss.numpy()}")
	
	def binary_cross_entropy(self, y_true, y_pred):
		return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-10) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-10))

	def predict(self, X):
		hidden_layer = self.activation_function(tf.matmul(X, self.w1) + self.b1)
		output_layer = 1 / (1 + tf.exp(-(tf.matmul(hidden_layer, self.wo) + self.bo)))
		
		return output_layer.numpy()

