import numpy as np
import tensorflow as tf

if __name__ == "__main__":
	print(tf.__file__)
	print(tf.__version__)


class Neural:
	def __init__(self, activation='sigmoid', input_size=2, learning_rate=0.1, epochs=1000):
		self.activation = activation
		self.learning_rate = learning_rate
		self.epochs = epochs

		hidden_layer_count1 = 16  # Количество нейронов первого скрытого слоя
		hidden_layer_count2 = 8   # Количество нейронов второго скрытого слоя
		
		# Скрытый слой 1
		self.W1 = tf.Variable(tf.random.normal([input_size, hidden_layer_count1]), dtype=tf.float32)
		self.b1 = tf.Variable(tf.fill([hidden_layer_count1], -1.0), dtype=tf.float32)

		# Скрытый слой 2
		self.W2 = tf.Variable(tf.random.normal([hidden_layer_count1, hidden_layer_count2]), dtype=tf.float32)
		self.b2 = tf.Variable(tf.fill([hidden_layer_count2], -1.0), dtype=tf.float32)

		# Выходной слой 1 нейрон
		self.Wo = tf.Variable(tf.random.normal([hidden_layer_count2, 1]), dtype=tf.float32)
		self.bo = tf.Variable([-1], dtype=tf.float32)

	def activation_function(self, x):
		if self.activation == 'sigmoid':
			return 1 / (1 + tf.exp(-x))  # Sigmoid
		elif self.activation == 'relu':
			return tf.maximum(x, 0.0)  # ReLU
		else:
			return x  # Linear

	def train(self, X, Y):
		print("Запуск тренировки")
		optimizer = tf.optimizers.Adam(self.learning_rate)
		
		for epoch in range(self.epochs):
			#Все операции, выполненные в блоке with, будут отслеживаться, чтобы можно было позже вычислить их производные.
			with tf.GradientTape() as tape:
				# Первый скрытый слой
				hidden_layer1 = self.activation_function(tf.matmul(X, self.W1) + self.b1)

				# Второй скрытый слой
				hidden_layer2 = self.activation_function(tf.matmul(hidden_layer1, self.W2) + self.b2)

				# Выходной слой
				output_layer = (tf.matmul(hidden_layer2, self.Wo) + self.bo)

				# Потеря для бинарных функций
				loss = self.mse(Y, output_layer)
				
			gradients = tape.gradient(loss, [self.W1, self.b1, self.W2, self.b2, self.Wo, self.bo])
			optimizer.apply_gradients(zip(gradients, [self.W1, self.b1, self.W2, self.b2, self.Wo, self.bo]))

			if epoch % 100 == 0:
				print(f"{self.activation} Epoch {epoch}, Loss: {loss.numpy()}")

	def mse(self, y_true, y_pred):
		return tf.reduce_mean(tf.square(y_true - y_pred))

	def predict(self, X):
		hidden_layer1 = self.activation_function(tf.matmul(X, self.W1) + self.b1)
		hidden_layer2 = self.activation_function(tf.matmul(hidden_layer1, self.W2) + self.b2)
		output_layer = (tf.matmul(hidden_layer2, self.Wo) + self.bo)
		
		return output_layer.numpy()
