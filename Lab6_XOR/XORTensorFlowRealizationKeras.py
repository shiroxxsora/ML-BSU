import numpy as np
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
	print(f"TF version {tf.__version__}")


class XORSigmoid:
	def __init__(self):
		inputs = keras.Input(shape=(2,)) # Входной слой с 2 параметрами
		x = keras.layers.Dense(4, activation="sigmoid")(inputs) # Скрытый слой с 4 нейронами
		outputs = keras.layers.Dense(1, activation="sigmoid")(x) # Выходной слой с 1 нейроном для XOR
		self.model = keras.Model(inputs=inputs, outputs=outputs)

		# Компилируем модель
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	def train(self, X, y):
		# Обучение модели
		self.model.fit(X, y, epochs=2000, verbose=0)

		# Оценка точности
		loss, accuracy = self.model.evaluate(X, y, verbose=0)
		print(f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

		# Проверка результатов
		predictions = self.model.predict(X)
		for i, (input_data, prediction) in enumerate(zip(X, predictions)):
			print(f"Sigmoid Input: {input_data}, Predicted: {prediction[0]:.4f}, Actual: {y[i]}")


class XORLinear:
	def __init__(self):
		inputs = keras.Input(shape=(2,)) # Входной слой с 2 параметрами
		x = keras.layers.Dense(4, activation="linear")(inputs) # Скрытый слой с 32 нейронами
		outputs = keras.layers.Dense(1, activation="linear")(x) # Выходной слой с 1 нейроном для XOR
		self.model = keras.Model(inputs=inputs, outputs=outputs)

		# Компилируем модель
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	def train(self, X, y):

		# Обучение модели
		self.model.fit(X, y, epochs=2000, verbose=0)

		# Оценка точности
		loss, accuracy = self.model.evaluate(X, y, verbose=0)
		print(f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

		# Проверка результатов
		predictions = self.model.predict(X)
		for i, (input_data, prediction) in enumerate(zip(X, predictions)):
			print(f"Linear Input: {input_data}, Predicted: {prediction[0]:.4f}, Actual: {y[i]}")


class XORReLu:
	def __init__(self):
		# Создание модели через апи keras
		inputs = keras.Input(shape=(2,)) # Входной слой с 2 параметрами
		x = keras.layers.Dense(4, activation="relu")(inputs) # Скрытый слой с 32 нейронами
		outputs = keras.layers.Dense(1, activation="relu")(x) # Выходной слой с 1 нейроном для XOR
		self.model = keras.Model(inputs=inputs, outputs=outputs)

		# Компилируем модель
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	def train(self, X, y):

		# Обучение модели
		self.model.fit(X, y, epochs=2000, verbose=0)

		# Оценка точности
		loss, accuracy = self.model.evaluate(X, y, verbose=0)
		print(f"Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

		# Проверка результатов
		predictions = self.model.predict(X)
		for i, (input_data, prediction) in enumerate(zip(X, predictions)):
			print(f"ReLU Input: {input_data}, Predicted: {prediction[0]:.4f}, Actual: {y[i]}")