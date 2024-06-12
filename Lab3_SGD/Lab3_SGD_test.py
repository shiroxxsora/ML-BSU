from sklearn.metrics import r2_score
import numpy as np
from ucimlrepo import fetch_ucirepo

class SGD:

    def __init__(self, X, y, tau, lm=0.001, eps=1e-6, cycles=5000):

        self.X = X  # Объекты
        self.y = y  # Целевые значения
        self.tau = tau  # Коэффициент регуляризации
        self.lm = lm  # Параметр эксп.среднего
        self.eps = eps  # Параметр остановки
        self.cycles = cycles  # Количество циклов

    # Функция потерь
    def loss_function(self, w, X, y, tau):

        # Первая часть МНК
        error = np.square(np.dot(w, X) - y)

        # Регуляризация
        reg = 0.5 * tau * (np.linalg.norm(w)) ** 2

        return error + reg

    # Градиент от функции ошибки
    def gradient(self, w, X, y, tau):

        # Первая часть МНК
        error = np.dot(X, w) - y

        # Градиент функции
        gradient = 2 * np.dot(error, X)

        return gradient

    # Градиентный шаг
    def gradient_step(self, w, X, y, tau, h):

        # Градиент функции
        gradient = self.gradient(w, X, y, tau)

        # Экспоненциальное скользящее среднее
        new_weight = w * (1 - h * tau) - h * gradient

        return new_weight

    # Коэффициент детерминации
    def r_squared(self, y_true, y_pred):

        # сумма квадратов остатков (ошибок) регрессии
        SSE = np.sum((y_pred - y_true) ** 2)

        # сумма квадратов отклонений
        SST = np.sum((y_true - np.mean(y_true)) ** 2)

        # Коэффициент детерминации
        R_2 = 1 - SSE / SST

        return R_2

    # Предсказание
    def predict(self):

        # Начальные веса
        w = np.zeros(self.X.shape[1])

        # Начальный функционал
        Q = np.mean([self.loss_function(w, X_i, y_i, self.tau) for X_i, y_i in zip(self.X, self.y)])

        # Массив функционалов
        Q_plot = [Q]

        # Счетчик итераций
        i = 1
        while self.cycles:

            # Темп обучения
            h = 1 / i

            # Случайный индексы
            k = np.random.randint(0, len(self.X) - 1)

            # Функция потерь по случайным объектам
            loss = self.loss_function(w, self.X[k], self.y[k], self.tau)

            # Перерасчет весов
            w = self.gradient_step(w, self.X[k], self.y[k], self.tau, h)

            # Перерасчет функционала
            Q = self.lm * loss + (1 - self.lm) * Q
            Q_plot.append(Q)

            # Вычисление коэффициента детерминации
            y_true = self.y
            y_pred = np.dot(self.X, w)
            R2 = self.r_squared(y_true, y_pred)

            # Условие выхода
            if abs(Q_plot[i] - Q_plot[i - 1]) < self.eps and Q_plot[i] - Q_plot[i - 1] != 0:
                # print('Выход')
                break

            # Счетик и цикл
            i += 1
            self.cycles -= 1

        return Q_plot, R2, w


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd

# Функция нормализации переменных
scaler = StandardScaler()

auto_mpg = fetch_ucirepo(id=9)

# Инициализация обучающей выборки
df = auto_mpg.data.features

# Удаление ненужных переменных и инициализация датасета в массив
#X = df['bias'] = -1
X = df.drop(columns=['model_year', 'origin'])
y = auto_mpg.data.targets
X = X.to_numpy()
y = y.to_numpy()

# Поиск nan значений
nan_indices = np.argwhere(np.isnan(y)).flatten()
X = np.delete(X, nan_indices, axis=0)
y = np.delete(y, nan_indices)

# Нормализация данных
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Преобразуем обратно в одномерный массив
y = y.ravel()

# Деленение на тестовые и тренировочные выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Визуализация данных
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('График зависимости цены от других параметров')
plt.show()

# Инициализация экземпляра класса
sgd = SGD(X_train, y_train, 0.001)
Q, R2, w = sgd.predict()

print('Функционал качества = ', Q)
print('Коэффициент детерминации = ', R2)

# Визуализация функционала качества
plt.plot(Q)
plt.grid(True)
plt.show()

taus = np.logspace(-4, -1, 100)
r2_scores = []
r2_scores_test = []
Q_plot = []

for tau in taus:
    sgd = SGD(X_train, y_train, tau)
    # print(len(X_train), len(y_train))
    Q, r2, w = sgd.predict()
    r2_scores.append(r2)

    # Тестовая выборка
    y_pred = np.dot(X_test, w)
    r2_test = sgd.r_squared(y_test, y_pred)
    r2_scores_test.append(r2_test)
    # Q_plot.append(Q)

plt.plot(taus, r2_scores)
plt.xlabel('tau')
plt.ylabel('r2')
plt.title('R^2 Score vs. Tau')
plt.show()

plt.plot(taus, r2_scores_test)
plt.xlabel('tau')
plt.ylabel('r2')
plt.title('R^2 Score vs. Tau On Test')
plt.show()

# 0.517497
# -0.0294039
# -0.142621
# -0.0646923