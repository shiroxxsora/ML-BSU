import numpy as np

class GuassianNaiveBayes:
    def __init__(self, X, y):
        print(y.shape)
        self.classes = np.unique(y)
        self.M = []
        self.D = []
        self.priors = []

        for c in self.classes:
            X_c = X[y == c]  # Признаки объектов класса c
            self.M.append(X_c.mean(axis=0))  # Средние значения признаков
            self.D.append(X_c.var(axis=0))  # Дисперсии признаков
            self.priors.append(X_c.shape[0] / X.shape[0])  # Априорная вероятность класса

        # Преобразуем списки в numpy массивы для удобства вычислений
        self.M = np.array(self.M)
        self.D = np.array(self.D) + 1e-6  # Регуляризация D для избежания деления на 0
        self.priors = np.array(self.priors)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        postariors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])  # Логарифм априорной вероятности класса
            likelihood = 0  # Инициализируем сумму для логарифма правдоподобия

            # Для каждого признака рассчитываем логарифм правдоподобия
            for j in range(x.shape[0]):
                mean = self.M[i, j]  # Среднее для признака j класса i
                var = self.D[i, j]  # Дисперсия для признака j класса i
                # Нормальное распределение (логарифм функции плотности)
                likelihood += -0.5 * np.log(2 * np.pi * var) - (x[j] - mean) ** 2 / (2 * var)

            postariors.append(prior + likelihood)  # Логарифм полной вероятности для класса c

        # Возвращаем класс с максимальной вероятностью
        return self.classes[np.argmax(postariors)]

class LaplacianNaiveBayes:
    def __init__(self, X, y):
        y = y.squeeze()
        self.classes = np.unique(y)
        self.M = []
        self.B = []  # Для Лаплассовского масштаба
        self.priors = []

        for c in self.classes:
            X_c = X[y == c]
            # Параметры Лапласа
            self.M.append(X_c.mean(axis=0))  # Средние значения признаков
            self.B.append(np.median(np.abs(X_c - X_c.mean(axis=0)), axis=0))  # Масштаб Лапласа (MAD)
            self.priors.append(X_c.shape[0] / X.shape[0])

        self.M = np.array(self.M)
        self.B = np.array(self.B) + 1e-6  # Регуляризация b для избежания деления на 0
        self.priors = np.array(self.priors)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])  # Логарифм априорной вероятности
            likelihood = 0

            for j in range(x.shape[0]):
                mean = self.M[i, j]  # Медиана
                b = self.B[i, j]  # Масштабный параметр
                likelihood += np.log(1 / (2 * b)) - np.abs(x[j] - mean) / b

            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]