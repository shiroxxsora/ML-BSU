import time

from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt

class SGD_Ridge:
    def __init__(self, X, Y, tau, lm=0.001, eps=1e-6, cycles=5000):
        idx = np.random.permutation(X.shape[0]) # Перемешиваем объекты
        training_idx = idx[:len(X)//2] # Половина в обучающую выборку
        test_idx = idx[len(X)//2:] # Половина в тестовую выборку
        self.X_train = np.array([X[i] for i in training_idx])
        self.X_test = np.array([X[i] for i in test_idx])
        self.Y_train = np.array([Y[i] for i in training_idx])
        self.Y_test = np.array([Y[i] for i in test_idx])


        self.w = self.InitW()
        self.cycles = cycles
        self.tau = tau
        self.SGD()
        self.Q(self.X_train, self.Y_train, self.w, self.tau)

    def InitW(self):
        return np.random.rand(len(X[1]))

    def SGD(self):

        self.Q(self.X_train, self.Y_train, self.w, self.tau)
        print("----------------------------------")

        #DrawEduPlots = []
        for j in range(1, self.cycles):
            rand_idx = np.random.randint(0, len(self.X_train)-1)
            x = self.X_train[rand_idx]
            y = self.Y_train[rand_idx]
            h = 1/j
            self.w = self.w*(1 - h*self.tau) - h * self.Gradient(x, y, self.w)

            #DrawEduPlots.append((j, self.Q(self.X_train, self.Y_train, self.w, self.tau)))
        #self.DrawEduPlot(DrawEduPlots)

        print("----------------------------------")
        self.Q(self.X_train, self.Y_train, self.w, self.tau)

    def CrossValidation(self):
        pass

    def r_squared(self, y_true, y_pred):

        # сумма квадратов остатков (ошибок) регрессии
        SSE = np.sum((y_pred - y_true) ** 2)

        # сумма квадратов отклонений
        SST = np.sum((y_true - np.mean(y_true)) ** 2)

        # Коэффициент детерминации
        R_2 = 1 - SSE / SST

        return R_2

    def PredictTest(self):
        pred = np.dot(self.X_test, self.w)
        r2 = self.r_squared(self.Y_test, pred)

        return r2
    def PredictTrain(self):
        pred = np.dot(self.X_train, self.w)
        r2 = self.r_squared(self.Y_train, pred)

        return r2

    def Loss(self, x, y, w, tau):
        return (np.dot(w, x) - y)**2 + 0.5 * tau * (np.linalg.norm(w)**2)

    def Gradient(self, x, y, w):
        return np.dot((np.dot(w, x) - y), x)

    # Показатель качества
    def Q(self, X, Y, w, tau):
        print(f" w = {w} Q = {np.sum([self.Loss(x, y, w, tau) for x, y in zip(X, Y)])/len(X)}")
        return np.sum([self.Loss(x, y, w, tau) for x, y in zip(X, Y)])/len(X)

    def DrawPlot(self):
        pred = np.dot(self.X_train, self.w)
        plt.plot(self.Y_train, pred, 'ro')
        plt.show()

    def DrawEduPlot(self, arr):
        data = np.array(arr)
        data = data.T
        plt.plot(data[0], data[1], 'ro')
        plt.show()


if __name__ == "__main__":

    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features.drop(columns=['model_year', 'origin'])
    Y = auto_mpg.data.targets.to_numpy()

    X['bias'] = -1
    X = X.to_numpy()
    nan_indices = np.argwhere(np.isnan(X).any(axis=1)).flatten()
    print(nan_indices)
    X = np.delete(X, nan_indices, axis=0)
    Y = np.delete(Y, nan_indices)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y.reshape(-1, 1)).ravel()
    print(X)
    print(Y)

    taus = np.logspace(-4, -1, 100)
    r2_scores = []
    r2_scores_test = []
    sgds = []
    for tau in taus:
        sgd = SGD_Ridge(X, Y, tau)
        r2_scores.append(sgd.PredictTrain())
        r2_scores_test.append(sgd.PredictTest())
        sgds.append(sgd)

    print(r2_scores)
    print(r2_scores_test)
    plt.plot(taus, r2_scores, 'r-')
    plt.plot(taus, r2_scores_test, 'g-')
    plt.draw()
    plt.gcf().canvas.flush_events()
    plt.show()

    sgd = sgds[np.argmax(r2_scores_test)]
    Q = sgd.Q(sgd.X_test, sgd.Y_test, sgd.w, sgd.tau)
    r2 = sgd.PredictTest()

    print(f"Функционал качества: {Q}")
    print(f"Веса w: {sgd.w}")
    print(f"Коэф. детерминации: {r2}")

    sgd.DrawPlot()

    #algorithm = SGD_Ridge(X, Y, 1, 1)