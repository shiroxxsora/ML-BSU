import time

from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt

class SGD_Ridge:
    def __init__(self, X, Y, la, eps):
        np.random.shuffle(X)

        idx = np.random.permutation(X.shape[0])
        training_idx = idx[:len(X)//2]
        test_idx = idx[len(X)//2:]
        self.X_train = np.array([X[i] for i in training_idx])
        self.X_test = np.array([X[i] for i in test_idx])
        self.Y_train = np.array([Y[i] for i in training_idx])
        self.Y_test = np.array([Y[i] for i in test_idx])

        self.w = self.InitW()
        self.tau = 0.001
        self.SGD()
        self.Q(self.X_train, self.Y_train)

    def InitW(self):
        return np.random.rand(len(X[1]))

    def SGD(self):

        self.Q(self.X_test, self.Y_test)
        print("----------------------------------")

        for j in range(1, 5000):
            rand_idx = np.random.randint(0, len(self.X_train)-1)
            x = self.X_train[rand_idx]
            y = self.Y_train[rand_idx]
            h = 1/j
            self.w = self.w*(1 - h*self.tau) - h * self.Gradient(x, y, self.w)

            #self.DrawPlot(j, self.Q(self.X_train, self.Y_train))


        print("----------------------------------")
        self.Q(self.X_test, self.Y_test)

    def CrossValidation(self):
        pass

    def Predict(self):
        pred = []
        for x in self.X_test:
            pred.append(np.dot(x, self.w))

        plt.plot(pred, self.Y_test, 'ro')
        plt.show()

    def Loss(self, x, y, w, tau):
        return (np.dot(w, x) - y)**2 + tau/2 * (np.linalg.norm(w)**2)

    def Gradient(self, x, y, w):
        return (np.dot(w, x) - y) * x

    # Показатель качества
    def Q(self, X, Y):
        print(f" w = {self.w} Q = {np.sum([self.Loss(x, y, self.w, self.tau) for x, y in zip(X, Y)])/len(X)}")
        return np.sum([self.Loss(x, y, self.w, self.tau) for x, y in zip(X, Y)])/len(X)

    def DrawPlot(self, i, q):
        plt.plot(i, q, 'ro')
        plt.draw()
        plt.gcf().canvas.flush_events()


if __name__ == "__main__":

    # fetch dataset
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features.drop(columns=['model_year', 'origin']).to_numpy()
    Y = auto_mpg.data.targets.to_numpy()

    nan_indices = np.argwhere(np.isnan(X).any(axis=1)).flatten()
    print(nan_indices)
    X = np.delete(X, nan_indices, axis=0)
    Y = np.delete(Y, nan_indices)

    taus = np.logspace(-4, -1, 100)
    print(taus)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(Y.reshape(-1, 1))
    print(X)
    print(Y)

    plt.ion()
    algorithm = SGD_Ridge(X, Y, 1, 1)
    plt.ioff()
    plt.show()

    algorithm.Predict()