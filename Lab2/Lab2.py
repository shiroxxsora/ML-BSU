import time
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


def distance(x, y):
    return np.linalg.norm(x - y)


class Kmeans:

    def initialization_centre(self):
        return np.array([self.dataset[i] for i in np.random.choice(self.dataset.shape[0], self.k, replace=False)])

    def expectation(self):
        self.X = [[] for i in range(self.k)]

        # Считаем расстояние от каждого объекта до центров и относим их к ближайшим классам
        for object in self.dataset:
            r = np.array([distance(object, mean) for mean in self.means]) # массив с расстоянием от объекта до центров классов
            self.X[np.argmin(r)].append(object)


    def maximization(self):
        # Пересчитываем центры для каждого класса
        return np.array([np.mean(cluster, axis=0, dtype=np.float64) for cluster in self.X])  # X массив с k классами

    def quality(self):
        s = [np.sum([distance(c[i], c[j]) for i in range(0, len(c)) for j in range(i, len(c))]) / len(c) for c in self.X]
        return s
        # return sum(s)/self.k

    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

        self.COLORS = ('green', 'blue', 'purple', 'black')
        self.X = [[] for i in range(self.k)]
        self.means = self.initialization_centre()

    def run(self):
        previous_means = np.zeros((self.k, 4))

        while not np.array_equal(previous_means, self.means):
            previous_means = self.means
            self.expectation()
            self.means = self.maximization()

    def create_plot(self, ax):
        for i in range(self.k):
            cluster = np.array(self.X[i]).T
            ax.scatter(cluster[0], cluster[1], s=10, color=self.COLORS[i])

        mx = [m[0] for m in self.means]
        my = [m[1] for m in self.means]
        ax.scatter(mx, my, s=50, color='red', marker='x')

        ax.set_xlabel('iris 1st coordinate')
        ax.set_ylabel('iris 2nd coordinate')
        q = self.quality()
        ax.set_title(f"Quality {q[0]}\n{q[1]}\n{q[2]}\n avg. {sum(q)/3}")


if __name__ == "__main__":
    plot_rows = 3
    plot_cols = 4
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(12, 8))
    axes = axes.flatten()

    m = plot_rows * plot_cols

    iris_data = load_iris().data
    for i in range(m):
        algorithm = Kmeans(iris_data, 3)
        algorithm.run()
        algorithm.create_plot(axes[i])

    plt.tight_layout()
    plt.show()
