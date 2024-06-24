import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.weights = np.array([])

    def predict(self, X: np.array) -> np.array:
        x = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        y = x.dot(self.weights)
        return y

    def fit(self, X: np.array, y: np.array) -> None:
        self.weights = np.random.rand(X.shape[1] + 1)
        x = np.concatenate((np.ones(shape=[X.shape[0], 1], dtype=float), X), axis=1)
        y = y.flatten()
        for _ in range(self.epochs):
            gm = self.get_gradient(x, y)
            step_size = self.learning_rate * gm
            if all(abs(step_size) < 0.00001):
                break
            self.weights -= step_size

    def get_gradient(self, x: np.array, y: np.array) -> np.array:
        gm = (1 / len(y)) * x.T.dot((x.dot(self.weights) - y))
        return gm


if __name__ == '__main__':
    linregressor = LinearRegression()
    x = np.random.random(size=(500, 9))
    y = (2 + x.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])))
    linregressor.fit(x, y)
    x_test = np.random.random(size=(5, 9))
    print("Predicted:\t", linregressor.predict(x_test))
    print("Actual:\t", (x_test.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])) + 2).T)
    print("Predicted Coefficients:\t", linregressor.weights)
    print("Actual Coefficients:\t", [2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
