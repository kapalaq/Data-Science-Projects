import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.array([])

    def predict(self, x):
        if x.shape[1] < self.weights.shape[0]:
            x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)
        y_pred = x.dot(self.weights)
        return y_pred

    def gradient(self, x, y):
        gm = (1 / len(y)) * x.T.dot(self.predict(x) - y)
        return gm

    def fit(self, x, y):
        x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)
        y = y.flatten()
        self.weights = np.random.rand(x.shape[1])
        for _ in range(self.epochs):
            gradient = self.gradient(x, y)
            step_size = self.learning_rate * gradient
            if all(abs(step_size) < 1e-5):
                break
            self.weights -= step_size


if __name__ == '__main__':
    linregressor = LinearRegression(0.01, 100000)
    x = np.random.random(size=(500, 9))
    y = (2 + x.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])))
    linregressor.fit(x, y)
    x_test = np.random.random(size=(5, 9))
    print("Predicted:\t", np.round(linregressor.predict(x_test)))
    print("Actual:\t", np.round((x_test.dot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])) + 2).T))
    print("Predicted Coefficients:\t", np.round(linregressor.weights))
    print("Actual Coefficients:\t", [2, 1, 2, 3, 4, 5, 6, 7, 8, 9])
