import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


class MyLinearRegression:

    def __init__(self, learning_rate: float = 0.001, batch_size: int = 10, tolerance: float = 1e-5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.weights = np.array([])

    def predict(self, x: np.array) -> np.array:
        if x.shape[1] < self.weights.shape[0]:
            x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)
        y_pred = x.dot(self.weights)
        return y_pred

    def mini_batch_gradient(self, x: np.array, y: np.array) -> np.array:
        gm = (2 / len(y)) * x.T.dot(self.predict(x) - y)
        return gm

    def fit(self, x: np.array, y: np.array):
        if self.batch_size > x.shape[0]:
            self.batch_size = x.shape[0]
        x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)
        y = y.flatten()
        self.weights = np.random.rand(x.shape[1])
        for _ in range(self.batch_size):
            for i in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
                gradient = self.mini_batch_gradient(x[i: i + self.batch_size], y[i: i + self.batch_size])
                step_size = self.learning_rate * gradient
                if np.all(abs(step_size) < self.tolerance):
                    break
                self.weights -= step_size


if __name__ == '__main__':
    linregressor = MyLinearRegression(0.1, 5, 1e-14)
    x, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_targets=1,
        random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    linregressor.fit(x_train, y_train)
    y_pred = linregressor.predict(x_test)

    print("+---------My model------+")
    print("MSE: %.10f" % mean_squared_error(y_test, y_pred))
    print("R-squared: %.10f" % r2_score(y_test, y_pred))

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_predic = model.predict(x_test)

    print("\n+----Sklearn model----+")
    print("MSE: %.10f" % mean_squared_error(y_test, y_predic))
    print("R-squared: %.10f" % r2_score(y_test, y_predic))


