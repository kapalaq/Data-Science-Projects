import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor


class MyLinearRegression:

    def __init__(self,
                 learning_rate: float = 0.001,
                 batch_size: int = 10,
                 tolerance: float = 1e-5,
                 regularization: str = "l1",
                 alpha: float = 0.5,
                 l1_ratio: float = 0.15):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tolerance = tolerance
        self._alpha = alpha
        self.weights = np.array([])

        if regularization == "l2":
            self._regularization = self._ridge
        elif regularization == "l1l2":
            self._ratio = l1_ratio
            self._regularization = self._elasticnet
        else:
            self._regularization = self._lasso

    def predict(self, x: np.array) -> np.array:
        if x.shape[1] < self.weights.shape[0]:
            x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)
        y_pred = x.dot(self.weights)
        return y_pred

    def mini_batch_gradient(self, x: np.array, y: np.array) -> np.array:
        gm = (2 / len(y)) * (x.T.dot(self.predict(x) - y) + self._regularization(len(y)))
        return gm

    def fit(self, x: np.array, y: np.array):
        if self.batch_size > x.shape[0]:
            self.batch_size = x.shape[0]
        x = np.concatenate((np.ones(shape=(x.shape[0], 1)), x), axis=1)
        y = y.flatten()
        self.weights = np.random.rand(x.shape[1])
        for _ in range(self.batch_size):
            for i in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
                gradient = self.mini_batch_gradient(x[i: i + self.batch_size],
                                                    y[i: i + self.batch_size])
                step_size = self.learning_rate * gradient
                if np.all(abs(step_size) < self.tolerance):
                    break
                self.weights -= step_size

    def _lasso(self, n: int) -> np.array:
        return self._alpha / n * np.sign(self.weights)

    def _ridge(self, n: int) -> np.array:
        return self._alpha / n * self.weights

    def _elasticnet(self, n: int) -> np.array:
        return self._ratio * self._lasso(n) + (1 - self._ratio) * self._ridge(n)


if __name__ == '__main__':
    linregressor = MyLinearRegression(learning_rate=0.046,
                                      batch_size=1,
                                      tolerance=1e-5,
                                      alpha=0.01)
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

    model = SGDRegressor(penalty="l1",
                         tol=1e-5,
                         alpha=0.01,
                         loss="squared_error")
    model.fit(x_train, y_train)
    y_predic = model.predict(x_test)

    print("\n+----Sklearn model----+")
    print("MSE: %.10f" % mean_squared_error(y_test, y_predic))
    print("R-squared: %.10f" % r2_score(y_test, y_predic))


