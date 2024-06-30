import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


class MyLogisticRegression:

    def __init__(self, learning_rate: float = 0.01, batch_size: int = 10, epochs: int = 1000, tolerance: float = 1e-5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.weights = np.array([])

    def fit(self, x: np.array, y: np.array) -> None:
        self.weights = np.ones(x.shape[1])
        self.weights[0] = -x.mean()

        for i in range(self.epochs):
            for j in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
                predictions = self._predict(x[j: j + self.batch_size])
                gradient = self._gradient(x[j: j + self.batch_size], y[j: j + self.batch_size], predictions.flatten())
                step_size = self.learning_rate * gradient
                if np.all(abs(step_size) < self.tolerance):
                    break
                self.weights += step_size
                self._schedule(i)

    def predict(self, x: np.array) -> np.array:
        return self._normalize(self._predict(x))

    def _schedule(self, epoch: int) -> None:
        # Cosine Annealing
        # return self.learning_rate * (1 + np.cos(np.pi * epoch / self.epochs)) / 2

        # Step Decay
        self.learning_rate *= (0.9 ** np.floor((1 + epoch) / 10))

    def _normalize(self, y: np.array) -> np.array:
        return np.round(y)

    def _sigmoid(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def _predict(self, x: np.array) -> np.array:
        return self._sigmoid(x.dot(self.weights))

    def _gradient(self, x: np.array, y: np.array, predictions: np.array) -> np.array:
        return x.T.dot(y - predictions)


if __name__ == '__main__':
    logregressor = MyLogisticRegression(0.1, 5, 1000, 1e-6)
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        random_state=42
    )
    x_scaled = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
    logregressor.fit(x_train, y_train)

    y_pred = logregressor.predict(x_test)

    print("+---------My model------+")
    print("Recall: %.3f" % recall_score(y_pred, y_test, average="weighted"))
    print("F1-score: %.3f" % f1_score(y_pred, y_test, average="weighted"))
    print("AUC score: %.3f" % roc_auc_score(y_pred, y_test))
    print("Precision: %.3f" % precision_score(y_pred, y_test, average="weighted"))
    print("Average Precision: %.3f" % average_precision_score(y_pred, y_test, average="weighted"))

    model = LogisticRegressionCV()
    model.fit(x_train, y_train)
    y_predic = model.predict(x_test)

    print("\n+----Sklearn model----+")
    print("Recall: %.3f" % recall_score(y_predic, y_test, average="weighted"))
    print("F1-score: %.3f" % f1_score(y_predic, y_test, average="weighted"))
    print("AUC score: %.3f" % roc_auc_score(y_predic, y_test))
    print("Precision: %.3f" % precision_score(y_predic, y_test, average="weighted"))
    print("Average Precision: %.3f" % average_precision_score(y_predic, y_test, average="weighted"))
