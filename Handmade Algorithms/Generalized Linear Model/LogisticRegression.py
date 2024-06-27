import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.array([])

    def normalize(self, y):
        return np.round(y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        return self.sigmoid(x.dot(self.weights))

    def gradient(self, x, y, predictions):
        return x.T.dot(y - predictions)

    def fit(self, x, y):
        self.weights = np.ones(x.shape[1])
        self.weights[0] = -x.mean()

        for _ in range(self.epochs):
            predictions = self.predict(x)
            gradient = self.gradient(x, y, predictions.flatten())
            step_size = self.learning_rate * gradient
            if all(abs(step_size) < 1e-5):
                break
            self.weights += step_size


if __name__ == '__main__':
    logregressor = LogisticRegression(0.01, 1000)
    x, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=6,
        random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    logregressor.fit(x_train, y_train)
    print("Prediction | Actual")
    pred, act = np.round(logregressor.predict(x_test)), y_test
    for i, j in zip(pred, act):
        print("\t%d\t|\t%d" % (i, j))

    print("Accuracy:", accuracy_score(pred, act))
