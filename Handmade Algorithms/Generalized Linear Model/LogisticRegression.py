import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class LogisticRegression:

    def __init__(self, learning_rate=0.01, batch_size=10, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = np.array([])

    def schedule(self, epoch):
        # Cosine Annealing
        # return self.learning_rate * (1 + np.cos(np.pi * epoch / self.epochs)) / 2

        # Step Decay
        return self.learning_rate * (0.9 ** np.floor((1 + epoch) / 10))

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

        for i in range(self.epochs):
            for j in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
                predictions = self.predict(x[j: j + self.batch_size])
                gradient = self.gradient(x[j: j + self.batch_size], y[j: j + self.batch_size], predictions.flatten())
                step_size = self.learning_rate * gradient
                if all(abs(step_size) < 1e-5):
                    break
                self.weights += step_size
                self.learning_rate = self.schedule(i)


if __name__ == '__main__':
    logregressor = LogisticRegression(0.1, 5, 1000)
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        random_state=42
    )
    x_scaled = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
    logregressor.fit(x_train, y_train)
    print("Predict | Actual")
    y_pred = logregressor.normalize(logregressor.predict(x_test))
    for i, j in zip(y_pred, y_test):
        print("\t%d\t|\t%d" % (i, j))

    print("Accuracy: %.3f" % accuracy_score(y_pred, y_test))
    print("AUC score: %.3f" % roc_auc_score(y_pred, y_test))
