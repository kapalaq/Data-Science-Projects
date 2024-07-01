import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


class MyDecisionTreeRegressor:

    def __init__(self, max_depth: int = 5, min_split: int = 2, feature_names: list = None):
        self.max_depth = max_depth
        self.min_split = min_split
        self.feature_names = feature_names
        self._tree = None

    def fit(self, x: np.array, y: np.array) -> np.array:
        self._tree = self._build(x, y)

    def predict(self, x: np.array) -> np.array:
        return np.array([self._predict(self._tree, row) for row in x])

    def print(self) -> None:
        self._print(self._tree, self.feature_names)

    @staticmethod
    def _mse(y: np.array) -> float:
        if y.size == 0:
            return np.Inf
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, x: np.array, y: np.array) -> dict:

        n_samples, n_features = x.shape
        min_mse = self._mse(y)

        split = None
        for feature in range(n_features):

            for i in range(1, n_samples):
                cat = False

                if isinstance(x[i, feature], str) or np.unique(x[:, feature]).size == 2:
                    value = x[i, feature]
                    left = x[:, feature] == value
                    right = x[:, feature] != value
                    cat = True
                else:
                    value = (x[i, feature] + x[i - 1, feature]) / 2
                    left = x[:, feature] <= value
                    right = x[:, feature] > value

                lmse = self._mse(y[left])
                rmse = self._mse(y[right])

                mse = (np.sum(left) * lmse + np.sum(right) * rmse) / n_samples

                if mse < min_mse:
                    min_mse = mse
                    split = {"feature": feature,
                             "threshold": value,
                             "isCategorical": cat,
                             "left": left,
                             "right": right}
        return split

    def _build(self, x: np.array, y: np.array, depth: int = 0) -> dict:
        n_samples, n_features = x.shape

        if depth >= self.max_depth or n_samples < self.min_split or len(np.unique(y)) == 1:
            return {"leaf": True,
                    "value": np.mean(y)}

        split = self._best_split(x, y)
        if split is None:
            return {"leaf": True,
                    "value": np.mean(y)}

        left = self._build(x[split["left"]], y[split["left"]], depth + 1)
        right = self._build(x[split["right"]], y[split["right"]], depth + 1)

        return {"leaf": False,
                "threshold": split["threshold"],
                "feature": split["feature"],
                "isCategorical": split["isCategorical"],
                "left": left, "right": right}

    def _predict(self, tree: dict, row: np.array) -> float:
        if tree["leaf"]:
            return tree["value"]
        else:
            if tree["isCategorical"]:
                if row[tree["feature"]] == tree["threshold"]:
                    return self._predict(tree["left"], row)
                else:
                    return self._predict(tree["right"], row)
            else:
                if row[tree["feature"]] <= tree["threshold"]:
                    return self._predict(tree["left"], row)
                else:
                    return self._predict(tree["right"], row)

    def _print(self, tree: dict, feature_names: list, depth: int = 0) -> None:
        if tree["leaf"]:
            print(f'{"|   " * depth}Predict: {tree["value"]}')
        else:
            feature = feature_names[tree["feature"]]
            threshold = tree["threshold"]
            if tree["isCategorical"]:
                # Categorical feature
                print(f"{'|   ' * depth}If {feature} == {threshold}:")
                self._print(tree["left"], feature_names, depth + 1)
                print(f"{'|   ' * depth}Else:")
                self._print(tree["right"], feature_names, depth + 1)
            else:
                # Numerical feature
                print(f"{'|   ' * depth}If {feature} <= {threshold}:")
                self._print(tree["left"], feature_names, depth + 1)
                print(f"{'|   ' * depth}Else:")
                self._print(tree["right"], feature_names, depth + 1)


if __name__ == '__main__':
    x, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_targets=1,
        random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    features = ["Unknown" + str(i) for i in range(10)]

    myModel = MyDecisionTreeRegressor(max_depth=10, min_split=7, feature_names=features)
    myModel.fit(x_train, y_train)
    y_pred = myModel.predict(x_test)

    myModel.print()

    print("\n+----------My Model----------+")
    print("RMSE: %.6f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R-squared: %.6f" % r2_score(y_test, y_pred))

    model = DecisionTreeRegressor(max_depth=10, min_samples_split=7, criterion="squared_error")
    model.fit(x_train, y_train)
    y_predic = model.predict(x_test)
    print("\n+--------Sklearn Model-------+")
    print("RMSE: %.6f" % np.sqrt(mean_squared_error(y_test, y_predic)))
    print("R-squared: %.6f" % r2_score(y_test, y_predic))
