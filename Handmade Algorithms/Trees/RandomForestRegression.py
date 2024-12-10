import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from typing import List, Dict, Tuple, Any

from DecisionTreeRegression import MyDecisionTreeRegressor


class MyRandomForestRegressor:
    _average_loss: List[float] = list()
    _trees: List[Tuple[MyDecisionTreeRegressor, np.array]] = list()

    def __init__(self, n_estimators: int = 10, max_depth: int = 5,
                 min_split: int = 2, bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_split = min_split
        self.bootstrap = bootstrap

    def fit(self, x: np.array, y: np.array) -> None:
        for i in range(self.n_estimators):
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth, min_split=self.min_split)
            if self.bootstrap:
                rows = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
                cols = np.random.choice(x.shape[1], size=x.shape[1] // 3, replace=False)
            else:
                rows = np.arange(x.shape[0])
                cols = np.random.choice(x.shape[1], size=x.shape[1] // 3, replace=False)
            tree.fit(x[rows, :][:, cols], y[rows])
            self._average_loss.append(np.mean(tree.get_loss_history()))
            self._trees.append((tree, cols))

    def predict(self, x: np.array) -> float:
        ans = np.array([tree.predict(x[:, cols]) for tree, cols in self._trees])
        return np.mean(ans, axis=0)

    def get_average_loss(self) -> float:
        return np.mean(self._average_loss)


if __name__ == '__main__':
    np.random.seed(42)
    x, y = make_regression(
        n_samples=3000,
        n_features=10,
        n_informative=6,
        n_targets=1,
        random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    myModel = MyRandomForestRegressor(n_estimators=10, max_depth=10, min_split=7)
    myModel.fit(x_train, y_train)
    y_pred = myModel.predict(x_test)
    print("\n+----------My Model----------+")
    print("Average loss of the forest: %.6f" % myModel.get_average_loss())
    print("RMSE: %.6f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R-squared: %.6f" % r2_score(y_test, y_pred))

    model = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_split=7, criterion="squared_error")
    model.fit(x_train, y_train)
    y_predic = model.predict(x_test)
    print("\n+--------Sklearn Model-------+")
    print("RMSE: %.6f" % np.sqrt(mean_squared_error(y_test, y_predic)))
    print("R-squared: %.6f" % r2_score(y_test, y_predic))