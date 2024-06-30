import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.tree import DecisionTreeClassifier


class MyDecisionTreeClassifier:

    def __init__(self, max_depth: int = 5, min_split: int = 2):
        self.max_depth = max_depth
        self.min_split = min_split
        self._tree = None

    def fit(self, x: np.array, y: np.array) -> None:
        self._tree = self._build(x, y)

    def predict(self, x: np.array):
        return np.array([self._predict(row, self._tree) for row in x])

    def _gini(self, y: np.array) -> float:
        n = y.shape[0]
        return 1.0 - sum((np.sum(y == c) / n) ** 2 for c in np.unique(y))

    @staticmethod
    def _most_common_label(y) -> np.signedinteger:
        return np.bincount(y).argmax()

    def _best_split(self, x: np.array, y: np.array) -> dict:
        max_gain = -1
        split = None
        n_rows, n_features = x.shape
        if n_features <= 1:
            return None

        curr_impurity = self._gini(y)

        for feature in range(n_features):
            categories = np.unique(x[:, feature])

            for value in categories:
                cat = False
                # Categorical
                if isinstance(value, str) or len(categories) == 2:
                    left = x[:, feature] == value
                    right = x[:, feature] != value
                    cat = True
                # Numerical
                else:
                    left = x[:, feature] <= value
                    right = x[:, feature] > value

                if (sum(left) == 0 and sum(right) != 0) or (sum(left) != 0 and sum(right) == 0):
                    continue

                lgini = self._gini(y[left])
                rgini = self._gini(y[right])
                impurity = (sum(left) * lgini + sum(right) * rgini) / n_rows

                gain = curr_impurity - impurity

                if gain >= max_gain:
                    max_gain = gain
                    split = {
                        "feature": feature,
                        "threshold": value,
                        "left": left,
                        "right": right,
                        "isCategorical": cat
                    }
        return split

    def _build(self, x: np.array, y: np.array, depth: int = 0) -> dict:
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_split:
            value = self._most_common_label(y)
            return {"leaf": True, "value": value}

        split = self._best_split(x, y)
        if split is None:
            value = self._most_common_label(y)
            return {"leaf": True, "value": value}

        left = self._build(x[split["left"]], y[split["left"]], depth + 1)
        right = self._build(x[split["right"]], y[split["right"]], depth + 1)

        return {"leaf": False,
                "feature": split["feature"],
                "threshold": split["threshold"],
                "isCategorical": split["isCategorical"],
                "left": left,
                "right": right,}

    def _predict(self, row: np.array, tree: dict) -> int:
        if tree["leaf"]:
            return tree["value"]
        else:
            # Categorical
            if tree["isCategorical"]:
                if row[tree["feature"]] == tree["threshold"]:
                    return self._predict(row, tree["left"])
                else:
                    return self._predict(row, tree["right"])
            else:
                if row[tree["feature"]] <= tree["threshold"]:
                    return self._predict(row, tree["left"])
                else:
                    return self._predict(row, tree["right"])

    def print_tree(self, tree=None, feature_names=None, depth=0):
        if tree is None:
            return self.print_tree(self._tree, feature_names)
        elif tree["leaf"]:
            print(f'{"|   " * depth}Predict: {tree["value"]}')
        else:
            feature = feature_names[tree["feature"]]
            threshold = tree["threshold"]
            if tree["isCategorical"]:
                # Categorical feature
                print(f"{'|   ' * depth}If {feature} == {threshold}:")
                self.print_tree(tree["left"], feature_names, depth + 1)
                print(f"{'|   ' * depth}Else:")
                self.print_tree(tree["right"], feature_names, depth + 1)
            else:
                # Numerical feature
                print(f"{'|   ' * depth}If {feature} <= {threshold}:")
                self.print_tree(tree["left"], feature_names, depth + 1)
                print(f"{'|   ' * depth}Else:")
                self.print_tree(tree["right"], feature_names, depth + 1)


if __name__ == "__main__":
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        random_state=42,
        n_classes=4
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    myModel = MyDecisionTreeClassifier(max_depth=10, min_split=20)
    myModel.fit(x_train, y_train)
    myModel.print_tree(feature_names=["Unknown" + str(i) for i in range(1, x_train.shape[1] + 1)])
    y_pred = myModel.predict(x_test)

    print("+---------My model------+")
    print("Recall: %.3f" % recall_score(y_pred, y_test, average="weighted"))
    print("F1-score: %.3f" % f1_score(y_pred, y_test, average="weighted"))
    print("AUC score: %.3f" % roc_auc_score(label_binarize(y_pred, classes=[0, 1, 2, 3]),
                                            label_binarize(y_test, classes=[0, 1, 2, 3]),
                                            multi_class="ovo"))
    print("Precision: %.3f" % precision_score(y_pred, y_test, average="weighted"))
    print("Average Precision: %.3f" %
          average_precision_score(label_binarize(y_pred, classes=[0, 1, 2, 3]),
                                  label_binarize(y_test, classes=[0, 1, 2, 3]),
                                  average="weighted"))

    model = DecisionTreeClassifier(max_depth=10, min_samples_split=20)
    model.fit(x_train, y_train)
    y_predic = model.predict(x_test)

    print("\n+----Sklearn model----+")
    print("Recall: %.3f" % recall_score(y_predic, y_test, average="weighted"))
    print("F1-score: %.3f" % f1_score(y_predic, y_test, average="weighted"))
    print("AUC score: %.3f" % roc_auc_score(label_binarize(y_predic, classes=[0, 1, 2, 3]),
                                            label_binarize(y_test, classes=[0, 1, 2, 3]),
                                            multi_class="ovo"))
    print("Precision: %.3f" % precision_score(y_predic, y_test, average="weighted"))
    print("Average Precision: %.3f" %
          average_precision_score(label_binarize(y_predic, classes=[0, 1, 2, 3]),
                                  label_binarize(y_test, classes=[0, 1, 2, 3]),
                                  average="weighted"))
