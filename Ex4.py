import pandas as pd
import numpy as np

np.random.seed(42)


def load_data(path) -> pd.DataFrame:
    """load data and return random 50-50 train-test split"""
    my_sep = r"\s+"
    # reading from the file
    data = pd.read_csv(path, sep=my_sep)
    data = data.to_numpy()
    # assume labels on last column
    y = data[:, -1]
    X = data[:, :-1]
    # train-test split
    indices = np.random.permutation(data.shape[0])
    training_idx, test_idx = (
        indices[: int(data.shape[0] * 0.5)],
        indices[int(data.shape[0] * 0.5) :],
    )
    X_train, X_test = X[training_idx, :], X[test_idx, :]
    y_train, y_test = y[training_idx], y[test_idx]
    return X_train, X_test, y_train.astype(int), y_test.astype(int)


def lp_distance(x, y, p):
    """compute distance in lp space use l=-1 for l=inf"""
    dist = abs(x - y)

    if p == -1:
        return np.amax(dist, axis=1)
    if p <= 0:
        raise ValueError
    return pow(np.sum(pow(abs(x - y), p), axis=1), 1 / p)


def knn_fit(x_train, y_train, k, p):
    def knn_predict_one(x_pred):
        """predict value of x_pred with knn method with k neighbors in lp space"""
        # compute distance from x_train
        dist = lp_distance(x_train, x_pred, p)
        # take closest k neighbors
        neighbors = np.argsort(dist)[:k]
        # return majority of neighbors labels
        labels = y_train[neighbors]
        # remap -1 to 0 for bincount
        labels_bin = (labels + 1) // 2
        prediction = np.argmax(np.bincount(labels_bin))
        # remap back 0 to -1
        return prediction * 2 - 1

    def knn_predict(x_test):
        """apply prediction for all elements in x_test"""
        return [knn_predict_one(x) for x in x_test]

    # return predicting function with x_train as base set
    return knn_predict


def error(pred, test):
    # how many errors
    misses = pred != test
    return sum(misses) / len(pred)


if __name__ == "__main__":
    print("Dataset: Two Circle Dataset")
    for k in [1, 3, 5, 7, 9]:
        for p in [1, 2, -1]:
            empirical_err_sum = 0
            true_err_sum = 0
            for i in range(100):
                X_train, X_test, y_train, y_test = load_data("two_circle.txt")
                knn_predict = knn_fit(X_train, y_train, k, p)
                y_test_pred = knn_predict(X_test)
                y_train_pred = knn_predict(X_train)
                empirical_err_sum += error(y_train_pred, y_train)
                true_err_sum += error(y_test_pred, y_test)
            lp = "inf" if p == -1 else p
            print(f"Empirical error for k={k} lp= {lp}: {empirical_err_sum/100}")
            print(f"True error for k={k} lp= {lp}: {true_err_sum/100}")
            print(
                f"Difference between errors for k={k} lp= {lp}: {abs(empirical_err_sum/100-true_err_sum/100)}\n"
            )
