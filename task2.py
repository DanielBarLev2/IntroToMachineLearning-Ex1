from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
data, labels = mnist["data"], mnist["target"]

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :]
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :]
test_labels = labels[idx[10000:]]


def train_and_predict(x_train: np.ndarray, y_train: np.ndarray, query: np.ndarray, k: int) -> int:
    """
        Train the model and predict the class label for a given query using the k-nearest neighbors algorithm.

        Parameters:
        - x_train (np.ndarray): Training data features,
            where each row represents a sample and each column represents a feature.
        - y_train (np.ndarray): Training data labels.
        - query (np.ndarray): Query data point for which the class label is to be predicted.
        - k (int): Number of nearest neighbors to consider for prediction.

        Returns:
        - int: Predicted class label for the query image.
    """
    # I choose a vectorization approach for the distance computation method.
    x_train_norms = np.sum(x_train ** 2, axis=1, keepdims=True)
    query_norms = np.sum(query ** 2, axis=0, keepdims=True)

    inner_product = np.matmul(x_train, query.T)
    inner_product = inner_product[:, np.newaxis]

    distances = np.sqrt(x_train_norms + query_norms - 2 * inner_product)
    k_nearest_neighbors = np.argsort(distances, axis=0)

    y_predictions = np.zeros(k)
    for i in range(k):
        y_predictions[i] = y_train[k_nearest_neighbors[i][0]]

    y_pred = int(np.argmax(np.bincount(y_predictions.astype(int))))
    return y_pred


def check_accuracy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, k: int) -> float:
    """
        Calculate the accuracy of the k-nearest neighbors algorithm on the test data.

        Parameters:
        - x_train (np.ndarray): Training data features.
        - y_train (np.ndarray): Training data labels.
        - x_test (np.ndarray): Test data features.
        - y_test (np.ndarray): Test data labels.
        - k (int): Number of nearest neighbors to consider for prediction.

        Returns:
        - float: Accuracy of the k-nearest neighbors algorithm on the test data.
    """
    accuracy_list = []

    for i in range(y_test.shape[0]):
        image_query = x_test[i]
        query_prediction = train_and_predict(x_train=x_train, y_train=y_train, query=image_query, k=k)

        if query_prediction == int(y_test[i]):
            accuracy_list.append(1)
        else:
            accuracy_list.append(0)

    accuracy = (sum(accuracy_list) / len(accuracy_list)) * 100
    return accuracy


if __name__ == '__main__':

    accuracies = []

    test_k = False
    test_n = True

    if test_k:
        n = 1000
        k_values = range(1, 101)
        for i, k in enumerate(k_values):
            acc = check_accuracy(x_train=train[:n], y_train=train_labels[:n], x_test=test, y_test=test_labels, k=k)
            accuracies.append(acc)
            print(f' For iter {i} | the accuracy is: {acc}% on k={k}, with n={n}')

        best_k = k_values[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        print(f"Best k: {best_k} with accuracy: {best_accuracy}")

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('k-NN Accuracy as a Function of k')
        plt.grid(True)
        plt.show()

    if test_n:
        n_values = range(5000, 5100)
        k = 1
        for i, n in enumerate(n_values):
            acc = check_accuracy(x_train=train[:n], y_train=train_labels[:n], x_test=test, y_test=test_labels, k=k)
            accuracies.append(acc)
            print(f' For iter {i} | the accuracy is: {acc}% on k={k}, with n={n}')

        best_n = n_values[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        print(f"Best n: {best_n} with accuracy: {best_accuracy}")

        plt.figure(figsize=(10, 6))
        plt.plot(n_values, accuracies, marker='o')
        plt.xlabel('Number of training samples (n)')
        plt.ylabel('Accuracy')
        plt.title('k-NN Accuracy as a Function of Training Set Size (k=1)')
        plt.grid(True)
        plt.show()