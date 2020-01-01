from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

class MyCluster:
    def __init__(self, centeroids):
        self.c = KMeans(n_clusters=centeroids)

    def fit(self, X):
        return self.c.fit(X)

    def predict(self, X_test):
        return self.c.predict(X_test)

    def score(self, X_test, y_test):
        return self.c.score(X_test, y_test)

if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
                [10, 2], [10, 4], [10, 0]])
kmeans = MyCluster(2).fit(X)
print(kmeans.labels_)
print(kmeans.predict([[0, 0], [12, 3]]))
