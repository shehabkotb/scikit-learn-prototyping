import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Classifier:
    def __init__(self, method):
        if(method == 'BAYES'):
            self.clf = GaussianNB()
        elif(method == 'DTREE'):
            self.clf = DecisionTreeClassifier()
        elif(method == 'KNN'):
            self.clf = KNeighborsClassifier()
        elif(method == 'RF'):
            self.clf = RandomForestClassifier()
        else:
            #picking bayes as default classifier if input invalid
            self.clf = GaussianNB()

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)


# example of usage
# if __name__ == "__main__":
#     X, y = datasets.load_iris(return_X_y=True)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#     clf = Classifier('KNN')
#     clf.fit(X_train, y_train)
#     print(clf.predict(X_test))
#     print(clf.score(X_test, y_test))


    