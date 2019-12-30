import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

class Regressor:
    def __init__(self, method, **kwargs):

        self.RegressorType = method

        if(method == 'LINEAR' or 'POLY'):
            self.clf = LinearRegression()

            # take default degree as 2 if none was given
            if 'degree' in kwargs:
                self.degree = kwargs['degree']
            else:
                self.degree = 2 

        elif(method == 'DTREE'):
            self.clf = DecisionTreeRegressor()
        elif(method == 'KNN'):
            self.clf = KNeighborsRegressor()       
        else:
            #picking linearReg as default regression if input invalid
            self.RegressorType = 'LINEAR'
            self.clf = LinearRegression()

    def fit(self, X_train, y_train):
        if(self.RegressorType != 'POLY'):
            self.clf.fit(X_train, y_train)
        else:
            poly = PolynomialFeatures(degree=self.degree)
            X_poly = poly.fit_transform(X_train)
            self.clf.fit(X_poly, y_train)
            

    def predict(self, X_test):
        if(self.RegressorType != 'POLY'):
            return self.clf.predict(X_test)
        else:
            poly = PolynomialFeatures(degree=self.degree)
            X_test_poly = poly.fit_transform(X_test)
            return self.clf.predict(X_test_poly)

    def score(self, X_test, y_test):
        if(self.RegressorType != 'POLY'):
            return self.clf.score(X_test, y_test)
        else:
            poly = PolynomialFeatures(degree=self.degree)
            X_test_poly = poly.fit_transform(X_test)
            return self.clf.score(X_test_poly, y_test)

# example of usage
# if __name__ == "__main__":
#     X, y = make_regression(n_samples=100, n_features=1, random_state=0, noise=5.0, bias=100.0)
#     plt.scatter(X,y)
#     plt.show()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     clf = Regressor('POLY', degree=3)
#     clf.fit(X_train, y_train)
#     print(clf.predict(X_test))
#     print(y_test)
#     print(clf.score(X_test, y_test))