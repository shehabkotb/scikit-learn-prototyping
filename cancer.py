import numpy as np
import pandas as pd
from Classifier import Classifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Cancer.csv')
y = data.diagnosis  # M or B
list = ['Unnamed: 32', 'id', 'diagnosis']
X = data.drop(list, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = Classifier('RF')
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))
