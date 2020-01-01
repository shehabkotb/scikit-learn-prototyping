import numpy as np
import pandas as pd
from Regressor import Regressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('diamonds.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

data = data[(data[['x', 'y','z']] != 0).all(axis=1)]
encoder = LabelEncoder()
data['cut'] = encoder.fit_transform(data['cut'])
data['color'] = encoder.fit_transform(data['color'])
data['clarity'] = encoder.fit_transform(data['clarity'])

y = data['price']
data = data.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=66)
reg = Regressor('KNN')
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))

