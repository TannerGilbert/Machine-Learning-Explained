import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class MultivariateLinearRegression:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.w = ""

    def cost_function(self, x, y):
        dif = np.dot(x, self.w)-y
        cost = np.sum(dif**2) / (2*np.shape(x)[0])
        return dif, cost

    def fit(self, x, y, num_iterations=10000):
        if self.w == "":
            _, num_features = np.shape(x)
            self.w = np.zeros(num_features)
        for i in range(num_iterations):
            dif, cost = self.cost_function(x, y)
            gradient = np.dot(x.transpose(), dif) / np.shape(x)[0]
            self.w = self.w - self.learning_rate * gradient
            if i % 500 == 0:
                print('error:', cost)

    def predict(self, x):
        return np.dot(x, self.w)


# Testing functionality
if __name__ == '__main__':
    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'])
    le = LabelEncoder()
    iris['label'] = le.fit_transform(iris['label'])
    X = np.array(iris.drop(['petal_width'], axis=1))
    y = np.array(iris['petal_width'])

    model = MultivariateLinearRegression(0.0001)
    model.fit(X, y, 10000)
