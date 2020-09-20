import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class MultivariateLinearRegression:

    def __init__(self, learning_rate, penalty='l2', C=1):
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.C = C
        self.w = ""
        assert penalty in ['l2', 'l1', None]

    def cost_function(self, x, y):
        dif = np.dot(x, self.w)-y
        if self.penalty == 'l1':
            cost = (np.sum(dif**2) + self.C * np.sum(np.absolute(self.w))) / (2*np.shape(x)[0]) 
        elif self.penalty == 'l2':
            cost = (np.sum(dif**2) + self.C * np.sum(np.square(self.w))) / (2*np.shape(x)[0])
        else:
            cost = np.sum(dif**2) / (2*np.shape(x)[0])
        return dif, cost

    def fit(self, x, y, num_iterations=10000):
        if self.w == "":
            _, num_features = np.shape(x)
            self.w = np.random.uniform(-1, 1, num_features)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for penalty in ('l2', 'l1', None):
        model = MultivariateLinearRegression(0.0001, penalty=penalty)
        model.fit(X_train, y_train, 10000)
        predictions = model.predict(X_test)
        mse = ((y_test - predictions)**2).mean(axis=0)
        print(penalty, 'loss:', mse)