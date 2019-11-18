import numpy as np
import pandas as pd


class LogisticRegression:

    def __init__(self, learning_rate, num_features):
        self.learning_rate = learning_rate
        self.b = 0
        self.w = np.zeros((1, num_features))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def cost_function(self, y, y_pred):
        y_T = y.T
        return (-1/y.shape[0]) * (np.sum((y_T*np.log(y_pred)) + ((1-y_T) * np.log(1-y_pred))))

    def fit(self, X, y, num_iterations):
        for i in range(num_iterations):
            pred = self.sigmoid(np.dot(self.w, X.T) + self.b)
            cost = self.cost_function(y, pred)

            # Calculate Gradients/Derivatives
            dw = (1 / X.shape[0]) * (np.dot(X.T, (pred - y.T).T))
            db = (1 / X.shape[0]) * (np.sum(pred - y.T))

            self.w = self.w - (self.learning_rate * dw.T)
            self.b = self.b - (self.learning_rate * db)

            if i % 100 == 0:
                print('Error:', cost)
        return self

    def predict(self, X):
        predictions = self.sigmoid(np.dot(self.w, X.T) + self.b)[0]
        print(predictions)
        return [1 if pred >= 0.5 else 0 for pred in predictions]


if __name__ == '__main__':
    df = pd.read_csv('D:/Datasets/Heart Disease UCI/heart.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    X, y = [np.array(df.drop('target', axis=1)), np.array(df['target'])]
    model = LogisticRegression(0.0001, X.shape[1])
    model.fit(X, y, 500000)
    print(model.predict(X[:5]))
    print(y[:5])
