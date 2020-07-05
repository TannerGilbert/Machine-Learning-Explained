import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class LogisticRegression:

    def __init__(self, learning_rate, num_features, penalty='l2', C=0.1):
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.C = C
        self.b = 0
        self.w = np.zeros((1, num_features))
        assert penalty in ['l2', 'l1', None]

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def cost_function(self, y, y_pred):
        y_T = y.T
        if self.penalty == 'l1':
            return (-1/y.shape[0]) * (np.sum((y_T*np.log(y_pred)) + ((1-y_T) * np.log(1-y_pred))) + self.C * np.sum(np.absolute(self.w)))
        elif self.penalty == 'l2':
            return (-1/y.shape[0]) * (np.sum((y_T*np.log(y_pred)) + ((1-y_T) * np.log(1-y_pred))) + self.C * np.sum(np.square(self.w)))
        else:
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

            #if i % 100 == 0:
                #print('Error:', cost)
        return self

    def predict(self, X):
        predictions = self.sigmoid(np.dot(self.w, X.T) + self.b)[0]
        return [1 if pred >= 0.5 else 0 for pred in predictions]

    def predict_proba(self, X):
        predictions = self.sigmoid(np.dot(self.w, X.T) + self.b)[0]
        return predictions


if __name__ == '__main__':
    df = pd.read_csv('data/heart.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    X, y = [np.array(df.drop('target', axis=1)), np.array(df['target'])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for penalty in ('l2', 'l1', None):
        model = LogisticRegression(0.0001, X.shape[1], penalty=penalty)
        model.fit(X_train, y_train, 500000)
        predictions = model.predict(X_test)
        print('Accuracy:', penalty, (predictions == y_test).sum()/len(predictions)*100)
