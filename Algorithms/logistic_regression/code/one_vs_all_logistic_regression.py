import numpy as np
from logistic_regression import LogisticRegression


class LogisticRegressionOneVsAll:

    def __init__(self, learning_rate, num_features, num_classes):
        self.models = [LogisticRegression(learning_rate, num_features) for _ in range(num_classes)]
        

    def fit(self, X, y, num_iterations):
        for i, model in enumerate(self.models):
            y_tmp = (y==i).astype(int)
            model.fit(X, y_tmp, num_iterations)

    def predict(self, X):
        predictions = np.array([model.predict_proba(X) for model in self.models])
        return np.argmax(predictions, axis=0)
