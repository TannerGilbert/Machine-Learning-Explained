from __future__ import annotations
import numpy as np


class SimpleLinearRegression:
    """Simple Linear Regression
    Parameters:
    -----------
    learning_rate: float
        The step length used when following the negative gradient during training.
    """
    def __init__(self, learning_rate: float) -> None:
        self.m = 0
        self.b = 0
        self.learning_rate = learning_rate

    def cost_function(self, x: np.ndarray, y: np.ndarray) -> float:
        total_error = 0
        for i in range(0, len(x)):
            total_error += (y[i]-(self.m*x[i]+self.b))**2
        return total_error/float(len(x))

    def fit(self, x: np.ndarray, y: np.ndarray, num_iterations: int) -> SimpleLinearRegression:
        N = float(len(x))
        for j in range(num_iterations):
            b_gradient = 0
            m_gradient = 0
            for i in range(0, len(x)):
                b_gradient += -(2/N) * (y[i] - ((self.m * x[i]) + self.b))
                m_gradient += -(2/N) * x[i] * \
                    (y[i] - ((self.m * x[i]) + self.b))
            self.b -= (self.learning_rate * b_gradient)
            self.m -= (self.learning_rate * m_gradient)
        return self

    def predict(self, xs: np.ndarray) -> list:
        return [(self.m * x + self.b) for x in xs]


# Testing functionality
if __name__ == '__main__':
    x = np.linspace(0, 100, 50)
    delta = np.random.uniform(-10, 10, x.size)
    y = 0.5 * x + 3 + delta

    model = SimpleLinearRegression(0.0001)
    model.fit(x, y, 100)
    print('Error:', model.cost_function(x, y))
