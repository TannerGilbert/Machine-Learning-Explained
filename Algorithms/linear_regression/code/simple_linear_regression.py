import numpy as np


class SimpleLinearRegression:

    def __init__(self, learning_rate):
        self.m = 0
        self.b = 0
        self.learning_rate = learning_rate

    def cost_function(self, x, y):
        totalError = 0
        for i in range(0, len(x)):
            totalError += (y[i]-(self.m*x[i]+self.b))**2
        return totalError/float(len(x))

    def fit(self, x, y, num_iterations):
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
            if j % 50 == 0:
                print('error:', self.cost_function(x, y))

    def predict(self, xs):
        return [(self.m * x + self.b) for x in xs]


# Testing functionality
if __name__ == '__main__':
    x = np.linspace(0, 100, 50)
    delta = np.random.uniform(-10, 10, x.size)
    y = 0.5 * x + 3 + delta

    model = SimpleLinearRegression(0.0001)
    model.fit(x, y, 100)
    print('Error:', model.cost_function(x, y))
