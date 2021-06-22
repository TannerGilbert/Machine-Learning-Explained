import numpy as np
import matplotlib.pyplot as plt
from sigmoid import Sigmoid

x = np.linspace(-10, 10, 100)
y = Sigmoid()(x)

plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.plot(x, y)
plt.show()