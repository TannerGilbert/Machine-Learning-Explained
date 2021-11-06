import numpy as np
from tweedie_deviance import TweedieDeviance


class D2Score:
    def __init__(self, power: int) -> None:
        self.tweedie = TweedieDeviance(power)

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return self.loss(y, y_pred)

    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> np.float64:
        return 1 - self.tweedie(y, y_pred) / self.tweedie(y, np.mean(y))
