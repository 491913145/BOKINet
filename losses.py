import numpy as np


class mse:
    def __init__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred

    def loss(self):
        return np.mean(1 / 2 * (self.y - self.y_pred) ** 2)

    def grad(self):
        return self.y - self.y_pred


class cross_entry:
    def __init__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred

    def loss(self):
        return np.mean(-np.sum(self.y * np.log(self.y_pred), axis=1))

    def grad(self):
        return self.y - self.y_pred
