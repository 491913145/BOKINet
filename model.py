import numpy as np
from losses import *
from layers import *


class Model():
    def __init__(self, timing=False):
        self.layers = []
        self.timing = timing
        self.parameters = []

    def addlayer(self, layer):
        if layer.__class__.__name__ == 'Conv2D':
            self.parameters += [layer.w, layer.b]
        elif layer.__class__.__name__ == 'Dense':
            self.parameters += [layer.w]
        self.layers += [layer]

    def compile(self, opt, lr=0.01, loss=mse):
        self.loss = loss
        self.opt = opt(self.parameters, lr=lr)

    def forward(self, x):
        for layer in self.layers:
            t_s = time.time()
            x = layer.forward(x)
            t_d = time.time()
            if self.timing:
                print('%s forward: %f ms' % (layer.__class__.__name__, (t_d - t_s) * 1000))
        return x

    def backward(self, loss):
        dt = loss.grad()
        for layer in self.layers[::-1]:
            t_s = time.time()
            dt = layer.backward(dt)
            t_d = time.time()
            if self.timing:
                print('%s backward: %f ms' % (self.layers[i].__class__.__name__, (t_d - t_s) * 1000))

    def fit(self, X, Y, epoch, btsize, verbose=1):
        loss = 0
        for j in range(epoch):
            for p in range(0, len(X), btsize):
                x = X[p:(p + btsize) if p + btsize < len(X) else len(X)]
                y = Y[p:(p + btsize) if p + btsize < len(Y) else len(Y)]
                y_pred = self.forward(x)
                loss = self.loss(y, y_pred)
                self.backward(loss)
                self.opt.update()
                if verbose == 1:
                    print("Epoch: %d/%d   Iter: %d/%d" % (j + 1, epoch, p, len(X)))
                    print("loss: ", loss.loss())
            if verbose:
                print("Epoch: %d/%d" % (j + 1, epoch))
                print("loss: ", loss.loss())

    def predict(self, X):
        return self.forward(X)
