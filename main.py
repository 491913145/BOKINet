import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import activation
from dataloader import load_mnist, load_sin
from layers import Dense, Conv2D, Pool2D, Flatten
from model import Model
from losses import mse, cross_entry
from compute import accuracy, to_categorical, shuffle
from optimizers import SGD, Adam, RMSprpo

x, y = load_mnist('mnist')
x = x / 255.
y = to_categorical(y, 10)
x, y = shuffle(x, y)
x = x[:100]
y = y[:100]
x = np.reshape(x, (-1, 1, 28, 28))

model = Model()
n = 32
act = activation.ReLU
model.addlayer(Conv2D(3, x.shape[1], n, 1, activate=act))
model.addlayer(Pool2D(2))
model.addlayer(Conv2D(3, n, n, 1, activate=act))
model.addlayer(Pool2D(2))
model.addlayer(Conv2D(3, n, n, 2, activate=act))
model.addlayer(Conv2D(3, n, 1, activate=act))
model.addlayer(Flatten())
model.addlayer(Dense(4 ** 2, y.shape[1], activate=activation.Softmax))

# model.addlayer(Dense(x.shape[1], n,activate=act))
# model.addlayer(Dense(n, n,activate=act))
# model.addlayer(Dense(n, n,activate=act))
# model.addlayer(Dense(n, y.shape[1], activate=activation.Softmax))

model.compile(opt=SGD, lr=1e-2, loss=cross_entry)
model.fit(x, y, epoch=50, btsize=10, verbose=2)
y_pred = model.forward(x)
print("accuracy: ", accuracy(y, y_pred))

# from keras.layers import Conv2D,Flatten,Dense
# from keras.optimizers import SGD
# from keras import models
#
# model = models.Sequential()
# model.add(Conv2D(32,3,strides=2, activation='relu', input_shape=(*x.shape[1:],),padding='same'))
# model.add(Conv2D(32,3,strides=2, activation='relu',padding='same'))
# model.add(Conv2D(32,3,strides=2, activation='relu',padding='same'))
# model.add(Conv2D(1,3,strides=1, activation='relu',padding='same'))
# model.add(Flatten())
# model.add(Dense(10,activation='softmax'))
#
# op = SGD(lr=0.01)
# model.compile(optimizer=op,
#               loss='categorical_crossentropy',
#               metrics=['acc'])

# model.summary()
#
# model.fit(x,y, epochs=100, batch_size=10,verbose=2)
# plt.scatter(x, model.predict(x),label="keras")
# plt.scatter(x, y_pred, label="model")
# plt.scatter(x, y, label="truth")
# plt.legend(loc='upper left')
# plt.show()

# y_pred = model.predict(x)
# y_pred = np.argmax(y_pred,axis=1)
# y = np.argmax(y,axis=1)
# correct_prediction = np.equal(y_pred, y)
# print("accuracy: ",np.mean(correct_prediction))
