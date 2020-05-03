import numpy as np


def Sigmoid(x):
    y = np.exp(x) / (np.exp(x) + 1)
    y_grad = y * (1 - y)
    return [y, y_grad]


def Tanh(x):
    y = np.tanh(x)
    y_grad = 1 - y * y
    return [y, y_grad]


def Swish(x, b):  # b是一个常数，指定b
    y = x * (np.exp(b * x) / (np.exp(b * x) + 1))
    y_grad = np.exp(b * x) / (1 + np.exp(b * x)) + x * (b * np.exp(b * x) / ((1 + np.exp(b * x)) * (1 + np.exp(b * x))))
    return [y, y_grad]


def ELU(x, alpha):  # alpha是个常数，指定alpha
    y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    y_grad = np.where(x > 0, 1, alpha * np.exp(x))
    return [y, y_grad]


def SELU(x, alpha, lamb):  # lamb大于1，指定lamb和alpha
    y = np.where(x > 0, lamb * x, lamb * alpha * (np.exp(x) - 1))
    y_grad = np.where(x > 0, lamb * 1, lamb * alpha * np.exp(x))
    return [y, y_grad]


def ReLU(x):
    y = np.where(x < 0, 0, x)
    y_grad = np.where(x < 0, 0, 1)
    return [y, y_grad]


def PReLU(x, a=2):  # a大于1，指定a
    y = np.where(x < 0, x / a, x)
    y_grad = np.where(x < 0, 1 / a, 1)
    return [y, y_grad]


def LeakyReLU(x, a=2):  # a大于1，指定a
    y = np.where(x < 0, x / a, x)
    y_grad = np.where(x < 0, 1 / a, 1)
    return [y, y_grad]


def Mish(x):
    f = 1 + np.exp(x)
    y = x * ((f * f - 1) / (f * f + 1))
    y_grad = (f * f - 1) / (f * f + 1) + x * (4 * f * (f - 1)) / ((f * f + 1) * (f * f + 1))
    return [y, y_grad]


def ReLU6(x):
    y = np.where(np.where(x < 0, 0, x) > 6, 6, np.where(x < 0, 0, x))
    y_grad = np.where(x > 6, 0, np.where(x < 0, 0, 1))
    return [y, y_grad]


def Hard_Swish(x):
    f = x + 3
    relu6 = np.where(np.where(f < 0, 0, f) > 6, 6, np.where(f < 0, 0, f))
    relu6_grad = np.where(f > 6, 0, np.where(f < 0, 0, 1))
    y = x * relu6 / 6
    y_grad = relu6 / 6 + x * relu6_grad / 6
    return [y, y_grad]


def Hard_Sigmoid(x):
    f = (2 * x + 5) / 10
    y = np.where(np.where(f > 1, 1, f) < 0, 0, np.where(f > 1, 1, f))
    y_grad = np.where(f > 0, np.where(f >= 1, 0, 1 / 5), 0)
    return [y, y_grad]


def Softmax(x):
    shift_x = x - np.max(x, axis=1, keepdims=True)  # 防止输入增大时输出为nan
    exp_x = np.exp(shift_x)
    softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax, 1


def Liner(x):
    return x, 1
