import numpy as np
from activation import Liner, Sigmoid, ReLU
import time
from compute import *
from parameters import Parameter


class Dense:
    def __init__(self, inpt_num, output_num, activate=Liner, timing=False):
        n, m = inpt_num, output_num
        self.active = activate
        if self.active.__name__ == "Sigmoid":
            self.w = Parameter(np.random.randn(n + 1, m) * np.sqrt(6. / (n + m + 1)))
        else:
            self.w = Parameter(np.random.uniform(-1, 1, (n + 1, m)) * np.sqrt(6. / (n + m + 1)))
        self.timing = timing

    def forward(self, x):
        self.x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        v = np.einsum('ij,jk->ik', self.x, self.w.data, optimize=True)
        x, self.xp = self.active(v)
        return x

    def backward(self, dt):
        dt = dt * self.xp
        self.w.grad = np.einsum('ij,ik->jk', self.x, dt, optimize=True) / dt.shape[0]
        dt = np.einsum('ij,kj->ik', dt, self.w.data[:-1], optimize=True)
        return dt


class Conv2D:
    def __init__(self, kernel_size, inpt_num, output_num, stride=1, activate=Liner):
        self.k = kernel_size
        self.n, self.m = inpt_num, output_num
        self.act = activate
        self.s = stride
        self.w = Parameter(np.random.uniform(-1, 1, (self.n, self.m, kernel_size, kernel_size)) * np.sqrt(
            6. / (self.n + self.m + kernel_size)))
        self.b = Parameter(np.random.uniform(-1, 1, (1)) * np.sqrt(6. / (self.n + self.m + kernel_size)))

    def forward(self, x):
        n, c, h, w = x.shape
        o_h = int(h / self.s + 0.5)
        o_w = int(w / self.s + 0.5)
        pad_h = max((o_h - 1) * self.s + self.k - h, 0)
        pad_w = max((o_w - 1) * self.s + self.k - w, 0)
        pad_top = pad_h // 2
        pad_btm = pad_h - pad_top
        pad_lft = pad_w // 2
        pad_rht = pad_w - pad_lft
        self.pad = (pad_top, pad_btm, pad_lft, pad_rht)
        self.x = np.lib.pad(x, ((0, 0), (0, 0), (pad_top, pad_btm), (pad_lft, pad_rht)), mode='constant')
        x = convolution(self.x, self.w.data, self.s) + self.b.data
        x, self.xp = self.act(x)
        return x

    def backward(self, dt):
        dt = dt * self.xp
        self.b.grad = np.mean(np.einsum('i...->i', dt, optimize=True))
        dt_inserted = insert_zeros(dt, stride=self.s - 1)
        # 在上面的矩阵外围填补上合适数目的0
        dt = np.lib.pad(dt_inserted, ((0, 0), (0, 0), (self.k - 1, self.k - 1), (self.k - 1, self.k - 1)),
                        mode='constant')
        # 将卷积核旋转180°
        w = rotate_180_degree(self.w.data).transpose(1, 0, 2, 3)
        # 将上面的两个矩阵进行卷积操作，步长为1，求得需要传递给下一层的误差矩阵
        dt = convolution(dt, w, 1)
        dt = dt[:, :, self.pad[0]:-self.pad[1], self.pad[2]:-self.pad[3]]
        # 参数更新
        # 将输入矩阵和插入0的矩阵进行步长为1的卷积，得到卷积核的更新梯度
        x = split_by_strides(self.x, dt_inserted.shape[2], dt_inserted.shape[3], 1)
        self.w.grad = np.einsum('ijkl,iqwekl->qjwe', dt_inserted, x, optimize=True) / x.shape[0]
        return dt


class Pool2D:
    def __init__(self, kernel_size=2, type='max'):
        self.k = kernel_size
        self.s = kernel_size
        self.type = type

    def forward(self, x):
        n, c, h, w = x.shape
        out = x.reshape(n, c, h // self.k, self.k, w // self.k, self.k)
        if self.type == 'max':
            out = out.max(axis=(3, 5))
            self.mask = out.repeat(self.k, axis=1).repeat(self.k, axis=2) != x
        if self.type == 'avg':
            out = out.mean(axis=(4, 5))
        return out

    def backward(self, dt, *args):
        if self.type == 'max':
            dt = dt.repeat(self.k, axis=2).repeat(self.k, axis=3)
            dt[self.mask] = 0
        if self.type == 'avg':
            dt = (dt / self.k ** 2).repeat(self.k, axis=2).repeat(self.k, axis=3)
        return dt


class Flatten:
    def __init__(self):
        self.inpt_shape = ()

    def forward(self, x):
        self.inpt_shape = x.shape
        return np.reshape(np.ravel(x), (x.shape[0], -1))

    def backward(self, dt, *args):
        return np.reshape(dt, self.inpt_shape)
