# coding: utf-8
"""
神经网络全连接层、激活层
"""
import numpy as np


# 全连接层表达式： Y = XW + 1b  其中Y为p * n维，X为p * m维， W为m * n维， 1为p * 1维， b为1 * n维
class FullyConnectedLayer(object):
    # 构造函数，初始化输入维度m，输出维度n
    def __init__(self, inputDim, outputDim):
        self.inputDim = inputDim
        self.outputDim = outputDim

    # 初始化参数
    def paramInit(self, mean=0.00, std=0.01):
        self.W = np.random.normal(loc=mean, scale=std, size=(self.inputDim, self.outputDim))
        self.b = np.random.normal(loc=mean, scale=std, size=(self.outputDim, ))

    # 前向传播，根据X计算Y
    def forward(self, input):
        self.input = input
        self.output = np.matmul(input, self.W) + self.b
        return self.output

    # 反向传播，根据dY计算dX并计算参数梯度
    def backward(self, forwardDiff, lam, p):
        self.d_W = np.matmul(self.input.T, forwardDiff) + lam / p * self.W
        self.d_b = np.sum(forwardDiff, axis=0) + lam / p * self.b
        backwardDiff = np.dot(forwardDiff, self.W.T)
        return backwardDiff

    # 参数更新
    def paramUpdate(self, lr):
        self.W -= lr * self.d_W
        self.b -= lr * self.d_b

    # 参数加载
    def load_param(self, W, b):
        # assert self.W.shape == W.shape, '加载参数维度不匹配'
        # assert self.b.shape == b.shape, '加载参数维度不匹配'
        self.W = W
        self.b = b

    # 参数保存
    def save_param(self):
        return self.W, self.b


# 中间层激活函数
class ReLULayer(object):
    def __init__(self):
        pass

    # 前向传播
    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    # 反向传播
    def backward(self, forwardDiff):
        backwardDiff = forwardDiff
        backwardDiff[self.input < 0] = 0
        return backwardDiff


# 最终输出端激活函数
class SoftmaxLossLayer(object):
    def __init__(self):
        pass

    # 前向传播
    def forward(self, input):
        maxInput = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - maxInput)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    # 计算损失函数
    def getLoss(self, label):
        self.batchsize = self.prob.shape[0]
        self.labelMatrix = np.zeros_like(self.prob)
        self.labelMatrix[np.arange(self.batchsize), label.astype(int)] = 1.00
        loss = -np.sum(np.log(self.prob) * self.labelMatrix) / self.batchsize
        return loss

    # 反向传播
    def backward(self):
        backwardDiff = (self.prob - self.labelMatrix) / self.batchsize
        return backwardDiff
