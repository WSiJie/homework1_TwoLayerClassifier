# coding: utf-8
"""
预处理MNIST数据集，得到标准化的训练集、测试集
"""
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def preProcessing():
    # 获取上一级目录
    path = os.path.dirname(os.getcwd())

    # 打开数据文件
    trainImage = open(path + '\\Data\\' + 'train-images.idx3-ubyte', 'rb')
    trainLabel = open(path + '\\Data\\' + 'train-labels.idx1-ubyte', 'rb')
    testImage = open(path + '\\Data\\' + 't10k-images.idx3-ubyte', 'rb')
    testLabel = open(path + '\\Data\\' + 't10k-labels.idx1-ubyte', 'rb')

    # 获取训练集标签
    struct.unpack('>II', trainLabel.read(8))
    y_trainLabel = np.fromfile(trainLabel, dtype=np.uint8).reshape(-1, 1)                  # (60000, 1)

    # 获取训练集数据
    struct.unpack('>IIII', trainImage.read(16))
    x_train = np.fromfile(trainImage, dtype=np.uint8).reshape(len(y_trainLabel), 784)      # (60000, 784)

    # 获取测试集标签
    struct.unpack('>II', testLabel.read(8))
    y_testLabel = np.fromfile(testLabel, dtype=np.uint8).reshape(-1, 1)                    # (10000, 1)

    # 获取测试集数据
    struct.unpack('>IIII', testImage.read(16))
    x_test = np.fromfile(testImage, dtype=np.uint8).reshape(len(y_testLabel), 784)         # (10000, 784)

    # 标准化
    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    # 数据合并
    trainData = np.concatenate((x_train, y_trainLabel), axis=1)
    testData = np.concatenate((x_test, y_testLabel), axis=1)

    # # 观察图像
    # data = x_train[250, :].reshape(28, 28)
    # plt.imshow(data, cmap='Greys', interpolation=None)
    # plt.show()

    # 关闭数据文件
    trainImage.close()
    trainLabel.close()
    testImage.close()
    testLabel.close()

    # 从训练集中分出一部分作为验证集
    validationData = trainData[-10000:]
    trainData = trainData[: 50000]

    # 最终训练集、验证集、测试集数据量之比为5: 1: 1
    return trainData, validationData, testData

