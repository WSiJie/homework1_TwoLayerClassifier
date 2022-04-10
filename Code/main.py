# coding: utf-8
"""
主函数
"""
from DataProcessing import preProcessing
from NeuralNetwork import NN_MNIST
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# 建模函数
def NN_Building(learningRate, hiddenNodes, lam, trainData, validationData, decayRate=0.95, decaySteps=1000,
                decayLimitRatio=0.1):
    # 生成模型
    model = NN_MNIST(learningRate=learningRate, hiddenNodes=hiddenNodes, lam=lam, decayRate=decayRate,
                     decaySteps=decaySteps, decayLimitRatio=decayLimitRatio)
    # 建立模型
    model.modelBuilding()
    # 模型初始化
    model.modelInit()
    # 模型训练
    model.train(trainData, validationData)
    # 返回模型
    return model


# 存储模型参数，存储loss和accuracy的数据
def dataStoring(model):
    # 设置路径、名称
    filePath = os.path.dirname(os.getcwd())
    fileName = 'learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, decayLimitRatio=%.2f'\
               % (model.learningRate_backup, model.hiddenNodes, model.lam, model.decayRate, model.decaySteps, model.decayLimitRatio)

    # 保存参数
    model.modelSaving(filePath + '\\Parameter_Store\\' + fileName + '.npy')

    # 保存loss和accuracy数据
    combinedArray = np.vstack((model.lossArray_train, model.lossArray_validation, model.accuracyArray_train,
                               model.accuracyArray_validation)).T
    combinedDataframe = pd.DataFrame(combinedArray, columns=['train_loss', 'validation_loss', 'train_accuracy',
                                                             'validation_accuracy'])
    combinedDataframe.to_csv(filePath + '\\DataOfLA_Store\\' + fileName + '.csv')

    # 保存loss和accuracy图像
    model.plotLA()

    # 保存每层网络参数图像
    model.plotParameter()


# 随机生成参数训练模型
def randomModeling(N, trainData, validationData):
    for i in range(N):
        # 随机生成参数
        learningRate = np.random.uniform(low=0, high=0.05)
        hiddenNodes = int(np.random.uniform(low=10, high=80))
        lam = np.random.uniform(low=0, high=0.1)
        decayRate = np.random.uniform(low=0.9, high=1)
        decaySteps = int(np.random.uniform(low=100, high=5000))
        decayLimitRatio = np.random.uniform(low=0.1, high=1)

        # 输出参数
        print('生成参数为：learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, '
              'decayLimitRatio=%.2f' % (learningRate, hiddenNodes, lam, decayRate, decaySteps, decayLimitRatio))

        # 生成模型
        myModel = NN_Building(learningRate=learningRate, hiddenNodes=hiddenNodes, lam=lam, decayRate=decayRate,
                              decaySteps=decaySteps, decayLimitRatio=decayLimitRatio, trainData=trainData,
                              validationData=validationData)

        # 保存模型
        dataStoring(myModel)


# 根据DataOfLA_Store中的数据集展示效果最好的模型
def show_bestModel():
    # 文件路径
    path = os.path.dirname(os.getcwd())
    DataOfLA_path = path + '\\DataOfLA_Store'
    ImageOfLA_path = path + '\\ImageOfLA_Store\\'
    ImageOfParameter_path = path + '\\ImageOfParameter_Store\\'
    # 获取文件夹下所有文件
    files = os.listdir(DataOfLA_path)
    # 创建array存储验证集accuracy
    value_accuracy = np.zeros(len(files))
    # 创建容器存储最优模型相关资料的文件名
    name_ImageOfLA_loss = files.copy()
    name_ImageOfLA_accuracy = files.copy()
    name_ImageOfParameter_W1 = files.copy()
    name_ImageOfParameter_b1 = files.copy()
    name_ImageOfParameter_W2 = files.copy()
    name_ImageOfParameter_b2 = files.copy()
    name_Parameter = files.copy()
    # 循环填充容器
    for i in range(len(files)):
        value_accuracy[i] = pd.read_csv(DataOfLA_path + '\\' + files[i], index_col=0).iloc[-1]['validation_accuracy']
        name_ImageOfLA_loss[i] = 'Loss图像，' + files[i].replace('.csv', '.jpg')
        name_ImageOfLA_accuracy[i] = 'Accuracy图像，' + files[i].replace('.csv', '.jpg')
        name_ImageOfParameter_W1[i] = 'W1灰度图像，' + files[i].replace('.csv', '.jpg')
        name_ImageOfParameter_b1[i] = 'b1灰度图像，' + files[i].replace('.csv', '.jpg')
        name_ImageOfParameter_W2[i] = 'W2灰度图像，' + files[i].replace('.csv', '.jpg')
        name_ImageOfParameter_b2[i] = 'b2灰度图像，' + files[i].replace('.csv', '.jpg')
        name_Parameter[i] = files[i].replace('.csv', '.npy')
    # 获取最优模型文件名
    best_name_ImageOfLA_loss = name_ImageOfLA_loss[np.argmax(value_accuracy)]
    best_name_ImageOfLA_accuracy = name_ImageOfLA_accuracy[np.argmax(value_accuracy)]
    best_name_ImageOfParameter_W1 = name_ImageOfParameter_W1[np.argmax(value_accuracy)]
    best_name_ImageOfParameter_b1 = name_ImageOfParameter_b1[np.argmax(value_accuracy)]
    best_name_ImageOfParameter_W2 = name_ImageOfParameter_W2[np.argmax(value_accuracy)]
    best_name_ImageOfParameter_b2 = name_ImageOfParameter_b2[np.argmax(value_accuracy)]
    best_name_Parameter = name_Parameter[np.argmax(value_accuracy)]
    # 打印最优模型参数信息
    print('===================================================================')
    print('最优模型参数选择结果为：' + best_name_Parameter.replace('.npy', ''))
    print('===================================================================')
    # 读取文件并展示
    plt.figure('最优模型Loss曲线', figsize=(9, 4))
    plt.imshow(plt.imread(ImageOfLA_path + best_name_ImageOfLA_loss))
    # plt.show()
    plt.figure('最优模型Accuracy曲线', figsize=(9, 4))
    plt.imshow(plt.imread(ImageOfLA_path + best_name_ImageOfLA_accuracy))
    # plt.show()
    plt.figure('最优模型W1可视化', figsize=(9, 4))
    plt.imshow(plt.imread(ImageOfParameter_path + best_name_ImageOfParameter_W1))
    # plt.show()
    plt.figure('最优模型b1可视化', figsize=(9, 4))
    plt.imshow(plt.imread(ImageOfParameter_path + best_name_ImageOfParameter_b1))
    # plt.show()
    plt.figure('最优模型W2可视化', figsize=(9, 4))
    plt.imshow(plt.imread(ImageOfParameter_path + best_name_ImageOfParameter_W2))
    # plt.show()
    plt.figure('最优模型b2可视化', figsize=(9, 4))
    plt.imshow(plt.imread(ImageOfParameter_path + best_name_ImageOfParameter_b2))
    plt.show()
    # 返回最优模型参数文件名称
    return best_name_Parameter


# 根据保存的参数文件载入参数，得到模型
def NN_Loading(fileName):
    # 设置文件读取路径
    parameter_filePath = os.path.dirname(os.getcwd()) + '\\Parameter_Store\\'
    # 生成模型
    model = NN_MNIST()
    # 建立模型
    model.modelBuilding()
    # 模型初始化
    model.modelInit()
    # 载入参数
    model.modelLoading(parameter_filePath + fileName)
    # 返回模型
    return model


if __name__ == '__main__':
    # 读取数据集
    trainData, validationData, testData = preProcessing()

    # 随机生成参数训练模型
    # randomModeling(10, trainData=trainData, validationData=validationData)

    # 根据DataOfLA_Store中训练模型结果的数据挑选出最优的神经网络模型，展示loss、accuracy曲线、参数可视化
    bestModel_parameterFileName = show_bestModel()

    # 加载模型
    myModel = NN_Loading(bestModel_parameterFileName)

    # 测试模型
    myModel.test(testData)

    # 调试点
    # pass
