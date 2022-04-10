# coding: utf-8
"""
用于识别MNIST的两层神经网络分类器
"""
import numpy as np
from Layer import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
import matplotlib.pyplot as plt
import os


class NN_MNIST(object):
    def __init__(self, learningRate=None, hiddenNodes=None, lam=None, decayRate=0.95, decaySteps=1000, decayLimitRatio=0.1, batchsize=20, epoch=120, print_iter=1000, inputNodes=784, outputNodes=10):
        """
        :param learningRate:    初始学习率
        :param decayRate:       衰减比率
        :param decaySteps:      衰减步长
        指数衰减学习率计算公式为： decayLR = learningRate * decayRate ** (当前已完成迭代次数 / decaySteps)
        :param decayLimitRatio: 衰减极限
        如果decayLR <= decayLimitRatio * learningRate，那么终值衰减

        :param hiddenNodes:     隐藏层结点数
        :param lam:             L2正则化强度
        :param batchsize:       batchsize，必须要能整除训练集、测试集样本数量
        :param epoch:           epoch，全样本循环训练的次数
        :param print_iter:      梯度下降的次数每过print_iter，就会在控制台打印相关信息
        :param inputNodes:      输入层结点数
        :param outputNodes:     输出层结点数
        """
        # 替换None值(为了代码的含义相对清晰，使用None做为默认值。但因为很多运算None无法参与，所以需要替换)
        if learningRate is None:
            learningRate = 3
        if hiddenNodes is None:
            hiddenNodes = 3
        if lam is None:
            lam = 3

        self.learningRate = learningRate
        self.decayRate = decayRate
        self.decaySteps = decaySteps
        self.decayLimitRatio = decayLimitRatio
        self.hiddenNodes = hiddenNodes
        self.lam = lam
        self.batchsize = batchsize
        self.epoch = epoch
        self.print_iter = print_iter
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes

        # 因后续learningRate会更新，因此先保存初始值
        self.learningRate_backup = self.learningRate

        # 定义衍生属性
        self.minLR = self.learningRate * self.decayLimitRatio
        self.lossArray_train = np.zeros(epoch)            # 用于存储训练集每个epoch的loss
        self.lossArray_validation = np.zeros(epoch)       # 用于存储验证集每个epoch的loss
        self.accuracyArray_train = np.zeros(epoch)        # 用于存储训练集每个epoch的accuracy
        self.accuracyArray_validation = np.zeros(epoch)   # 用于存储验证集每个epoch的accuracy

    # 洗牌函数，用于打乱训练样本顺序
    def dataShuffling(self):
        np.random.shuffle(self.trainData)

    # 建立神经网络
    def modelBuilding(self):
        print('建立神经网络...')
        self.layer1 = FullyConnectedLayer(self.inputNodes, self.hiddenNodes)
        self.relu1 = ReLULayer()
        self.layer2 = FullyConnectedLayer(self.hiddenNodes, self.outputNodes)
        self.softmax = SoftmaxLossLayer()
        self.layerList = [self.layer1, self.layer2]
        print('建立完毕！')

    # 初始化模型
    def modelInit(self):
        print('初始化神经网络参数...')
        for layer in self.layerList:
            layer.paramInit()
        print('初始化完毕！')

    # 储存模型
    def modelSaving(self, file):
        print('将模型参数信息存储于 ' + file)
        params = {}
        params['W1'], params['b1'] = self.layer1.save_param()
        params['W2'], params['b2'] = self.layer2.save_param()
        params['learningRate'] = self.learningRate_backup
        params['hiddenNodes'] = self.hiddenNodes
        params['lam'] = self.lam
        params['decayRate'] = self.decayRate
        params['decaySteps'] = self.decaySteps
        params['decayLimitRatio'] = self.decayLimitRatio
        np.save(file, params)

    # 加载模型
    def modelLoading(self, file):
        print('从 ' + file + ' 中读取参数信息')
        params = np.load(file, allow_pickle=True).item()
        self.layer1.load_param(params['W1'], params['b1'])
        self.layer2.load_param(params['W2'], params['b2'])
        self.learningRate = params['learningRate']
        self.hiddenNodes = params['hiddenNodes']
        self.lam = params['lam']
        self.decayRate = params['decayRate']
        self.decaySteps = params['decaySteps']
        self.decayLimitRatio = params['decayLimitRatio']

    # 前向传播
    def forwardPropagation(self, input):
        z1 = self.layer1.forward(input)
        a1 = self.relu1.forward(z1)
        z2 = self.layer2.forward(a1)
        prob = self.softmax.forward(z2)
        return prob

    # 反向传播
    def backwardPropagation(self):
        d_z2 = self.softmax.backward()
        d_a1 = self.layer2.backward(d_z2, self.lam, self.batchsize)
        d_z1 = self.relu1.backward(d_a1)
        d_x = self.layer1.backward(d_z1, self.lam, self.batchsize)

    # 参数更新
    def paramUpdate(self, lr):
        for layer in self.layerList:
            layer.paramUpdate(lr)


    # 模型训练，并存储loss以及accuracy
    def train(self, trainData, validationData):
        self.trainData = trainData
        batch_num_train = self.trainData.shape[0] / self.batchsize
        if batch_num_train % 1 != 0:
            raise Exception('batchsize无法整除训练集样本数！')

        print('训练模型：')
        for epoch_index in range(self.epoch):
            # 洗牌
            self.dataShuffling()
            for batch_index in range(int(batch_num_train)):
                # 提取图像与标签
                batch_images_train = self.trainData[batch_index * self.batchsize:(batch_index + 1) * self.batchsize, :-1]
                batch_labels_train = self.trainData[batch_index * self.batchsize:(batch_index + 1) * self.batchsize, -1]
                # 前向传播
                self.forwardPropagation(batch_images_train)
                # 计算损失，加上L2正则化
                loss = self.softmax.getLoss(batch_labels_train)\
                       + self.lam / 2 / self.batchsize\
                       * (np.sum(self.layer1.W ** 2) + np.sum(self.layer1.b ** 2)
                          + np.sum(self.layer2.W ** 2) + np.sum(self.layer1.b ** 2))
                # 后向传播
                self.backwardPropagation()
                # 参数更新
                self.learningRate *= self.decayRate ** ((epoch_index * batch_num_train + batch_index) / self.decaySteps)
                if self.learningRate <= self.minLR:
                    self.paramUpdate(self.minLR)
                else:
                    self.paramUpdate(self.learningRate)
                # 打印训练信息
                if batch_index % self.print_iter == 0:
                    print('in Epoch %d, and now loss is %.6f' % (epoch_index, loss))

            # 存储一轮epoch训练后的loss以及accuracy信息
            self.storeLA(epoch_index, data=trainData, dataName='测试集')
            self.storeLA(epoch_index, data=validationData, dataName='验证集')


    # 用于存储每轮epoch训练之后，测试集以及验证集上的loss以及accuracy
    def storeLA(self, epoch_index, data, dataName):
        batch_num = data.shape[0] / self.batchsize
        if batch_num % 1 != 0:
            raise Exception('batchsize无法整除' + dataName + '样本数！')
        predictedResults = np.zeros(data.shape[0])
        # 创建array用于存储每一batch的loss
        miniSet_loss = np.zeros(int(batch_num))

        for batch_index in range(int(batch_num)):
            batch_images = data[batch_index * self.batchsize: (batch_index + 1) * self.batchsize, :-1]
            batch_labels = data[batch_index * self.batchsize: (batch_index + 1) * self.batchsize, -1]
            # 前向传播
            prob = self.forwardPropagation(batch_images)
            # 计算该batch下loss
            miniSet_loss[batch_index] = self.softmax.getLoss(batch_labels) \
                   + self.lam / 2 / self.batchsize \
                   * (np.sum(self.layer1.W ** 2) + np.sum(self.layer1.b ** 2)
                      + np.sum(self.layer2.W ** 2) + np.sum(self.layer1.b ** 2))
            # 整理预测结果
            predictedLabels = np.argmax(prob, axis=1)
            predictedResults[batch_index * self.batchsize: (batch_index + 1) * self.batchsize] = predictedLabels
        # 计算loss以及accuracy
        loss = np.mean(miniSet_loss)
        accuracy = np.mean(predictedResults == data[:, -1])
        # 存储loss以及accuracy
        if dataName == '测试集':
            self.lossArray_train[epoch_index] = loss
            self.accuracyArray_train[epoch_index] = accuracy
        elif dataName == '验证集':
            self.lossArray_validation[epoch_index] = loss
            self.accuracyArray_validation[epoch_index] = accuracy
        else:
            raise Exception('参数dataName输入错误！')


    # 画出loss以及accuracy曲线并保存
    def plotLA(self):
        # 设置保存路径
        filePath = os.path.dirname(os.getcwd()) + '\\ImageOfLA_Store\\'
        # 绘制loss曲线
        figureName_loss = 'Loss图像，learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, ' \
                          'decayLimitRatio=%.2f' % (self.learningRate_backup, self.hiddenNodes, self.lam, self.decayRate,
                                                    self.decaySteps, self.decayLimitRatio)
        plt.figure(figureName_loss, figsize=(9, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止图例中文乱码
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        plt.xlabel('epoch')
        # plt.ylabel()
        # plt.ylim((0, 5))                              # 设置y轴范围
        plt.plot(self.lossArray_train, label='训练集loss曲线')
        plt.plot(self.lossArray_validation, label='验证集loss曲线')
        plt.legend()
        # plt.show()
        plt.savefig(filePath + figureName_loss + '.jpg')
        plt.close()

        # 绘制accuracy曲线
        figureName_accuracy = 'Accuracy图像，learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, ' \
                              'decayLimitRatio=%.2f' % (self.learningRate_backup, self.hiddenNodes, self.lam, self.decayRate,
                                                        self.decaySteps, self.decayLimitRatio)
        plt.figure(figureName_accuracy, figsize=(9, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止图例中文乱码
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        plt.xlabel('epoch')
        # plt.ylabel()
        # plt.ylim((0, 1))                              # 设置y轴范围
        plt.plot(self.accuracyArray_train, label='训练集accuracy曲线')
        plt.plot(self.accuracyArray_validation, label='验证集accuracy曲线')
        plt.legend()
        # plt.show()
        plt.savefig(filePath + figureName_accuracy + '.jpg')
        plt.close()


    # 画出各层网络参数并保存
    def plotParameter(self):
        # 设置保存路径
        filePath = os.path.dirname(os.getcwd()) + '\\ImageOfParameter_Store\\'
        # 绘制W1灰度图
        figureName_W1 = 'W1灰度图像，learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, ' \
                          'decayLimitRatio=%.2f' % (self.learningRate_backup, self.hiddenNodes, self.lam, self.decayRate,
                                                    self.decaySteps, self.decayLimitRatio)
        plt.figure(figureName_W1, figsize=(9, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止图例中文乱码
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        plt.imshow(self.layer1.W, cmap='Greys', interpolation=None)
        # plt.legend()
        # plt.show()
        plt.savefig(filePath + figureName_W1 + '.jpg')
        plt.close()

        # 绘制W2灰度图
        figureName_W2 = 'W2灰度图像，learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, ' \
                        'decayLimitRatio=%.2f' % (self.learningRate_backup, self.hiddenNodes, self.lam, self.decayRate,
                                                  self.decaySteps, self.decayLimitRatio)
        plt.figure(figureName_W2, figsize=(9, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止图例中文乱码
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.imshow(self.layer2.W, cmap='Greys', interpolation=None)
        # plt.legend()
        # plt.show()
        plt.savefig(filePath + figureName_W2 + '.jpg')
        plt.close()

        # 绘制b1灰度图
        figureName_b1 = 'b1灰度图像，learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, ' \
                        'decayLimitRatio=%.2f' % (self.learningRate_backup, self.hiddenNodes, self.lam, self.decayRate,
                                                  self.decaySteps, self.decayLimitRatio)
        plt.figure(figureName_b1, figsize=(9, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止图例中文乱码
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.imshow(self.layer1.b.reshape(1, -1), cmap='Greys', interpolation=None)
        # plt.legend()
        # plt.show()
        plt.savefig(filePath + figureName_b1 + '.jpg')
        plt.close()

        # 绘制b2灰度图
        figureName_b2 = 'b2灰度图像，learningRate=%.4f, hiddenNodes=%d, lam=%.2f, decayRate=%.2f, decaySteps=%d, ' \
                        'decayLimitRatio=%.2f' % (self.learningRate_backup, self.hiddenNodes, self.lam, self.decayRate,
                                                  self.decaySteps, self.decayLimitRatio)
        plt.figure(figureName_b2, figsize=(9, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止图例中文乱码
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.imshow(self.layer2.b.reshape(1, -1), cmap='Greys', interpolation=None)
        # plt.legend()
        # plt.show()
        plt.savefig(filePath + figureName_b2 + '.jpg')
        plt.close()


    # 在测试集上的模型测试
    def test(self, testData):
        self.tempData = testData
        batch_num = self.tempData.shape[0] / self.batchsize
        if batch_num % 1 != 0:
            raise Exception('batchsize无法整除测试集样本数！')
        predictedResults = np.zeros(self.tempData.shape[0])

        print('测试模型：')
        for batch_index in range(int(batch_num)):
            batch_images = self.tempData[batch_index * self.batchsize:(batch_index + 1) * self.batchsize, :-1]
            # 前向传播
            prob = self.forwardPropagation(batch_images)
            # 整理预测结果
            predictedLabels = np.argmax(prob, axis=1)
            predictedResults[batch_index * self.batchsize: (batch_index + 1) * self.batchsize] = predictedLabels
        # 识别准确率
        accuracy = np.mean(predictedResults == self.tempData[:, -1])
        print('在测试集上的识别准确率为: %f' % accuracy)
