# homework1_TwoLayerClassifier
《神经网络和深度学习》_第一次作业：构建两层神经网络

 作者：汪思杰 21210980016

Github地址：[构建两层神经网络分类器 (github.com)](https://github.com/WSiJie/homework1_TwoLayerClassifier)

## 项目目的
基于MINIST手写数字集，构建两层神经网络分类器，实现手写数字的识别。

## 项目组成
项目包含数据集、代码和模型输出三大部分：

### 数据集
存于Data目录中，内容为MINIST手写数字数据集，分为训练集数据+训练集标签、测试集数据+测试集标签共四个文件。

### 代码
存于Code目录中：

1、DataProcessing.py负责处理原始MINIST数据集，使其结构便于神经网络模型训练与预测。对其原始训练集按照5:1的方式拆分为训练集和验证集，以便完成神经网路模型的训练与调参；

2、Layer.py中从参数初始化、前向传播、反向传播、参数更新、参数保存、参数加载、计算Loss的角度，构造了神经网络的全连接层、ReLU激活函数层、Softmax激活函数层；

3、NeuralNetwork.py构建了具体应用于MINIST数据集的两层神经网络结构，包括网络构建、参数初始化、前向传播、反向传播、参数更新、参数保存、参数加载、绘制Loss/Accuracy曲线、模型训练、模型测试等功能；

4、main.py是主程序，负责读取数据集、随机生成一系列不同参数的神经网络模型并进行训练、挑选模型仓库中的最好模型、加载模型、测试模型等功能。

### 模型输出
存于DataOfLA_Store、ImageOfLA_Store、ImageOfParameter_Store、Parameter_Store四个目录中：

1、DataOfLA_Store存储各个神经网络模型的Loss和Accuracy曲线原始数据

2、ImageOfLA_Store存储各个神经网络的Loss和Accuracy曲线图像

3、ImageOfParameter_Store存储各个神经网络的参数可视化图像

4、Parameter_Store存储各个神经网络的参数

## 模型使用
运行main.py

相关代码功能说明：

### 读取数据
```python
# 读取数据集
trainData, validationData, testData = preProcessing()
```

### 随机生成一系列学习率(使用了指数衰减的学习率，并且设置了缩减的上限，即学习率不能衰减至初始学习率的某一比例之下)、隐藏层大小、正则化强度的参数训练模型。并存储模型数据
```python
# 随机生成参数训练模型
randomModeling(10, trainData=trainData, validationData=validationData)
```

### 根据已经存储的所有模型，在模型仓库中选择出最优的模型并可视化
```python
# 根据DataOfLA_Store中训练模型结果的数据挑选出最优的神经网络模型，展示loss、accuracy曲线、参数可视化
bestModel_parameterFileName = show_bestModel()
```

### 根据上面挑选最优模型函数的输出，加载模型。
```python
# 加载模型
myModel = NN_Loading(bestModel_parameterFileName)
```

### 测试模型
```python
# 测试模型
myModel.test(testData)
```

### 备注
上面的模型使用中的五个步骤并不是必须连贯的。可以分为：

1、读取数据+随机生成模型并存储；

2、读取数据+在模型仓库中挑选最优模型+加载模型+测试模型

这两大功能，如果只想使用其中一项功能只需注释掉其他功能相应代码即可。

## 最优模型结果
最优模型参数选择结果为：learningRate=0.0445, hiddenNodes=73, lam=0.02, decayRate=0.91, decaySteps=4012, decayLimitRatio=0.39

在测试集上的识别准确率为: 97.61%
