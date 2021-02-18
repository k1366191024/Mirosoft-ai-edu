#神经网络任务二
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import sys
import pickle
from enum import Enum
from pathlib import Path
from matplotlib.colors import LogNorm

# from HelperClass2.NeuralNet_2_2 import *
# from HelperClass2.Visualizer_1_1 import *
train_data_name = 'D:\iris - train.npy'
test_data_name = 'D:\iris - test.npy'

def DrawThreeCategoryPoints(X1, X2, Y_onehot, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['b', 'r', 'g']
    shapes = ['o', 'x', 's']
    assert(X1.shape[0] == X2.shape[0] == Y_onehot.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = np.argmax(Y_onehot[i])
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    #end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def ShowClassificationResult25D(net, count, title):
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    output = net.inference(input)
    if net.hp.net_type == NetType.BinaryClassifier:
        Z = output.reshape(count,count)
    elif net.hp.net_type == NetType.MultipleClassifier:
        sm = np.argmax(output, axis=1)
        Z = sm.reshape(count,count)

    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, zorder=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)

class WeightsBias_1_0(object):
    def __init__(self, n_input, n_output, init_method, eta):
        self.num_input = n_input
        self.num_output = n_output
        self.init_method = init_method
        self.eta = eta
        self.initial_value_filename = str.format("w_{0}_{1}_{2}_init", self.num_input, self.num_output, self.init_method.name)

    def InitializeWeights(self, folder, create_new):
        self.folder = folder
        if create_new:
            self.__CreateNew()
        else:
            self.__LoadExistingParameters()
        # end if
        #self.__CreateOptimizers()

        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

    def __CreateNew(self):
        self.W, self.B = WeightsBias_1_0.InitialParameters(self.num_input, self.num_output, self.init_method)
        self.__SaveInitialValue()
        
    def __LoadExistingParameters(self):
        file_name = str.format("{0}/{1}.npz", self.folder, self.initial_value_filename)
        w_file = Path(file_name)
        if w_file.exists():
            self.__LoadInitialValue()
        else:
            self.__CreateNew()
        # end if

    def Update(self):
        self.W = self.W - self.eta * self.dW
        self.B = self.B - self.eta * self.dB

    def __SaveInitialValue(self):
        file_name = str.format("{0}/{1}.npz", self.folder, self.initial_value_filename)
        np.savez(file_name, weights=self.W, bias=self.B)

    def __LoadInitialValue(self):
        file_name = str.format("{0}/{1}.npz", self.folder, self.initial_value_filename)
        data = np.load(file_name)
        self.W = data["weights"]
        self.B = data["bias"]

    def SaveResultValue(self, folder, name):
        file_name = str.format("{0}/{1}.npz", folder, name)
        np.savez(file_name, weights=self.W, bias=self.B)

    def LoadResultValue(self, folder, name):
        file_name = str.format("{0}/{1}.npz", folder, name)
        data = np.load(file_name)
        self.W = data["weights"]
        self.B = data["bias"]

    @staticmethod
    def InitialParameters(num_input, num_output, method):
        if method == InitialMethod.Zero:
            # zero
            W = np.zeros((num_input, num_output))
        elif method == InitialMethod.Normal:
            # normalize
            W = np.random.normal(size=(num_input, num_output))
        elif method == InitialMethod.MSRA:
            W = np.random.normal(0, np.sqrt(2/num_output), size=(num_input, num_output))
        elif method == InitialMethod.Xavier:
            # xavier
            W = np.random.uniform(-np.sqrt(6/(num_output+num_input)),
                                  np.sqrt(6/(num_output+num_input)),
                                  size=(num_input, num_output))
        # end if
        B = np.zeros((1, num_output))
        return W, B

class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

class InitialMethod(Enum):
    Zero = 0,
    Normal = 1,
    Xavier = 2,
    MSRA = 3

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class TrainingHistory_2_2(object):
    def __init__(self):
        self.loss_train = []
        self.accuracy_train = []
        self.iteration_seq = []
        self.epoch_seq = []
        self.loss_val = []
        self.accuracy_val = []
  
    def Add(self, epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld):
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)

        return False

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        fig = plt.figure(figsize=(12,5))

        axes = plt.subplot(1,2,1)
        #p2, = axes.plot(self.iteration_seq, self.loss_train)
        #p1, = axes.plot(self.iteration_seq, self.loss_val)
        p2, = axes.plot(self.epoch_seq, self.loss_train)
        p1, = axes.plot(self.epoch_seq, self.loss_val)
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Loss")
        axes.set_ylabel("loss")
        axes.set_xlabel("epoch")
        if xmin != None or xmax != None or ymin != None or ymax != None:
            axes.axis([xmin, xmax, ymin, ymax])
        
        axes = plt.subplot(1,2,2)
        #p2, = axes.plot(self.iteration_seq, self.accuracy_train)
        #p1, = axes.plot(self.iteration_seq, self.accuracy_val)
        p2, = axes.plot(self.epoch_seq, self.accuracy_train)
        p1, = axes.plot(self.epoch_seq, self.accuracy_val)
        axes.legend([p1,p2], ["validation","train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")
        axes.set_xlabel("epoch")
        
        title = params.toString()
        plt.suptitle(title)
        plt.show()
        return title

    def ShowLossHistory4(self, axes, params, xmin=None, xmax=None, ymin=None, ymax=None):
        p2, = axes.plot(self.epoch_seq, self.loss_train)
        p1, = axes.plot(self.epoch_seq, self.loss_val)
        title = params.toString()
        axes.set_title(title)
        axes.set_xlabel("epoch")
        axes.set_ylabel("loss")
        if xmin != None and ymin != None:
            axes.axis([xmin, xmax, ymin, ymax])
        return title

    def Dump(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump(self, f)

    def Load(file_name):
        f = open(file_name, 'rb')
        lh = pickle.load(f)
        return lh

    def GetEpochNumber(self):
        return self.epoch_seq[-1]

    def GetLatestAverageLoss(self, count=10):
        total = len(self.loss_val)
        if count >= total:
            count = total
        tmp = self.loss_val[total-count:total]
        return sum(tmp)/count

class LossFunction_1_1(object):
    def __init__(self, net_type):
        self.net_type = net_type
    # end def

    # fcFunc: feed forward calculation
    def CheckLoss(self, A, Y):
        m = Y.shape[0]
        if self.net_type == NetType.Fitting:
            loss = self.MSE(A, Y, m)
        elif self.net_type == NetType.BinaryClassifier:
            loss = self.CE2(A, Y, m)
        elif self.net_type == NetType.MultipleClassifier:
            loss = self.CE3(A, Y, m)
        #end if
        return loss
    # end def

    def MSE(self, A, Y, count):
        p1 = A - Y
        LOSS = np.multiply(p1, p1)
        loss = LOSS.sum()/count/2
        return loss
    # end def

    # for binary classifier
    def CE2(self, A, Y, count):
        p1 = 1 - Y
        p2 = np.log(1 - A)
        p3 = np.log(A)

        p4 = np.multiply(p1 ,p2)
        p5 = np.multiply(Y, p3)

        LOSS = np.sum(-(p4 + p5))  #binary classification
        loss = LOSS / count
        return loss
    # end def

    # for multiple classifier
    def CE3(self, A, Y, count):
        p1 = np.log(A)
        p2 =  np.multiply(Y, p1)
        LOSS = np.sum(-p2) 
        loss = LOSS / count
        return loss
    # end def

class CActivator(object):
    # z = 本层的wx+b计算值矩阵
    def forward(self, z):
        pass

    # z = 本层的wx+b计算值矩阵
    # a = 本层的激活函数输出值矩阵
    # delta = 上（后）层反传回来的梯度值矩阵
    def backward(self, z, a, delta):
        pass

class Sigmoid(CActivator):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return dz, da
class Tanh(CActivator):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
        return a

    def backward(self, z, a, delta):
        da = 1 - np.multiply(a, a)
        dz = np.multiply(delta, da)
        return dz, da

class CClassifier(object):
    def forward(self, z):
        pass

# equal to sigmoid but it is used as classification function
class Logistic(CClassifier):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

class Softmax(CClassifier):
    def forward(self, z):
        shift_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift_z)
        a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return a


class NeuralNet_2_2(object):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.subfolder = os.getcwd() + "\\" + self.__create_subfolder()
        print(self.subfolder)

        self.wb1 = WeightsBias_1_0(self.hp.num_input, self.hp.num_hidden, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias_1_0(self.hp.num_hidden, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("\\")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def forward(self, batch_x):
        # layer 1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        # layer 2
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A2 = Logistic().forward(self.Z2)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A2 = Softmax().forward(self.Z2)
        else:   # NetType.Fitting
            self.A2 = self.Z2
        #end if
        self.output = self.A2

    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]
        # 第二层的梯度输入 公式5
        dZ2 = self.A2 - batch_y
        # 第二层的权重和偏移 公式6
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式7 对于多样本计算，需要在横轴上做sum，得到平均值
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 
        # 第一层的梯度输入 公式8
        d1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, d1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        # 公式12 对于多样本计算，需要在横轴上做sum，得到平均值
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 

    def update(self):
        self.wb1.Update()
        self.wb2.Update()

    def inference(self, x):
        self.forward(x)
        return self.output

    def train(self, dataReader, checkpoint, need_test):
        # calculate loss to decide the stop condition
        self.loss_trace = TrainingHistory_2_2()
        self.loss_func = LossFunction_1_1(self.hp.net_type)
        if self.hp.batch_size == -1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.hp.max_epoch):
            #print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # get z from x,y
                batch_a = self.forward(batch_x)
                # calculate gradient of w and b
                self.backward(batch_x, batch_y, batch_a)
                # update w,b
                self.update()

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration+1) % checkpoint_iteration == 0:
                    need_stop = self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
                    if need_stop:
                        break                
                    #end if
                #end if
            # end for
            if need_stop:
                break
        # end for
        self.SaveResult()
        #self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
        if need_test:
            print("testing...")
            accuracy = self.Test(dataReader)
            print(accuracy)
        # end if

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))

        # calculate train loss
        self.forward(train_x)
        loss_train = self.loss_func.CheckLoss(self.output, train_y)
        accuracy_train = self.__CalAccuracy(self.output, train_y)
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.forward(vld_x)
        loss_vld = self.loss_func.CheckLoss(self.output, vld_y)
        accuracy_vld = self.__CalAccuracy(self.output, vld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))

        need_stop = self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld)
        if loss_vld <= self.hp.eps:
            need_stop = True
        return need_stop

    def Test(self, dataReader):
        x,y = dataReader.GetTestSet()
        self.forward(x)
        correct = self.__CalAccuracy(self.output, y)
        print(correct)

    def __CalAccuracy(self, a, y):
        assert(a.shape == y.shape)
        m = a.shape[0]
        if self.hp.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a-y)**2)/m
            r2 = 1 - mse / var
            return r2
        elif self.hp.net_type == NetType.BinaryClassifier:
            b = np.round(a)
            r = (b == y)
            correct = r.sum()
            return correct/m
        elif self.hp.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = r.sum()
            return correct/m

    def SaveResult(self):
        self.wb1.SaveResultValue(self.subfolder, "wb1")
        self.wb2.SaveResultValue(self.subfolder, "wb2")

    def LoadResult(self):
        self.wb1.LoadResultValue(self.subfolder, "wb1")
        self.wb2.LoadResultValue(self.subfolder, "wb2")

    def ShowTrainingHistory(self):
        self.loss_trace.ShowLossHistory(self.hp)

    def GetTrainingHistory(self):
        return self.loss_trace

    def GetEpochNumber(self):
        return self.loss_trace.GetEpochNumber()

    def GetLatestAverageLoss(self, count=10):
        return self.loss_trace.GetLatestAverageLoss(count)

class DataReader(object):
    def __init__(self, train_file, test_file):
        self.train_file_name = train_file
        self.test_file_name = test_file
        self.num_train = 0        # num of training examples
        self.num_test = 0         # num of test examples
        self.num_validation = 0   # num of validation examples
        self.num_feature = 0      # num of features
        self.num_category = 0     # num of categories

        self.XTrain = None        # training feature set
        self.YTrain = None        # training label set
        self.XTest = None         # test feature set
        self.YTest = None         # test label set
        self.XTrainRaw = np.zeros((141,4))     # training feature set before normalization
        self.YTrainRaw = np.zeros((141,1))     # training label set before normalization
        self.XTestRaw = np.zeros((9,4))      # test feature set before normalization
        self.YTestRaw = np.zeros((9,1))      # test label set before normalization
        self.XDev = None          # validation feature set
        self.YDev = None          # validation lable set

    # read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name, allow_pickle=True)
            self.XTrainRaw[:,0] = data[:,0]
            self.XTrainRaw[:,1] = data[:,1]
            self.XTrainRaw[:,2] = data[:,2]
            self.XTrainRaw[:,3] = data[:,3]
            self.YTrainRaw[:,0] = data[:,4]

            assert(self.XTrainRaw.shape[0] == self.YTrainRaw.shape[0])
            self.num_train = self.XTrainRaw.shape[0]
            self.num_feature = self.XTrainRaw.shape[1]
            self.num_category = len(np.unique(self.YTrainRaw))
            # this is for if no normalize requirment
            self.XTrain = self.XTrainRaw
            self.YTrain = self.YTrainRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if

        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = np.load(self.test_file_name, allow_pickle=True)
            self.XTestRaw[:,0] = data[:,0]
            self.XTestRaw[:,1] = data[:,1]
            self.XTestRaw[:,2] = data[:,2]
            self.XTestRaw[:,3] = data[:,3]
            self.YTestRaw[:,0] = data[:,4]

            assert(self.XTestRaw.shape[0] == self.YTestRaw.shape[0])
            self.num_test = self.XTestRaw.shape[0]
            # this is for if no normalize requirment
            self.XTest = self.XTestRaw
            self.YTest = self.YTestRaw
            # in case there has no validation set created
            self.XDev = self.XTest
            self.YDev = self.YTest
        else:
            raise Exception("Cannot find test file!!!")
        #end if

    # merge train/test data first, normalize, then split again
    def NormalizeX(self):
        x_merge = np.vstack((self.XTrainRaw, self.XTestRaw))
        x_merge_norm = self.__NormalizeX(x_merge)
        train_count = self.XTrainRaw.shape[0]
        self.XTrain = x_merge_norm[0:train_count,:]
        self.XTest = x_merge_norm[train_count:,:]

    def __NormalizeX(self, raw_data):
        temp_X = np.zeros_like(raw_data)
        self.X_norm = np.zeros((2, self.num_feature))
        # 按行归一化,即所有样本的同一特征值分别做归一化
        for i in range(self.num_feature):
            # get one feature from all examples
            x = raw_data[:, i]
            max_value = np.max(x)
            min_value = np.min(x)
            # min value
            self.X_norm[0,i] = min_value 
            # range value
            self.X_norm[1,i] = max_value - min_value 
            x_new = (x - self.X_norm[0,i]) / self.X_norm[1,i]
            temp_X[:, i] = x_new
        # end for
        return temp_X

    def NormalizeY(self, nettype, base=0):
        if nettype == NetType.Fitting:
            y_merge = np.vstack((self.YTrainRaw, self.YTestRaw))
            y_merge_norm = self.__NormalizeY(y_merge)
            train_count = self.YTrainRaw.shape[0]
            self.YTrain = y_merge_norm[0:train_count,:]
            self.YTest = y_merge_norm[train_count:,:]                
        elif nettype == NetType.BinaryClassifier:
            self.YTrain = self.__ToZeroOne(self.YTrainRaw, base)
            self.YTest = self.__ToZeroOne(self.YTestRaw, base)
        elif nettype == NetType.MultipleClassifier:
            self.YTrain = self.__ToOneHot(self.YTrainRaw, base)
            self.YTest = self.__ToOneHot(self.YTestRaw, base)

    def __NormalizeY(self, raw_data):
        assert(raw_data.shape[1] == 1)
        self.Y_norm = np.zeros((2,1))
        max_value = np.max(raw_data)
        min_value = np.min(raw_data)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[1, 0] = max_value - min_value 
        y_new = (raw_data - min_value) / self.Y_norm[1, 0]
        return y_new

    def DeNormalizeY(self, predict_data):
        real_value = predict_data * self.Y_norm[1,0] + self.Y_norm[0,0]
        return real_value

    def __ToOneHot(self, Y, base=0):
        count = Y.shape[0]
        temp_Y = np.zeros((count, self.num_category))
        for i in range(count):
            n = (int)(Y[i,0])
            temp_Y[i,n-base] = 1
        return temp_Y

    # for binary classifier
    # if use tanh function, need to set negative_value = -1
    def __ToZeroOne(Y, positive_label=1, negative_label=0, positiva_value=1, negative_value=0):
        temp_Y = np.zeros_like(Y)
        count = Y.shape[0]
        for i in range(count):
            if Y[i,0] == negative_label:     # 负类的标签设为0
                temp_Y[i,0] = negative_value
            elif Y[i,0] == positive_label:   # 正类的标签设为1
                temp_Y[i,0] = positiva_value
            # end if
        # end for
        return temp_Y

    # normalize data by specified range and min_value
    def NormalizePredicateData(self, X_predicate):
        X_new = np.zeros(X_predicate.shape)
        n_feature = X_predicate.shape[0]
        for i in range(n_feature):
            x = X_predicate[i,:]
            X_new[i,:] = (x-self.X_norm[0,i])/self.X_norm[1,i]
        return X_new

    # need explicitly call this function to generate validation set
    def GenerateValidationSet(self, k = 10):
        self.num_validation = (int)(self.num_train / k)
        self.num_train = self.num_train - self.num_validation
        # validation set
        self.XDev = self.XTrain[0:self.num_validation]
        self.YDev = self.YTrain[0:self.num_validation]
        # train set
        self.XTrain = self.XTrain[self.num_validation:]
        self.YTrain = self.YTrain[self.num_validation:]

    def GetValidationSet(self):
        return self.XDev, self.YDev

    def GetTestSet(self):
        return self.XTest, self.YTest

    # 获得批样本数据
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    # permutation only affect along the first axis, so we need transpose the array first
    # see the comment of this class to understand the data format
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP


class HyperParameters_2_0(object):
    def __init__(self, n_input, n_hidden, n_output, 
                 eta=0.1, max_epoch=10000, batch_size=5, eps = 0.1, 
                 net_type = NetType.Fitting,
                 init_method = InitialMethod.Xavier):

        self.num_input = n_input
        self.num_hidden = n_hidden
        self.num_output = n_output

        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        self.batch_size = batch_size  

        self.net_type = net_type
        self.init_method = init_method
        self.eps = eps

    def toString(self):
        title = str.format("bz:{0},eta:{1},ne:{2}", self.batch_size, self.eta, self.num_hidden)
        return title

if __name__ == '__main__':
    dataReader = DataReader(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)

    # fig = plt.figure(figsize=(6,6))
    # DrawThreeCategoryPoints(dataReader.XTrainRaw[:,0], dataReader.XTrainRaw[:,1], dataReader.YTrain, "Source Data")
    # plt.show()

    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 16
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.01, 15, 10000
    eps = 0.001

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_2(hp, "Bank_233")

    #net.LoadResult()
    net.train(dataReader, 100, True)
    net.ShowTrainingHistory()

    # fig = plt.figure(figsize=(6,6))
    # DrawThreeCategoryPoints(dataReader.XTrain[:,0], dataReader.XTrain[:,1], dataReader.YTrain, hp.toString())
    # ShowClassificationResult25D(net, 50, hp.toString())
    # plt.show()
