#神经网络任务一
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import LogNorm

file_name = 'C:/Users/lenovo/Desktop/mlm.npy'

class TrainingHistory_1_0(object):
    def __init__(self):
        self.iteration = []
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def AddLossHistory(self, iteration, loss, w, b):
        self.iteration.append(iteration)
        self.loss_history.append(loss)
        self.w_history.append(w)
        self.b_history.append(b)

    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.iteration, self.loss_history)
        # title = params.toString()
        # plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return 1
        # return title

    def GetLast(self):
        count = len(self.loss_history)
        return self.loss_history[count-1], self.w_history[count-1], self.b_history[count-1]
# end class

class HyperParameters(object):
    def __init__(self, input_size, output_size, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps

class DataReader(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.XTrain = np.zeros((1000,2))  # normalized x, if not normalized, same as YRaw
        self.YTrain = np.zeros((1000,1))  # normalized y, if not normalized, same as YRaw
        self.XRaw = np.zeros((1000,2))    # raw x
        self.YRaw = np.zeros((1000,1))    # raw y

    # read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name)
            self.XRaw[:,0] = data[:,0]
            self.XRaw[:,1] = data[:,1]
            self.YRaw[:,0] = data[:,2]
            self.num_train = self.XRaw.shape[0]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if

    # normalize data by extracting range from source data
    # return: X_new: normalized data with same shape
    # return: X_norm: N x 2
    #               [[min1, range1]
    #                [min2, range2]
    #                [min3, range3]]
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape) #返回值X_new是标准化后的样本，和原始数据的形状一样
        num_feature = self.XRaw.shape[1] #确定有多少列，即多少特征值
        self.X_norm = np.zeros((num_feature,2)) #特征值为行 2列
        # 按列归一化,即所有样本的同一特征值分别做归一化
        for i in range(num_feature):
            # get one feature from all examples
            col_i = self.XRaw[:,i] #把每一列分别给col i
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            # min value
            self.X_norm[i,0] = min_value 
            # range value
            self.X_norm[i,1] = max_value - min_value 
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            X_new[:,i] = new_col
        #end for
        self.XTrain = X_new

    # normalize data by self range and min_value
    def NormalizePredicateData(self, X_raw):
        X_new = np.zeros(X_raw.shape)
        n = X_raw.shape[1]
        for i in range(n):
            col_i = X_raw[:,i]
            X_new[:,i] = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
        return X_new

    def NormalizeY(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        # min value
        self.Y_norm[0, 0] = min_value 
        # range value
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new

    # get batch training data
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # get batch training data
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

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

class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.w = np.zeros((self.params.input_size, self.params.output_size))
        self.b = np.zeros((1, self.params.output_size))

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.w) + self.b
        return Z

    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB

    def __update(self, dW, dB):
        self.w = self.w - self.params.eta * dW
        self.b = self.b - self.params.eta * dB

    def inference(self, x):
        return self.__forwardBatch(x)

    def train(self, dataReader):
        # calculate loss to decide the stop condition
        loss_history = TrainingHistory_1_0()

        if self.params.batch_size == -1:
            self.params.batch_size = dataReader.num_train
        max_iteration = (int)(dataReader.num_train / self.params.batch_size)

        for epoch in range(self.params.max_epoch):
            print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                # get z from x,y
                batch_z = self.__forwardBatch(batch_x)
                # calculate gradient of w and b
                dW, dB = self.__backwardBatch(batch_x, batch_y, batch_z)
                # update w,b
                self.__update(dW, dB)
                if iteration % 2 == 0:
                    loss = self.__checkLoss(dataReader)
                    print(epoch, iteration, loss)
                    loss_history.AddLossHistory(epoch*max_iteration+iteration, loss, self.w[0,0], self.b[0,0])
                    if loss < self.params.eps:
                        break
                    #end if
                #end if
            # end for
            if loss < self.params.eps:
                break
        # end for
        loss_history.ShowLossHistory(self.params)
        print(self.w, self.b)
   
        # self.loss_contour(dataReader, loss_history, self.params.batch_size, epoch*max_iteration+iteration)
    
    def getWB(self):
        return self.w , self.b

    def __checkLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss

    def loss_contour(self, dataReader,loss_history,batch_size,iteration):
        last_loss, result_w, result_b = loss_history.GetLast()
        X,Y=dataReader.GetWholeTrainSamples()
        len1 = 50
        len2 = 50
        w = np.linspace(result_w-1,result_w+1,len1)
        b = np.linspace(result_b-1,result_b+1,len2)
        W,B = np.meshgrid(w,b)
        len = len1 * len2
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = np.dot(X, W.ravel().reshape(1,len)) + B.ravel().reshape(1,len)
        Loss1 = (Z - Y)**2
        Loss2 = Loss1.sum(axis=0,keepdims=True)/m
        Loss3 = Loss2.reshape(len1, len2)
        plt.contour(W,B,Loss3,levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)

        # show w,b trace
        w_history = loss_history.w_history
        b_history = loss_history.b_history
        plt.plot(w_history,b_history)
        plt.xlabel("w")
        plt.ylabel("b")
        title = str.format("batchsize={0}, iteration={1}, eta={2}, w={3:.3f}, b={4:.3f}", batch_size, iteration, self.params.eta, result_w, result_b)
        plt.title(title)

        plt.axis([result_w-1,result_w+1,result_b-1,result_b+1])
        plt.show()

    def ShowResult(net, dataReader):
        
        # # draw sample data
        # plt.plot(X, Y, "b.")
        # # draw predication data
        # # PX = np.linspace(0,1,5).reshape(5,1)
        # # PZ = net.inference(PX)
        # # plt.plot(PX, PZ, "r")
        # plt.title("Air Conditioner Power")
        # plt.xlabel("Number of Servers(K)")
        # plt.ylabel("Power of Air Conditioner(KW)")
        # plt.show()   

        X,Y = dataReader.GetWholeTrainSamples()
        # W,B = NeuralNet(params).getWB()
        # w1=W[1][1]
        # w2=W[1][0]
        # b=B
        # print(w1,w2,b)

        ax=plt.axes(projection="3d")
        ax.scatter3D(X[:,0],X[:,1],Y)

        x_drawing=np.linspace(0,100)
        y_drawing=np.linspace(0,100)
        X_drawing,Y_drawing=np.meshgrid(x_drawing,y_drawing)

        # ax.plot_surface(X=X[:,0] , Y=X[:,1] , Z=X*w1+Y*w2+b , color='r',alpha=0.5)

        ax.view_init(elev=30,azim=30)
        plt.show()
        
if __name__ == '__main__':
    # data
    reader = DataReader(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    params = HyperParameters(2, 1, eta=0.01, max_epoch=10, batch_size=1, eps = 1e-6)
    net = NeuralNet(params)
    net.train(reader)
    # inference
    x1 = 82.07
    x2 = 4.74
    x = np.array([x1,x2]).reshape(1,2)
    x_new = reader.NormalizePredicateData(x)

    z=net.inference(x_new)
    print("z=", z)

    Z_real = z * reader.Y_norm[0,1] + reader.Y_norm[0,0]
    print("Z_real=", Z_real)

    NeuralNet.ShowResult(net, reader)