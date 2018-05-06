from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import read_data_linux
import numpy as np

batch_size = 50# 每次训练和梯度更新块的大小。
nb_classes = 2 # 共有两种类型，积极，消极
nb_epoch = 3 # 迭代次数。

nb_filters = 32
nb_pool = 2
nb_conv = 3

path = r"/home/chenhangting/pengshuo/faces_test"
train_label_path = r"/home/chenhangting/pengshuo/CV/fold1_train.txt"
test_label_path = r"/home/chenhangting/pengshuo/CV/fold1_test.txt"
scene_data = read_data_linux.scene_data(path, train_label_path, test_label_path, 100)
data_train_set = scene_data.data_train_set
data_test_set = scene_data.data_test_set



X_train = []
y_train = []
X_test = []
y_test = []



#对数据进行归一化到0-1 因为图像数据最大是15
#X_train /= 15
#X_test /= 15

img_rows = 100
img_cols = 100

#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')


model = Sequential()#建立模型

#第一个卷积层，2个卷积核，每个卷积核大小3*3。1表示输入的图片的通道,灰度图为1通道。
#border_mode可以是valid或者full
#激活函数用relu
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,3)))
convout1 = Activation('relu')
model.add(convout1)

#第二个卷积层，32个卷积核，每个卷积核大小3*3。
#激活函数用relu
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)

#第三个卷积层，32个卷积核，每个卷积核大小3*3。
#激活函数用relu
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout3 = Activation('relu')
model.add(convout3)

#采用maxpooling，poolsize为(2,2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

#按概率来将x中的一些元素值置零，并将其他的值放大。
#用于进行dropout操作，一定程度上可以防止过拟合 
#x是一个张量，而keep_prob是一个[0,1]之间的值。
#x中的各个元素清零的概率互相独立，为1-keep_prob,
#而没有清零的元素，则会统一乘以1/keep_prob, 
#目的是为了保持x的整体期望值不变。
model.add(Dropout(0.5))

#全连接层，先将前一层输出的二维特征图flatten为一维的,压扁平准备全连接。
model.add(Flatten())
model.add(Dense(512))#添加512节点的全连接
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))#添加输出3个节点
model.add(Activation('softmax'))#采用softmax激活
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#用于训练一个固定迭代次数的模型
#返回：记录字典，包括每一次迭代的训练误差率和验证误差率；
#X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train, y_train = scene_data.get_next_batch(scene_data.data_train_set, len(data_train_set))
Y_train = np_utils.to_categorical(y_train, nb_classes)


hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1)

model.save('my_model.h5')

#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#              shuffle=True, verbose=1, validation_split=0.2)


#展示模型在验证数据上的效果
#返回：误差率或者是(误差率，准确率)元组（if show_accuracy=True）

model = load_model('my_model.h5')

score = model.evaluate(X_train, Y_train,  verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from sklearn.metrics import classification_report

Y_pred = model.predict(X_train)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
 

target_names = ['class 0(Negative)', 'class 1(Positive)']
print(classification_report(np.argmax(Y_train,axis=1), y_pred,target_names=target_names))