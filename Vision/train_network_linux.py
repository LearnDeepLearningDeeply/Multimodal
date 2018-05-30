from keras.models import Sequential, Model,load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import optimizers
import keras

import read_data_linux
import numpy as np



batch_size = 200
nb_classes = 2
nb_epoch = 15

nb_filters = 4
nb_pool = 2
nb_conv1_1 = 4
nb_conv1_2 = 4
nb_conv2_1 = 3
nb_conv2_2 = 3

nb_conv3_1 = 3
nb_conv3_2 = 3

img_rows = 100
img_cols = 100




for i in range(1,6):
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv1_1, nb_conv1_2,
                        border_mode='valid',
                        input_shape=(img_rows, img_cols,3)))
    convout1 = Activation('relu')
    model.add(convout1)
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, nb_conv2_1, nb_conv2_2))
    convout2 = Activation('relu')
    model.add(convout2)
    #model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, nb_conv3_1, nb_conv3_2))
    convout3 = Activation('relu')
    model.add(convout3)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512,activation='relu',name="Dense_1"))
    #model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    print("in the %i"%i)
    
    path = r"/home/chenhangting/pengshuo/faces_test"
    train_label_path = r"/home/chenhangting/pengshuo/CV/fold%i_train.txt"%i
    test_label_path = r"/home/chenhangting/pengshuo/CV/fold%i_test.txt"%i
    scene_data = read_data_linux.scene_data(path, train_label_path, test_label_path, img_rows, img_cols)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    Y_test = []
    Y_train = []
    Y_train_new = []
    train_names = []
    test_names = []
    Y_pred = []
    train_names, X_train, y_train = scene_data.get_next_batch(scene_data.data_train_set, len(scene_data.data_train_set))
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    #np.savetxt('result/path_train_%i.txt'%i, m, fmt = '%s')
       
    test_names, X_test, y_test = scene_data.get_next_batch(scene_data.data_test_set, len(scene_data.data_test_set))
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    #np.savetxt('result/path_test_%i.txt'%i, n, fmt = '%s')
    
    hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=0)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])         
    
    #保存训练特征，其中Y_train_new是图像特征，m是视频及图像名称
    m = np.array(train_names)
    Y_train_new = model.predict(X_train)
    m_new = np.column_stack((m,Y_train_new))
    np.savetxt('result1/result_train_%i.txt'%i, m_new, fmt = '%s')
    
    #保存测试特征，Y_pred是图像特征，n是视频及图像名称
    n = np.array(test_names)
    Y_pred = model.predict(X_test)   
    n_new = np.column_stack((n,Y_pred))
    np.savetxt('result1/result_test_%i.txt'%i, n_new, fmt = '%s')
    
    y_pred = np.argmax(Y_pred, axis=1)
    #np.savetxt('result1/result_%i.txt'%i, y_pred, fmt = '%s') 
    
    from sklearn.metrics import classification_report
    target_names = ['class 0(Negative)', 'class 1(Positive)']
    print(classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names))