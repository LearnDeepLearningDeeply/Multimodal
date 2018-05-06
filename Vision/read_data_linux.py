import os
#import random
import numpy as np
from skimage import io
from skimage import transform
import pandas as pd
    
class scene_data(object):
    data_train_set = []
    data_test_set = []
    image_size = 0
    
    def __init__(self, path, train_label_path, test_label_path, img_size=128):
        self.image_size = img_size
    
        class_list = os.listdir(path)
        print("all class and label:")
        
        #获取训练集标签
        train_file_label = open(train_label_path)
        lines_train = train_file_label.readlines()
        aa = []
        for line in lines_train:
            temp = line.replace('\n','').split('\t')
            aa.append(temp)
        x = np.array(aa)
        print(x[:,1])
        
        #获取测试集标签
        test_file_label = open(test_label_path)
        lines_test = test_file_label.readlines()
        bb = []
        for line in lines_test:
            temp = line.replace('\n','').split('\t')
            bb.append(temp)
        y = np.array(bb)
        print(y[:,0])
        print(y[:,1])
        
        for dir in class_list:
            img_list = os.listdir(path+"/"+dir)
            #print(path+"\\"+dir)            
            for img_name in img_list:
                img_path = path+"/"+dir+"/"+img_name
                image = io.imread(img_path)
                image = self.img_resize(image, img_size)
                #print(img_path)
                #获取训练集
                if dir in x[:,0]:
                    print("train")
                    train_label = x[x[:,0].tolist().index(dir), 1]
                    temp = {}
                    temp["image"] = image
                    temp["label"] = train_label    
                    self.data_train_set.append(temp)
                    print(img_path + "  " + train_label)
                #获取测试集
                elif dir in y[:,0]:
                    print("test")
                    test_label = y[y[:,0].tolist().index(dir), 1]
                    temp = {}
                    temp["image"] = image
                    temp["label"] = test_label    
                    self.data_test_set.append(temp)
                    print(img_path + "  " + test_label)
                #print(img_path)
                #print(label)

        print("train_set:\t", int(1.0*len(self.data_train_set)))
    
    def img_resize(self, img, img_size):
        # resize the image to the specific size
        if (img.shape[1] > img.shape[0]):
            scale = float(img_size) / float(img.shape[0])
            img = np.array(transform.resize(np.array(img), (
            int(img.shape[1] * scale + 1), img_size))).astype(np.float32)
        else:
            scale = float(img_size) / float(img.shape[1])
            img = np.array(transform.resize(np.array(img), (
            img_size, int(img.shape[0] * scale + 1)))).astype(np.float32)
        # crop the proper size and scale to [-1, 1]
        img = (img[
                  (img.shape[0] - img_size) // 2:
                  (img.shape[0] - img_size) // 2 + img_size,
                  (img.shape[1] - img_size) // 2:
                  (img.shape[1] - img_size) // 2 + img_size,
                  :]-127)/255
        #print("长: ",img.shape[0])
        #print("宽：",img.shape[1])
        return img
    
    def get_next_batch(self, data_set, batch_size):
        # random.shuffle(data_set)
    
        images = []
        labels = []
        for i in range(batch_size):
            images.append(data_set[i]["image"])
            labels.append(data_set[i]["label"])
    
        image_batch = np.reshape(images, (batch_size, self.image_size, self.image_size, 3))
        label_batch = np.reshape(labels, (batch_size))
    
        return image_batch, label_batch
    
    
if __name__ == "__main__":
    path = r"D:\deep_learning_homework\dataset\MOSI\raw\Video\faces_test"
    train_label_path = r"D:\deep_learning_homework\CV\fold1_train.txt"
    test_label_path = r"D:\deep_learning_homework\CV\fold1_test.txt"
    train_data_set, test_data_set = scene_data(path, train_label_path, test_label_path, 128)
    
    '''
    file = open(train_label_path)
    lines = file.readlines()
    aa = []
    for line in lines:
        temp = line.replace('\n','').split('\t')
        aa.append(temp)
    x = np.array(aa)
    y = 'G-xst2euQUc_19'
    if y in x[:,0]:
        print(x[x[:,0].tolist().index(y), 1])
    '''    
    
    
    
    
    #print(label_all[[0,1]])
    #x = label_all[[0]].values.tolist()
    #y = '[TvyZBvOMOTc_1]'
    #q = [1,2,3]
    #r = 1
    #if r in q:
    #    print('true')
    #else:
    #    print('f')
    #df = pd.DataFrame({'a':[1,3,5,7,4,5,6,4,7,8,9],'b':[3,5,6,2,4,6,7,8,7,8,9]})
    #print(x[2])
    #x, y = data_set.get_next_batch(data_set.train_set, 32)
    #print(x.shape, y.shape)
    #print(y)
    # for i in range(10):
    #     image = data_set.train_set[i]["image"]
    #     label = data_set.train_set[i]["label"]
    #     print(label)
    #     print(image)
    #     cv2.imshow("img", image)
    #     cv2.waitKey()
