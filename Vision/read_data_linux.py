import os
#import random
import numpy as np
from skimage import io
from skimage import transform
import pandas as pd
from PIL import Image
    
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
                #image = Image.open(img_path)
                image = self.img_resize(img_path, img_size)
                
                #获取训练集
                if dir in x[:,0]:
                    #print("train")
                    train_label = x[x[:,0].tolist().index(dir), 1]
                    temp = {}
                    temp["image"] = image
                    temp["label"] = train_label    
                    self.data_train_set.append(temp)
                    #print(img_path + "  " + train_label)
                #获取测试集
                elif dir in y[:,0]:
                    #print("test")
                    test_label = y[y[:,0].tolist().index(dir), 1]
                    temp = {}
                    temp["image"] = image
                    temp["label"] = test_label    
                    self.data_test_set.append(temp)
                    #print(img_path + "  " + test_label)
                #print(img_path)
                #print(label)

        print("train_set:\t", int(1.0*len(self.data_train_set)))
    
    def img_resize(self, img_path, img_size):
        # resize the image to the specific size
        image = Image.open(img_path)
        image_test = image.resize((img_size, img_size), Image.ANTIALIAS)
        data = image_test.getdata()
        data = np.array(data)
        data = data/255
        data_test = np.reshape(data, (img_size, img_size, 3))
        #print("长: ",img.shape[0])
        #print("宽：",img.shape[1])
        return data_test
    
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
        
        
    def get_test_image(self, path, test_label_path, img_size):
        class_list = os.listdir(path)
        data_test_set = []
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
            img_list = os.listdir(path + '/' +dir)
            for img_name in img_list:
                img_path = path+"/"+dir+"/"+img_name
                image = self.img_resize(img_path, img_size)
                
                if dir in y[:,0]:
                    print("test")
                    test_label = y[y[:,0].tolist().index(dir), 1]
                    temp = {}
                    temp["image"] = image
                    temp["label"] = test_label    
                    data_test_set.append(temp)
                    
                    return data_test_set
    