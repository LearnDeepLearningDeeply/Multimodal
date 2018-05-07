split_linux.py 用于把视频分帧，这里利用ffmpeg 1s提取一帧。    
face_linux.py 用于把图像中的人脸提取出来，这里使用了opencv库。    
read_data_linux.py 读取数据，并分为测试集和训练集。    
train_network_linux.py 搭建神经网络，对fold1-5分别进行训练和测试，并输出准确率、召回率和F-Score。搭建神经网络用到了keras和tensorflow。    
