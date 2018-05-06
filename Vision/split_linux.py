# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:54:51 2018

@author: pengshuo
"""

import os
c = 'abdce'
f = 'aaaa'
#print(r'%s\%s' %(c,f))


segmented1 = r"D:\deep_learning_homework\dataset\MOSI\raw\Video\Segmented"
#segmented = r"/home/chenhangting/pengshuo/MOSI/raw/Video/Segmented"


for root, dirs, files in os.walk(segmented1):
    for file in files:
        '''
        input = os.path.join(root,file)
       
        output = r'/home/chenhangting/pengshuo/frames_test'
        print(os.walk(output))
        if file in os.walk(output):
            os.system(r'ffmpeg -i %s -r 1 -f image2 /home/chenhangting/pengshuo/frames_test/%s/image-%3d.jpeg'%(input,file))
        else:
        '''
        in_file = os.path.join(root,file)
        #print(os.path.splitext(file)[0])
        #print(r'%s\%s' %(output,file))
        
        os.mkdir(r'/home/chenhangting/pengshuo/frames_test/%s' %os.path.splitext(file)[0])
        os.system(r'/home/chenhangting/pengshuo/FFmpeg/bin/ffmpeg -i %s -r 1 -f image2 /home/chenhangting/pengshuo/frames_test/%s/image-%%3d.jpeg'%(in_file,os.path.splitext(file)[0]))
        
        #print(input)
        #print(file)
 
