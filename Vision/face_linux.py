# -*- coding: utf-8 -*-

#import numpy
import cv2
import os


face_cascade = cv2.CascadeClassifier(r'/home/chenhangting/pengshuo/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier(r'D:\opencv-source\data\haarcascades\haarcascade_eye.xml')

segmented = r"/home/chenhangting/pengshuo/frames_test"
output = r"/home/chenhangting/pengshuo/faces_test"
i = 1

for root, dirs, files in os.walk(segmented):
     for file in files:
         input = os.path.join(root,file)
         print(root.split('/')[5])
         print(os.path.exists(r'/home/chenhangting/pengshuo/faces_test/%s' %root.split('/')[5]));

         if(os.path.exists(r'/home/chenhangting/pengshuo/faces_test/%s' %root.split('/')[5]) == True):
             img = cv2.imread(input)
             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
             for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                face = img[y:y+h,x:x+w]
                cv2.imwrite(r"/home/chenhangting/pengshuo/faces_test/%s/face_%d.jpg"%(root.split('/')[5],i), face)
                i = i + 1
         else:
             os.mkdir(r'/home/chenhangting/pengshuo/faces_test/%s' %root.split('/')[5]) 
             
             img = cv2.imread(input)
             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
             for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
                face = img[y:y+h,x:x+w]
                cv2.imwrite(r"/home/chenhangting/pengshuo/faces_test/%s/face_%d.jpg"%(root.split('/')[5],i), face)
                i = i + 1
     i = 1

#-----------------------------------------
# up half of the face is set to find eyes!
#-----------------------------------------
'''
roi_gray = gray[y:y+h/2, x:x+w]
roi_color = img[y:y+h/2, x:x+w]

eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
'''
print("你好")
#cv2.imshow('img',img)
print("hhhh")
#cv2.imwrite("face_detected_1.jpg", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
