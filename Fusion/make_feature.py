import sys
import operator
import numpy as np
import os

list_file = open('lists/fold5.train.list','r')
video_feats = np.load("video-feature-npy-new/fold5_train_new.npy")
text_feats = np.load("data/textFeat/fold5.train.textFeat.npy")
search_file='data/textFeat/fold5_train.txt'
audio_feats_path='data/audio_fusion_feature/audio_fusion_feature/fold5/'
result_feature='data/total_feature/fold5_train_total.npy'
result_label='data/label_file/fold5_train_label.npy'
[len_1,row_1]=np.shape(video_feats)
[len_2,row_2]=np.shape(text_feats)
row_3=257       #256 for feature 1 for result
feature_lists =  np.zeros( (len_1,row_1+row_2-1+row_3-1) )        #定义新的特征数组
label_file=np.zeros(len_1)
'''
if len_1!=len_2:
   print("len is not compatiable")
   print("len_1 is %d len_2 is %d" %(len_1,len_2))
   sys.exit(0)
else :
   print ("the len is %d"%len_1)
'''

if row_1!=301:
   print("row_1 is %d"%row_1)
   sys.exit(0)

if row_2!=512:
   print("row_2 is %d"%row_1)
   sys.exit(0)



n1=0
n2=0

feature_lists[:,0:300]=video_feats[:,0:300]        # video_feats

for (num,line) in enumerate(list_file):
   line=line.strip('\n')
   file = open(search_file,'r')
   for (position,value) in enumerate(file):
      (tmp1,tmp2)=value.split("\t",1)
      tmp2=tmp2.strip('\n')
	  
      if operator.eq(line,tmp1) :
       n1+=1	  
       feature_lists[num,300:812]=text_feats[position,:]            # text_feats
       if operator.eq(tmp2,'positive'):
          label_file[num]=1   
       elif operator.eq(tmp2,'negative'):
          label_file[num]=0
       else:
          print("label make failed")
          sys.exit()
       break
      
       
   file.close()
 
   if os.path.isfile(audio_feats_path+line+'.npy'):           # text_feats
     audio_feats = np.load(audio_feats_path+line+'.npy')
     feature_lists[num,812:]=audio_feats[0:256]	 
     n2+=1	 
   else :
     print("search for",line,"failed")
     sys.exit(0)
	 
	 	 
   
   
   
     
list_file.close()   
print("the search num is ",n1,n2) 


[n1,n2]=np.shape(feature_lists)
print ("the dimntion is",n1,n2)
np.save(result_feature,feature_lists)
np.save(result_label,label_file)