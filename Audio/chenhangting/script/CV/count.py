import os
import csv
import sys

PATHCVdir=r'H:\Multimodal\Audio\chenhangting\script\CV\folds'

labels=('positive','negative')
procedures=('train','eva','test')

for i in range(5):
    fold=i+1
    for p in procedures:
        PATHCVTXT=os.path.join(PATHCVdir,'fold{}_{}.txt'.format(fold,p))
        with open(PATHCVTXT,'r',newline='') as f:
            filedictcount={}
            for l in labels:filedictcount[l]=0
            reader=csv.reader(f,delimiter='\t')
            for row in reader:
                if(len(row)!=2):sys.exit("wrong")
                filedictcount[row[1]]+=1
            filedictweight={}
            for l in labels:filedictweight[l]=0
            count=0
            for k in filedictcount:
                count+=filedictcount[k]
            for k in filedictweight:
                filedictweight[k]=float(filedictcount[k])/float(count)
            print(PATHCVTXT)
            print(filedictweight)
            print(filedictcount)