#coding:utf-8
'''
@author : chenhangting
@date : 2018/4/25
@note : a script to count log files
'''

import numpy as np
import sys

num_folds=5
PATHlogroot=sys.argv[1]

acc=[];fscore=[]
lastline=''
for i in range(num_folds):
    fold=i+1
    PATHlog="{}{}.log".format(PATHlogroot,fold)
    with open(PATHlog,'r') as f:
        temp_acc='';temp_fscore=''
        for line in f:
            if(line[0:8]=='Test set'):temp_acc=float(line.split(" (")[1].split("%)")[0])
            elif(line[0:5]=='macro'):temp_fscore=float(line.strip()[-8:])
            lastline=line
        if(lastline[0:5]!='macro' and lastline!='\n'):sys.exit("Unvalid file %s\n%s"%(PATHlog,lastline,))
        acc.append(temp_acc)
        fscore.append(temp_fscore)

acc=np.array(acc)
fscore=np.array(fscore)
if(len(acc)==num_folds and len(fscore)==num_folds):
    print("acc:");print(acc);print("fscore");print(fscore)
    print("Average acc {:.04f}-+{:.04f}".format(np.mean(acc),np.std(acc,ddof=1)))
    print("Average marco f-score {:.04f}-+{:.04f}".format(np.mean(fscore),np.std(fscore,ddof=1)))
else:
    print(acc)
    print(fscore)
