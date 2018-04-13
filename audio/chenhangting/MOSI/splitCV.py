import csv
import sys
import os
from sklearn.model_selection import GroupKFold

PATHSPEAKER=r'./speaker.txt'
PATHCAT=r'./cats.txt'
PATHCVDIR=r'./CV'
nsplits=5

def findRepeat(speaker,speakerRepeat):
    for r in speakerRepeat:
        if(speaker in r):return r[0]
    return speaker

def writeCSV(samples,labels,inx,PATHCSV):
    temp=[]
    for i in inx:temp.append((samples[i],labels[i]))
    with open(PATHCSV,'w',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerows(temp)

speakerRepeat=[]
with open(PATHSPEAKER,'r',newline='') as f:
    reader=csv.reader(f,delimiter=' ')
    for row in reader:speakerRepeat.append(row)
#print(speakerRepeat)

filedict={}
with open(PATHCAT,'r',newline='') as f:
    reader=csv.reader(f,delimiter='\t')
    for row in reader:
        if(len(row)!=2):
            print(row)
            sys.exit("Wring input row")
        elif(row[1]=='neutral'):pass
        else:filedict[row[0]]=row[1]

filedict=filedict.items()
groups=[];samples=[];labels=[]
for row in filedict:
    samples.append(row[0])
    labels.append(row[1])
    speaker=row[0].split('_')[0]
    groups.append(findRepeat(speaker,speakerRepeat))
#print(groups)

gkf=GroupKFold(n_splits=nsplits)
for fold,(train,test) in enumerate(gkf.split(samples,labels,groups=groups)):
    PATHCSV=os.path.join(PATHCVDIR,'fold{}_{}.txt'.format(fold+1,'train'))
    writeCSV(samples,labels,train,PATHCSV)
    PATHCSV=os.path.join(PATHCVDIR,'fold{}_{}.txt'.format(fold+1,'test'))
    writeCSV(samples,labels,test,PATHCSV)