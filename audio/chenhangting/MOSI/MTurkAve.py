import csv
import sys

PATH1=r'H:\Dataset\MOSI\MOSI\labels\MTurkAnnotations.csv'
PATH2=r'H:\cv\cats.txt'

filedict={}
with open(PATH1,'r',newline='') as f1:
    reader=csv.reader(f1)
    for row in reader:
        if(len(row)!=6):sys.exit("row split wrong occurs")
        sumScore=0.0;name=''
        for inx,s in enumerate(row):
            if(inx==0):name=s
            else:
                if(s=='ambiguous'):pass
                else:sumScore+=float(s)
        sumScore=sumScore/5.0
        if(sumScore>0):sumScore='positive'
        elif(sumScore<0):sumScore='negative'
        else:sumScore='neutral'
        filedict[name]=sumScore

with open(PATH2,'w',newline='') as f2:
    writer=csv.writer(f2,delimiter='\t')
    filedict=filedict.items()
    writer.writerows(filedict)