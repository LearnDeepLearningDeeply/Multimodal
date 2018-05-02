import csv
import os
import sys

PATHROOT=r'H:\Dataset\MOUD\media\bighdd6\datasets_copy\datasets\MOUD\raw'
PATHSEGDIR=r'H:\Dataset\MOUD\media\bighdd6\datasets_copy\datasets\MOUD\raw\Video\Segments'
PATHEXE=r'C:\Users\Administrator\Downloads\ffmpeg\bin\ffmpeg.exe'

PATHCAT=os.path.join(PATHROOT,'cats.txt')
PATHTRANSCRIPTSDIR=os.path.join(PATHROOT,'Transcripts')
PATHVIDEO=os.path.join(PATHROOT,r'Video\Full')


filedict={}
with open(PATHCAT,'r',newline='') as f:
    reader=csv.reader(f,delimiter='\t')
    for row in reader:
        if(row[1]=='neutral'):pass
        else:
            name=row[0].rsplit('_',maxsplit=1)[0]
            inx=int(row[0].rsplit('_',maxsplit=1)[1])
            if(name in filedict):filedict[name].append(inx)
            else:filedict[name]=[inx,]

n=0
for fname in filedict:
    fpath=os.path.join(PATHTRANSCRIPTSDIR,fname+'.csv')
    with open(fpath,'r',newline='') as f:
        reader=csv.DictReader(f,delimiter=r';')
        for inx,row in enumerate(reader):
            if(inx+1 in filedict[fname]):
                n+=1
                fromvideoname=os.path.join(PATHVIDEO,fname+'.mp4')
                starttime=row['#starttime']
                endtime=row['#endtime']
                segname=os.path.join(PATHSEGDIR,fname+'_'+str(inx+1)+'.mp4')
                print(PATHEXE+'\t-ss\t'+starttime+'\t-i\t'+fromvideoname+'\t-c copy\t-to\t'+endtime+'\t'+segname)
                os.system(PATHEXE+'\t-ss\t'+starttime+'\t-i\t'+fromvideoname+'\t-c copy\t-to\t'+endtime+'\t'+segname)


print('total process %d videos'%(n,))