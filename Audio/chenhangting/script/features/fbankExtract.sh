#!/bin/bash

PATHCAT=/mnt/c/chenhangting/Project/Multimodal/Audio/chenhangting/CV/MOSI/cats.txt
PATHWAV=/mnt/c/chenhangting/Datasets/MOSI/raw/Audio/WAV_16000/Segmented
PATHNPY=/mnt/c/chenhangting/Project/Multimodal/Audio/chenhangting/features/basicfbank
PATHEXE=/mnt/c/chenhangting/Programs/MFCC/mfcc
PATHTEMP=./temp
PATHFILELIST=./fileList.txt

mkdir -p $PATHNPY
mkdir -p $PATHTEMP
rm ${temp}/* 2>/dev/null
rm -f $PATHFILELIST && touch $PATHFILELIST
while read filename filelabel; do
#	echo -e "$filename\t$filelabel"
	filenpyname="${filename}.npy"
	basedir=$PATHNPY/${filename}
	echo -e "$PATHWAV/$filename.wav\t$PATHNPY/$filenpyname" >> $PATHFILELIST
done < $PATHCAT


echo "[Frame];
sampleRate = 16000 ;
hipassfre = 8000 ;
lowpassfre = 10 ;
preemphasise = 0.97 ;
wlen = 400 ;
inc = 160 ;
saveType = n ;
vecNum = 1 ;
fileList = $PATHFILELIST ;

[MFCC];
fbankFlag = 1 ;
bankNum = 40 ;
MFCCNum = -1 ;
MFCC0thFlag = 0 ;
	 
[Others];
energyFlag = 1 ;
zeroCrossingFlag = 1 ;
brightFlag = 1 ;
subBandEFlag = 8 ;
fftLength = 0 ;
	 
[Regression];
regreOrder = 3 ;
delwin = 9 ;" > $PATHTEMP/config.ini

$PATHEXE $PATHTEMP/config.ini > $PATHTEMP/log.log
