#!/bin/bash

PATHPY=./baselinelstm_early_stopping.py
PATHLOGDIR=../../log/baselinelstm
PATHMODELDIR=../../temp
startcard=0
startfold=1
foldnum=5
batch_size=16

mkdir -p $PATHLOGDIR
mkdir -p $PATHMODELDIR

for i in $(seq 0 `expr $foldnum - 1`);do
    card=`expr $startcard + $i`
    fold=`expr $startfold + $i`
    nohup python -u $PATHPY --cvnum $fold --batch_size $batch_size --device_id $card --savepath $PATHMODELDIR/model${fold}.pkl > $PATHLOGDIR/log${fold}.log 2>&1 &
done
