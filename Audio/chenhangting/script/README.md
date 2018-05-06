# 音频实验结果

## 流程

1. 将原来的训练集`Multimodal\CV\MOSI\CV\fold{i}_train.txt`,根据各标签的比例划分成`Multimodal\Audio\chenhangting\script\CV\folds\fold{i}_train.txt`和`Multimodal\Audio\chenhangting\script\CV\folds\fold{i}_eva.txt`,前者用来训练，后者用来early stopping
2. 根据`Multimodal\Audio\chenhangting\script\CV\folds\foldsfold{i}_train.txt`训练
3. 根据`Multimodal\Audio\chenhangting\script\CV\folds\foldsfold{i}_eva.txt`调节超参
4. 根据`Multimodal\Audio\chenhangting\script\CV\folds\foldsfold{i}_test.txt`得到测试结果

## 特征

1. 25ms帧长，10ms帧移
2. 40个mel滤波器
3. 加入能量、谱重心、过零率、子带能量（8个子带），共51维
4. 加入delta和delta-delta，共51*3=153维

## 神经网络

1. DNN,4*256ReLu,
2. LSTM,4*128 BiLstm_cell,
3. LSTM+attention,4*128 BiLstm_cell,
4. LSTM+CNN,2层COV，2层128 BiLstm_cell,


## 实验结果

| Net | Feature | Macro F-score |
| :- | - | -: |
| dnn | fbank+others | 0.546 |
| lstm | fbank+others | 0.548 |
| lstm+attention | fbank+others | 0.548 |
| lstm+cnn | fbank+others | 0.542 |
| SVM???[1] | OpenSmile HSF | 0.571 |




## Ref.

[1] Cambria, E., et al. (2017). "Benchmarking Multimodal Sentiment Analysis."

[2] Poria, S., et al. (2017). Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis. IEEE International Conference on Data Mining.
	
