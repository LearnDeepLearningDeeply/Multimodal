# 交叉验证流程

## folds

1. 交叉测试根据说话人划分为5组，
2. folds里面包含5组文件。fold{i}\_train 是 第{i}组交叉测试的训练集，fold{i}\_test 是 第{i}组交叉测试的测试集。 整个系统根据fold{i}\_train 训练，根据fold{i}\_test 测试。
3. 实验数据仅包含2类情感，`positive negative`
4. 本交叉验证集的设定参考 Cambria, E., et al. (2017). "Benchmarking Multimodal Sentiment Analysis."

| samples num in train/test | positive | negtive |
| - | :-: | -: | 
| fold1 |  842/238 | 843/180 | 
| fold2 |  879/201 | 798/225 |
| fold3 |  888/192 | 798/225 |
| fold4 |  878/202 | 808/215 |
| fold5 |  833/247 | 845/178 |

## cats.txt

所有样本的标签

# speaker.txt

同一说话人标记

# MTurkAve.py

统计5人打分，给出正负标签。生成cats.txt

# splitCV.py

用sklearn生成5折交叉测试文件
