# 交叉验证流程

## folds

1. 交叉测试根据说话人划分为5组，
2. folds里面包含5组文件。fold{i}\_train 是 第{i}组交叉测试的训练集，fold{i}\_test 是 第{i}组交叉测试的测试集。 整个系统根据fold{i}\_train 训练，根据fold{i}\_test 测试。
3. 实验数据仅包含4类情感，`positive negative`
4. 本交叉验证集的设定参考 Cambria, E., et al. (2017). "Benchmarking Multimodal Sentiment Analysis."

| samples num in train/test | positive | negtive |
| - | :-: | -: | 
| fold1 |  843/239 | 843/180 |
| fold2 |  880/202 | 798/225 |
| fold3 |  889/193 | 798/225 |
| fold4 |  879/203 | 808/215 |
| fold5 |  834/248 | 845/178 |

## cats.txt

产生folds中的文件，不用关心
