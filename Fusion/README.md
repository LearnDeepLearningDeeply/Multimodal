
make_feature.py　　　不同模态特征的拼接和label的生成

pca_ana.m     将语音特征由256维降到100维，图像特征由300降到30维度，文本特征保持不变512维

marco_Fscore.m  由预测结果计算模型的Marco F-score

trainClassifier_pca_a3.m  svm模型的生成算法，kenel为gaussian ,kenel scale 设为 270

svm_pca_a3.m       将5折的数据分别训练对应的模型并测试，得到最终结果
