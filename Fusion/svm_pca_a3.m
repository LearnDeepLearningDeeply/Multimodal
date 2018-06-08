F_score=zeros(5,1);
data={'fold1_train','fold2_train','fold3_train','fold4_train','fold5_train'};
test_data={'fold1_test','fold2_test','fold3_test','fold4_test','fold5_test'};


for i=1:5
load(char(data(i)));
x2=pca_ana(char(data(i)),30,100);
svm_data=[x2(:,1:30+512+100),y];
[svm_model,bc]=trainClassifier_pca_a3(svm_data);
load(char(test_data(i)));
x3=pca_ana(char(test_data(i)),30,100);
yfit=svm_model.predictFcn(x3(:,1:30+512+100));
F_score(i)=macro_Fscore(y,yfit);
end
mean(F_score)
