function result_data = pca_ana(source_data,n1,n3)
load(source_data);
   feature=x(:,1:300);
  [pc,score,latent,tsquare] = pca(feature);%我们这里需要他的pc和latent值做分析
   feature1=score(:,1:n1);
 %feature1=feature;
  feature=x(:,301:812);
%  [pc,score] = pca(feature);%我们这里需要他的pc和latent值做分析
% feature2=score(:,1:426); 
 feature2=feature;
 feature=x(:,813:1068);
[pc,score,latent,tsquare] = pca(feature);%我们这里需要他的pc和latent值做分析
feature3=score(:,1:n3);

[a,~]=size(x);
clear x
x=zeros(a,n1+512+n3);
x(:,1:n1)=feature1;
x(:,n1+1:n1+512)=feature2;
x(:,n1+513:n1+512+n3)=feature3;
result_data=x;
%cumsum(latent)./sum(latent)

% tran=pc(:,1:50);
% feature= bsxfun(@minus,feature,mean(feature,1));
% feature_after_PCA= feature*tran;
% feature_after_PCA=score(:,1:50);