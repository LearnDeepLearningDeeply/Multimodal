function F=macro_Fscore(ylabel,yresult)

if size(ylabel)~=size(yresult)
    F=0;
    return ;
end
TP=zeros(2);
FP=zeros(2);
FN=zeros(2);
F_score=zeros(2);
P=zeros(2);
R=zeros(2);
for i=1:size(ylabel)
    if ylabel(i)==1
       if yresult(i)==1
           TP(1)=TP(1)+1;
       else
           FN(1)=FN(1)+1;
           FP(2)=FP(2)+1;
       end
    else
         if yresult(i)==-1
           TP(2)=TP(2)+1;
       else
           FN(2)=FN(2)+1;
           FP(1)=FP(1)+1;
         end
    end
end
P(1)=TP(1)/(TP(1)+FP(1));
R(1)=TP(1)/(TP(1)+FN(1));
F_score(1)=2*P(1)*R(1)/(R(1)+P(1));

P(2)=TP(2)/(TP(2)+FP(2));
R(2)=TP(2)/(TP(2)+FN(2));
F_score(2)=2*P(2)*R(2)/(R(2)+P(2));

F=(F_score(1)+F_score(1))/2;


    