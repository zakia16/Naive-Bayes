train_data = dlmread('E:\DataForNaiveBayes\trainingData.txt');
train_label = dlmread('E:\DataForNaiveBayes\trainingLabels.txt');
test_data = dlmread('E:\DataForNaiveBayes\testData.txt');
test_label= dlmread('E:\DataForNaiveBayes\testLabels.txt');

c = NaiveBayes.fit(train_data,train_label,'dist',...
    {'normal','kernel','normal','kernel'})
l= c.predict(test_data)
cm= confusionmat(test_label,l)

precision=[];
s=0;c=1;
for i=1:3
    for j=1:3
        s=s+cm(i,j);
        if(i==j)
            p=cm(i,j);
        end
    end
    precision(c,1)= p/s;
    s=0;p=0;
    c=c+1;
end

recall=[];
s=0;c=1;
for i=1:3
    for j=1:3
        s=s+cm(j,i);
        if(i==j)
            p=cm(i,j);
        end
    end
    recall(c,1)= p/s;
    s=0;p=0;
    c=c+1;
end

macroaverage_precision= sum(precision)/3
macroaverage_recall= sum(recall)/3
%microaverage_precision= sum(diag(cm))/sum(sum(cm))
%microaverage_recall= sum(diag(cm))/sum(sum(cm))

F_measure= 2*(macroaverage_precision*macroaverage_recall)/(macroaverage_precision+macroaverage_recall)







