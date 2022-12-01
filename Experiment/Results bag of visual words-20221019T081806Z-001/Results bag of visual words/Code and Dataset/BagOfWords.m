% function BagOfWords()
% So lan lap
k=1;         
% Do chinh xac trung binh cua tap trainingSets
Average_Accuracy_train=0; 
% Do chinh xac trung binh cua tap testingSets
Average_Accuracy_test=0;        
tMatrix=zeros(2);         % Ma tran nham lan trainingSets
vMatrix=zeros(2);         % Ma tra nham lan testingSets

A = zeros(k,6);
for i=1:k
%% Khoi tao data set
%dataFolder = fullfile('E:\BTL Y sinh\BTL XLAYT', 'data');
%%%%%%%%%%%%
% imgSets = [ imageSet(fullfile('GRAYoriginal','them' ,'Abnormal'))
%             imageSet(fullfile('GRAYoriginal', 'them' ,'Normal'))];
%% Chia data anh thanh cac tap bang nhau
% DataSet
% Xac dinh so luong anh it nhat trong 2 tap Abnormal va Normal
% minSetCount = min([imgSets.Count]); 
% Can bang so luong anh moi loai
% imgSets = partition(imgSets, minSetCount, 'randomize');
% % Hien thi so luong anh moi tap 
% [imgSets.Count]
%% Chuan bi 70% data anh de dao tao va 30% de kiem dinh
% [trainingSets, testingSets] = partition(imgSets, 0.7, 'randomize');
 trainingSets= [imageSet(fullfile('GRAYoriginal','them' ,'Train','Abnormal'))
                imageSet(fullfile('GRAYoriginal','them' ,'Train','Normal')) ];
 testingSets = [imageSet(fullfile('GRAYoriginal','them' ,'Test','Abnormal'))
                imageSet(fullfile('GRAYoriginal','them' ,'Test','Normal')) ];
%% Tao tui tinh nang
 
bag = bagOfFeatures(trainingSets,'VocabularySize',600,'StrongestFeatures',0.2,'Verbose',false,'PointSelection','Detector');
 
%% Dao tao bo phan loai - Classifier
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);
 
%% Dung Classifier phan loai trainingSets
trainMatrix = evaluate(categoryClassifier, trainingSets);
 
%% Tinh do chinh xac trung binh khi phan loai trainingSets

Average_Accuracy_train=mean(diag(trainMatrix));
tMatrix=tMatrix+trainMatrix;
sensiTrain = tMatrix(1,1)/( tMatrix(1,1)+ tMatrix(1,2));
specTrain = tMatrix(2,2)/( tMatrix(2,2)+ tMatrix(2,1));
%% Dung Classifier phan loai testingSets
testMatrix = evaluate(categoryClassifier, testingSets);
 
%% Tinh do chinh xac trung binh khi phan loai testingSets
Average_Accuracy_test=mean(diag(testMatrix));
vMatrix=vMatrix+testMatrix;
sensiTest = vMatrix(1,1)/( vMatrix(1,1)+ vMatrix(1,2));
specTest = vMatrix(2,2)/( vMatrix(2,2)+ vMatrix(2,1)); 

A (i,:)= [Average_Accuracy_test, Average_Accuracy_train, sensiTest, sensiTrain, specTest,specTrain];
end
% Tinh do chinh xac trung binh cua cac tap
% Average_Accuracy_train=Average_Accuracy_train/k; 
% Average_Accuracy_test=Average_Accuracy_test/k;
% % Tinh ma tran nham lan trung binh cua cac tap
% tMatrix=tMatrix/k;
% vMatrix=vMatrix/k;
% % Luu ket qua vao file mat
% save('Ketqua_0.7_1_9.mat','tMatrix');
% save('Ketqua_0.7_1_9.mat','vMatrix','-append');
% save('Ketqua_0.7_1_9.mat','Average_Accuracy_train','-append');
% save('Ketqua_0.7_1_9.mat','Average_Accuracy_test','-append');

% save('KetquabalanceRGBorigin_0.9_max_9.mat','A');
