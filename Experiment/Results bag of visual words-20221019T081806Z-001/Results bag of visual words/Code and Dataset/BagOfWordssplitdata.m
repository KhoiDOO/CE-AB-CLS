% function BagOfWordssplit()
% So lan lap
k=10;
% Do chinh xac trung binh cua tap trainingSets
Average_Accuracy_train=0;
% Do chinh xac trung binh cua tap testingSets
Average_Accuracy_test=0;
tMatrix=zeros(2);         % Ma tran nham lan trainingSets
vMatrix=zeros(2);         % Ma tra nham lan testingSets
A = zeros(k,6);
% Acc_train = [];
% Acc_test = [];

%figure(1);
for i=1:k
    %% Khoi tao data set
    %dataFolder = fullfile('E:\BTL Y sinh\BTL XLAYT', 'data');
    imgSetsNorm = imageSet(fullfile('them', 'Normal')) ;
    imgSetsAbnorm = imageSet(fullfile('them', 'Abnormal'));
    
    imgSets = [imgSetsNorm, imgSetsAbnorm] ;
    %% Chia data anh thanh cac tap bang nhau
    % DataSet
    
    % Xac dinh so luong anh it nhat trong 2 tap Abnormal va Normal
    %%minSetCount = min([imgSets.Count]);
    % Can bang so luong anh moi loai
    %%imgSets = partition(imgSets, minSetCount, 'randomize');
    
    % Hien thi so luong anh moi tap
    %%[imgSets.Count]
    %% Chuan bi 70% data anh de dao tao va 30% de kiem dinh
    [Setnorm1, Setnorm2] = partition(imgSetsNorm, 0.6, 'randomize');
    [Setabnorm1, Setabnorm2] = partition(imgSetsAbnorm, 0.6,'randomize');
    trainingSets = [ Setnorm1, Setabnorm1 ] ;
    testingSets = [ Setnorm2 ,Setabnorm2 ];
    %[trainingSets, testingSets] = partition(imgSets, 0.7, 'randomize');
    
    %% Tao tui tinh nang
    
    bag = bagOfFeatures(trainingSets,'VocabularySize',200,'StrongestFeatures',0.7,'Verbose',false,'PointSelection','Detector');
    
    %% Dao tao bo phan loai - Classifier
    categoryClassifier = trainImageCategoryClassifier( trainingSets, bag);
    
    %% Dung Classifier phan loai trainingSets
    tMatrix = evaluate(categoryClassifier, trainingSets);
    
    %% Tinh do chinh xac trung binh khi phan loai trainingSets
%     Average_Accuracy_train=Average_Accuracy_train+mean(diag(trainMatrix));
%     Acc_train = [Acc_train; mean(diag(trainMatrix))]; % Luu cac gia tri can plot vao de plot sau ne
%     
%     tMatrix=tMatrix+trainMatrix;
    sensiTrain = tMatrix(1,1)/( tMatrix(1,1)+ tMatrix(1,2));
    specTrain = tMatrix(2,2)/( tMatrix(2,2)+ tMatrix(2,1));
    Average_Accuracy_train= (tMatrix(1,1)*34+tMatrix(2,2)*72)/106;
    %% Dung Classifier phan loai testingSets
    [vMatrix, ~,~,nghia_score] = evaluate(categoryClassifier, testingSets);
    
    %% Tinh do chinh xac trung binh khi phan loai testingSets
%     Average_Accuracy_test=Average_Accuracy_test+mean(diag(testMatrix));
    
%     Acc_test = [Acc_test; mean(diag(testMatrix))]; % cai nay cung luu ne :x, the thoi, xong plot
%     vMatrix=vMatrix+testMatrix;
    sensiTest = vMatrix(1,1)/( vMatrix(1,1)+ vMatrix(1,2));
    specTest = vMatrix(2,2)/( vMatrix(2,2)+ vMatrix(2,1)); 
    Average_Accuracy_test=  (vMatrix(1,1)*22 + vMatrix(2,2)*48)/70;
    A (i,:)= [Average_Accuracy_test, Average_Accuracy_train, sensiTest, sensiTrain, specTest, specTrain];
end



% results(i,:)= Acc;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  % Bat dau tu day
% n_normal = 39;
% n_abnormal =84 ;
% 
% % tao ground-truth
% Ytest = [zeros(n_normal,1);ones(n_abnormal,1)];
% Ytest = (Ytest == 1);
% %(Model.ClassName 
% mcln = [1==0; 1==1];
% 
% % Back up cai nghia_score lai cho do mat neu uncomment phan ben duoi
% nghia_score_ori = nghia_score;
% 
% 
% % nghia_score = nghia_score - [nghia_score(:,1) nghia_score(:,1)];
% 
% [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Ytest, nghia_score(:,mcln),'true');
% 
% 
% % 
% plot(Xsvm,Ysvm)
% legend('linearSVM','Location','SE');
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by BOW')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Luu ket qua vao file mat
% save('KetquafullRGBorigin_0.5_max_1.mat','A');