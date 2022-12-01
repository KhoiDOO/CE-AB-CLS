% Khoi tao so lan lap
k=10;
% Do chinh xac trung binh cua tap trainingSets
Average_Accuracy_train=0;
% Do chinh xac trung binh cua tap testingSets
Average_Accuracy_test=0;
tMatrix=zeros(2);         % Ma tran nham lan trainingSets
vMatrix=zeros(2);         % Ma tra nham lan testingSets

Acc_train = [];
Acc_test = [];
%for i=1:k
    %% Khoi tao data set
    %dataFolder = fullfile('E:\BTL Y sinh\BTL XLAYT', 'data');
    rootFolder = fullfile('E:\BTL_Y_sinh\BLT_XLAYT', 'imagepre');
    categories = {'Normal', 'Abnormal'};
    imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
    %imgSetsNorm = imageSet(fullfile('imagepre', 'Normal')) ;
    %imgSetsAbnorm = imageSet(fullfile('imagepre', 'Abnormal'));
    %imgSets = [imgSetsNorm, imgSetsAbnorm] ;
    %% Chia data anh thanh cac tap bang nhau
    % DataSet
    % Hien thi so luong anh moi tap
    %%[imgSets.Count]
    %% Chuan bi 70% data anh de dao tao va 30% de kiem dinh
%     [Setnorm1, Setnorm2] = partition(imgSetsNorm,0.7, 'randomize');
%     [Setabnorm1, Setabnorm2] = partition(imgSetsAbnorm, 0.7,'randomize');
%     trainingSets = [ Setnorm1, Setabnorm1 ];
%     testingSets = [ Setnorm2 ,Setabnorm2 ];'
[trainingSets, testSets] = splitEachLabel(imds, 0.7, 'randomize');
%     [trainNorm, testNorm] = splitEachLabel(imds,0.7,'randomize','Include','Normal');
%     [trainAbnorm, testAbnorm] = splitEachLabel(imds,0.7,'randomize','Include','Abnormal');
    
    %% Tao tui tinh nang
    
    bag = bagOfFeatures(trainingSets,'VocabularySize',200,'StrongestFeatures',0.7,'Verbose',false,'PointSelection','Detector');
    %Training an SVM classifier
    %SVMModel = fitcsvm(X,Y,'KernelFunction','rbf',...
    %'Standardize',true,'ClassNames',{'Abnormal','Normal'},'CrossVal','on');
    trainingLabels = trainingSets.Labels;
    classifier = fitcecoc(bag, trainingLabels);
    % Danh gia bo phan loai qua trainSets
    trainMatrix = evaluate(classifier, trainingSets);
    %% Tinh do chinh xac trung binh khi phan loai trainingSets

    Average_Accuracy_train=Average_Accuracy_train+mean(diag(trainMatrix));
    tMatrix=tMatrix+trainMatrix;
    predictedLabels = predict(classifier, testFeatures);
    
    categoryClassifier = trainImageCategoryClassifier(imgSets, bag);
 
%% Dung Classifier phan loai trainingSets
trainMatrix = evaluate(categoryClassifier, trainingSets);
 
%% Tinh do chinh xac trung binh khi phan loai trainingSets

Average_Accuracy_train=Average_Accuracy_train+mean(diag(trainMatrix));
tMatrix=tMatrix+trainMatrix;
%% Dung Classifier phan loai testingSets
testMatrix = evaluate(categoryClassifier, testingSets);

%% Tinh do chinh xac trung binh khi phan loai testingSets
Average_Accuracy_test=Average_Accuracy_test+mean(diag(testMatrix));
vMatrix=vMatrix+testMatrix;