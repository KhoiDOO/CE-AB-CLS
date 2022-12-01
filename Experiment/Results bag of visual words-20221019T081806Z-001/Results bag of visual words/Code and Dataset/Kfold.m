tic
imgFolder1 = fullfile('GRAYoriginal','them','Fold_10','Abnormal');
Fold_x_Abnorm = imageSet(imgFolder1);
imgFolder2 = fullfile('GRAYoriginal', 'them','Fold_10' ,'Normal');
Fold_x_Norm = imageSet(imgFolder2);
imgFolder3 = fullfile('GRAYoriginal', 'them','Train_10' ,'Abnormal');
Train_Abnorm = imageSet(imgFolder3);
imgFolder4 = fullfile('GRAYoriginal', 'them','Train_10' ,'Normal');
Train_Norm = imageSet(imgFolder4);
% imdsChung = combine(Fold_1_Abnorm,Fold_1_Norm);
n_abnorm_test = Fold_x_Abnorm.Count;
n_norm_test = Fold_x_Norm.Count; % so anh norm
n_abnorm_train = Train_Abnorm.Count;
n_norm_train = Train_Norm.Count;

extractorFcn = @exampleBagOfFeaturesExtractor;% extract custom features

% abn_nt = 60 ; % so anh abnorm de train
% n_nt = 28; % so anh norm de train
% n_train = abn_nt + n_nt;
vocalsize= 1000;
A = zeros(n_abnorm_train, vocalsize);       % feature vector abnorm train
B = zeros(n_norm_train, vocalsize);         % feature vector norm train
C = zeros(n_abnorm_test, vocalsize);   % feature vector abnorm test
D = zeros(n_norm_test, vocalsize);      % feature vector norm test

% Img_train_idx =[abnorm_train_idx; norm_train_idx] ; % tao tap train bang cac ghep abnorm train vs norm train
%%%%%%%%%%%%%%%%%%%
% imgSetIn1 = select(imgSetsAbnorm, abnorm_train_idx);
% imgSetIn2 = select(imgSetsNorm, norm_train_idx);
imdsCombinedIn = [Train_Abnorm Train_Norm]; %%% ImgSet train


%%%%%%%%%%%%%%%%%%%%%%%

imdsCombinedOut = [Fold_x_Abnorm Fold_x_Norm];  %%x% ImgSet test

% bag = bagOfFeatures(imdsCombinedIn,'VocabularySize',200,'StrongestFeatures',0.3,'Verbose',false,'PointSelection','Detector');
n=20;
results = zeros(n,1);
AUCs = zeros(n,1);
for i= 1:n
bag = bagOfFeatures(imdsCombinedIn,'VocabularySize',1000,'StrongestFeatures',1,'Verbose',false,'CustomExtractor',extractorFcn);
%%%%%%%%%%%%%%%%%%%%%%
for j= 1:n_abnorm_train %%% Create features vector5

imgAbTrain = readimage(Train_Abnorm, j);
 featureVector = encode(bag, imgAbTrain);
 A (j,:)= featureVector(1,:);
end 
for k = 1: n_norm_train    %%% Create features vector
    
 imgNorTrain = readimage(Train_Norm, k);
 featureVector = encode(bag, imgNorTrain);
 B (k,:)= featureVector(1,:);
end
 X_train = [A;B];
 for h = 1:n_abnorm_test   %%% Create features vector

 imgAbTest = readimage(Fold_x_Abnorm, h);
 featureVector = encode(bag, imgAbTest);
 C(h,:)= featureVector(1,:);
 end 
 
 for m = 1:n_norm_test   %%% Create features vector

 imgNorTest = readimage(Fold_x_Norm, m);
 featureVector = encode(bag, imgNorTest);
 D(m,:)= featureVector(1,:);
 end
 X_test= [C;D]; 
Y_train = [true(n_abnorm_train,1);false(n_norm_train,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_abnorm_test,1); false(n_norm_test,1)]; % label cho tap test

% model = fitcsvm(X_train, Y_train,'KernelFunction', 'LINEAR'); % train svm
% model = fitcdiscr(X_train, Y_train);
model = fitcknn(X_train, Y_train); 

[Y_pred, score] = predict(model, X_test);

Z = (Y_pred == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
% [tpr,fpr,thresholds] = roc(irisTargets,irisOutputs)
Acc = sum(Z(:)) / length(Y_test); % tinh ra phan tram
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Y_test, score(:,model.ClassNames),'true');
results(i,:)= Acc;

AUCs(i,:)= AUCsvm; 
end
mean(results)
mean(AUCs)
toc