tic
imgFolder1 = fullfile('GRAYoriginal','them' ,'Abnormal');
imgSetsAbnorm = imageSet(imgFolder1);
imgFolder2 = fullfile('GRAYoriginal', 'them' ,'Normal');
imgSetsNorm = imageSet(imgFolder2);
% imdsChung = combine(imgSetsAbnorm,imgSetsNorm);

n_image_abnorm = imgSetsAbnorm.Count;
n_image_norm = imgSetsNorm.Count; % so anh norm

extractorFcn = @exampleBagOfFeaturesExtractor;% extract custom features
% abnorm = abnorm1_A;
% norm = norm1_A;

% rng('default');
% rng(1,'combRecursive'); %su dung seed trong random
% s = rng; %su dung seed trong random
n=10;
results = zeros(n,1);
AUCs = zeros(n,1);
for i= 1:n

abnorm_random_idx = randperm(n_image_abnorm); % random toan bo dong abnorm

norm_random_idx = randperm(n_image_norm); % random toan bo dong norm

% rng (s);%su dung seed trong randomcross

abn_nt = 84 ; % so anh abnorm de train
n_nt = 39; % so anh norm de train
n_train = abn_nt + n_nt;
vocalsize= 1000;
A = zeros(abn_nt, vocalsize);       % feature vector abnorm train
B = zeros(n_nt, vocalsize);         % feature vector norm train
C = zeros(120-abn_nt, vocalsize);   % feature vector abnorm test
D = zeros(56-n_nt, vocalsize);      % feature vector norm test

abnorm_train_idx = abnorm_random_idx(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = norm_random_idx(1:n_nt); % index cua toan bo anh norm dung de train
% Img_train_idx =[abnorm_train_idx; norm_train_idx] ; % tao tap train bang cac ghep abnorm train vs norm train
%%%%%%%%%%%%%%%%%%%
imgSetIn1 = select(imgSetsAbnorm, abnorm_train_idx);
imgSetIn2 = select(imgSetsNorm, norm_train_idx);
imdsCombinedIn = [imgSetIn1 imgSetIn2]; %%% ImgSet train

abnorm_test_idx = abnorm_random_idx(abn_nt+1:end);
norm_test_idx = norm_random_idx(n_nt+01:end); % index cua test
% X_test = [abnorm(abnorm_test_idx,:); norm(norm_test_idx,:)]; % tao tap test tuong tu

%%%%%%%%%%%%%%%%%%%%%%%
imgSetOut1 = select(imgSetsAbnorm, abnorm_test_idx);
imgSetOut2 = select(imgSetsNorm, norm_test_idx);
imdsCombined = [imgSetOut1 imgSetOut2];  %%% ImgSet test

% bag = bagOfFeatures(imdsCombinedIn,'VocabularySize',200,'StrongestFeatures',0.3,'Verbose',false,'PointSelection','Detector');


bag = bagOfFeatures(imdsCombinedIn,'VocabularySize',1000,'StrongestFeatures',1,'Verbose',false,'CustomExtractor',extractorFcn);
%%%%%%%%%%%%%%%%%%%%%%
for j= 1:abn_nt  %%% Create features vector5

imgAbTrain = readimage(imgSetIn1, j);
 featureVector = encode(bag, imgAbTrain);
 A (j,:)= featureVector(1,:);
end 
for k = 1: n_nt    %%% Create features vector
    
 imgNorTrain = readimage(imgSetIn2, k);
 featureVector = encode(bag, imgNorTrain);
 B (k,:)= featureVector(1,:);
end
 X_train = [A;B];
 for h = 1:120-abn_nt   %%% Create features vector

 imgAbTest = readimage(imgSetOut1, h);
 featureVector = encode(bag, imgAbTest);
 C(h,:)= featureVector(1,:);
 end 
 
 for m = 1:56-n_nt   %%% Create features vector

 imgNorTest = readimage(imgSetOut2, m);
 featureVector = encode(bag, imgNorTest);
 D(m,:)= featureVector(1,:);
 end
 X_test= [C;D]; 
Y_train = [true(abn_nt,1);false(n_nt,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_image_abnorm-abn_nt,1); false(n_image_norm-n_nt,1)]; % label cho tap test

model = fitcsvm(X_train, Y_train,'KernelFunction', 'RBF'); % train svm

% model = fitcknn(X_train, Y_train); 

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