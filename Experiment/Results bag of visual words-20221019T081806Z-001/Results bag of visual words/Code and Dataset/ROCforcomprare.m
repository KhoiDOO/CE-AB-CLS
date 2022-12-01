clear
clc
load('AbnormBagofwords.mat');
abnorm_A = A;
load('NormBagofwords.mat');
norm_A = A;
clear A;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgSetsNorm = imageSet(fullfile('RGBoriginal', 'Normal')) ;
imgSetsAbnorm = imageSet(fullfile('RGBoriginal', 'Abnormal'));
imgSets = [imgSetsNorm, imgSetsAbnorm] ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_image_abnorm = size(abnorm_A,1); % so anh abnorm
n_image_norm = size(norm_A,1); % so anh norm
% rng(10,'combRecursive');
abnorm_random_idx = randperm(n_image_abnorm); % random toan bo dong abnorm
norm_random_idx = randperm(n_image_norm); % random toan bo dong norm

% s = rng;
% rng (s);

abn_nt = 36 ; % so anh abnorm de train
n_nt = 17; % so anh norm de train

% n=10;
% results = zeros(10,1);
% for i= 1:n
abnorm_train_idx = abnorm_random_idx(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = norm_random_idx(1:n_nt); % index cua toan bo anh norm dung de train

abnorm_test_idx = abnorm_random_idx(abn_nt+1:end); % index cua test
norm_test_idx = norm_random_idx(n_nt+1:end); % index cua test

% ====== ====== ====== ====== ====== ======
all_images_norm_list = dir('RGBoriginal\Normal\');
all_images_norm_list = {all_images_norm_list(3:end).name};
for i = 1:length(all_images_norm_list)
	all_images_norm_list{i} = strcat('RGBoriginal\Normal\',all_images_norm_list{i});
end

all_images_abnorm_list = dir('RGBoriginal\Abnormal\');
all_images_abnorm_list = {all_images_abnorm_list(3:end).name};
for i = 1:length(all_images_abnorm_list)
	all_images_abnorm_list{i} = strcat('RGBoriginal\Abnormal\',all_images_abnorm_list{i});
end
% ====== ====== ====== ====== ====== ======

% ====== ====== ====== ====== ====== ======
all_images_norm_list_train = all_images_norm_list(norm_train_idx);
norm_train_set = imageSet(all_images_norm_list_train);

all_images_norm_list_test = all_images_norm_list(norm_test_idx);
norm_test_set = imageSet(all_images_norm_list_test);

all_images_abnorm_list_train = all_images_abnorm_list(abnorm_train_idx);
abnorm_train_set = imageSet(all_images_abnorm_list_train);

all_images_abnorm_list_test= all_images_abnorm_list(abnorm_test_idx);
abnorm_test_set = imageSet(all_images_abnorm_list_test);

% train_list = [all_images_abnorm_list_train all_images_norm_list_train];
% test_list = [all_images_abnorm_list_test all_images_norm_list_test ];
% train_set = imageSet(train_list);
% test_set = imageSet(test_list);
train_set = [norm_train_set abnorm_train_set ];
test_set = [norm_test_set abnorm_test_set ];
% ====== ====== ====== ====== ====== ======

bag = bagOfFeatures(train_set,'VocabularySize',200,'StrongestFeatures',0.7,'Verbose',false,'PointSelection','Detector');

X_train = [abnorm_A(abnorm_train_idx,:); norm_A(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
X_test = [abnorm_A(abnorm_test_idx,:); norm_A(norm_test_idx,:)]; % tao tap test tuong tu

Y_train = [true(abn_nt,1); false(n_nt,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_image_abnorm-abn_nt,1); false(n_image_norm-n_nt,1)]; % label cho tap test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bag = X_train;
% bag = bagofFeatures(X_train)
categoryClassifier = trainImageCategoryClassifier(imgSets, bag);
tMatrix = evaluate(categoryClassifier, train_set);
% sensiTrain = tMatrix(1,1)/( tMatrix(1,1)+ tMatrix(1,2));
% specTrain = tMatrix(2,2)/( tMatrix(2,2)+ tMatrix(2,1));
Average_Accuracy_train= (tMatrix(1,1)*17+tMatrix(2,2)*36)/53;
[vMatrix, ~,~,nghia_score] = evaluate(categoryClassifier, test_set);
% sensiTest = vMatrix(1,1)/( vMatrix(1,1)+ vMatrix(1,2));
% specTest = vMatrix(2,2)/( vMatrix(2,2)+ vMatrix(2,1)); 
Average_Accuracy_test=  (vMatrix(1,1)*39 + vMatrix(2,2)*84)/123 ;
% n_normal = 39;
% n_abnormal =84 ;

% tao ground-truth
% Ytest = [zeros(n_normal,1);ones(n_abnormal,1)];
Y_test = (Y_test == 1);
%(Model.ClassName 
mcln = [1==0; 1==1];

% Back up cai nghia_score lai cho do mat neu uncomment phan ben duoi
nghia_score_ori = nghia_score;

% nghia_score = nghia_score - [nghia_score(:,1) nghia_score(:,1)];

[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Y_test, nghia_score(:,mcln),'true');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
modelKNN = fitcknn(X_train, Y_train);
% model = fitcsvm(X_train, Y_train,'KernelFunction', 'LINEAR'); % train svm
% CVSVMModel = crossval(model);
% TrainedModel = CVSVMModel.Trained{1};

% KNNModel = fitPosterior(model);
% [Y_pred, score] = resubPredict(SVMModel);

[Y_predknn, score_KNN] = predict(modelKNN, X_test);


% [Y_pred, score] = predict(TrainedModel, X_test); % predict, Y_pred la label du doan ra voi score tuong ung

Z1 = (Y_predknn == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
Accknn = sum(Z1(:)) / length(Y_test); % tinh ra phan tram
[Xknn,Yknn,Tknn,AUCknn] = perfcurve(Y_test, score_KNN(:,modelKNN.ClassNames),'true');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ROC Decision Tree classifier
mdlDTree = fitctree(X_train, Y_train);
[Y_predtree, score_DTree] = predict(mdlDTree, X_test);
Z2 = (Y_predtree == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
AccDtree = sum(Z2(:)) / length(Y_test); % tinh ra phan tram
[Xtree,Ytree,Ttree,AUCtree] = perfcurve(Y_test, score_DTree(:,mdlDTree.ClassNames),'true');
% results(i,:)= Acc;
% end
plot(Xsvm,Ysvm)
hold on
plot(Xknn,Yknn)
plot(Xtree,Ytree)
legend('Linear SVM','K nearest neighbor','Decision Tree','Location','Best');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Linear SVM, KNN Classification and Decision Tree ')
hold off
