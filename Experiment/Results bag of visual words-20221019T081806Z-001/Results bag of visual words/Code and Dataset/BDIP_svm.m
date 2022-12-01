clear
clc

load( 'Abnorm1BDIPblock5stride1.mat');
abnorm1_A = B; %sua A thanh B
load( 'Abnorm2BDIPblock5stride1.mat');
abnorm2_A = B;
load( 'Norm1BDIPblock5stride1.mat');
norm1_A = B;
load( 'Norm2BDIPblock5stride1.mat');
norm2_A = B;
clear B;
%%%%%%%%%%%   SURF
abnorm = cat(3, abnorm1_A, abnorm2_A); % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
norm = cat(3, norm1_A, norm2_A); % Tuong tu
n_image_abnorm = size(abnorm,3); % so anh abnorm
n_image_norm = size(norm,3); % so anh norm
%%%%%%%%%%%%%%   bagofwords
% n_image_abnorm = size(abnorm_A,1); % so anh abnorm
% n_image_norm = size(norm_A,1); % so anh norm
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rng(1,'combRecursive'); %su dung seed trong random

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% s = rng;%su dung seed trong random
% rng (s);%su dung seed trong random


n=20;
resultA = zeros(n,1); %%%%%% luu ket qua vao bien results
AUCA = zeros(n,1);
for i= 1:n
abnorm_random_idx = randperm(n_image_abnorm); % random toan bo dong abnorm
norm_random_idx = randperm(n_image_norm); % random toan bo dong norm

abn_nt = 60; % so anh abnorm de train
n_nt = 28; % so anh norm de train

abnorm_train_idx = abnorm_random_idx(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = norm_random_idx(1:n_nt); % index cua toan bo anh norm dung de train

abnorm_test_idx = abnorm_random_idx(abn_nt+1:end); % index cua test
norm_test_idx = norm_random_idx(n_nt+1:end); % index cua test

%%%%%%%%%%%%%% SURF
X_train = cat(3, abnorm(:,:,abnorm_train_idx), norm(:,:,norm_train_idx)); % tao tap train bang cac ghep abnorm train vs norm train
X_test = cat(3,abnorm(:,:,abnorm_test_idx), norm(:,:,norm_test_idx)); % tao tap test tuong tu
%%%%%%%%%%%%%%%  bagofwords
% X_train = [abnorm_A(abnorm_train_idx,:); norm_A(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
% X_test = [abnorm_A(abnorm_test_idx,:); norm_A(norm_test_idx,:)]; % tao tap test tuong tu

Y_train = [true(abn_nt,1); false(n_nt,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_image_abnorm-abn_nt,1); false(n_image_norm-n_nt,1)]; % label cho tap test

model = fitcsvm(X_train, Y_train, 'KernelFunction', 'LINEAR'); % train svm

% TrainedModel = CVSVMModel.Trained{1};

%  SVMModel = fitPosterior(modelA);

% [Y_pred_svm, score] = predict(SVMModel, X_test);

[Y_pred_svm, score] = predict(model, X_test);

% [Y_pred, score] = predict(TrainedModel, X_test); % predict, Y_pred la label du doan ra voi score tuong ung

Zsvm = (Y_pred_svm == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
Accsvm = sum(Zsvm(:)) / length(Y_test); % tinh ra phan tram
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Y_test, score(:,model.ClassNames),'true');
resultA(i,:)= Accsvm;
AUCA(i,:) = AUCsvm ;
end
mean(resultA)
mean(AUCA)
% yiA= smooth(Ysvm);
% plot(Xsvm,yiA)
% legend('linearSVM','Location', acc'SE');
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by SVM')