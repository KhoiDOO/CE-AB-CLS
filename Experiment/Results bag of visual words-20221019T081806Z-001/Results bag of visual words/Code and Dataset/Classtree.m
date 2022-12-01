clear
clc
load('Abnorm1KAZE64select100.mat');
abnorm1_A = A;
load('Abnorm2KAZE64select100.mat');
abnorm2_A = A;
load('Norm1KAZE64select100.mat');
norm1_A = A;
load('Norm2KAZE64select100.mat');
norm2_A = A;
clear A;
%%%%%%%%%%%   SURF
abnorm = [abnorm1_A; abnorm2_A]; % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
norm = [norm1_A; norm2_A]; % Tuong tu
n_image_abnorm = size(abnorm,1); % so anh abnorm
n_image_norm = size(norm,1); % so anh norm
%%%%%%%%%%%%%%   bagofwords
% n_image_abnorm = size(abnorm_A,1); % so anh abnorm
% n_image_norm = size(norm_A,1); % so anh norm
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rng(1,'combRecursive'); %su dung seed trong random

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% s = rng;%su dung seed trong random
% rng (s);%su dung seed trong random


n=20;
results = zeros(n,1); %%%%%% luu ket qua vao bien results
AUCs = zeros(n,1);
for i= 1:n
abnorm_random_idx = randperm(n_image_abnorm); % random toan bo dong abnorm
norm_random_idx = randperm(n_image_norm); % random toan bo dong norm

abn_nt = 84; % so anh abnorm de train
n_nt = 39 ; % so anh norm de train

abnorm_train_idx = abnorm_random_idx(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = norm_random_idx(1:n_nt); % index cua toan bo anh norm dung de train

abnorm_test_idx = abnorm_random_idx(abn_nt+1:end); % index cua test
norm_test_idx = norm_random_idx(n_nt+1:end); % index cua test

%%%%%%%%%%%%%% SURF
X_train = [abnorm(abnorm_train_idx,:); norm(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
X_test = [abnorm(abnorm_test_idx,:); norm(norm_test_idx,:)]; % tao tap test tuong tu
%%%%%%%%%%%%%%%  bagofwords
% X_train = [abnorm_A(abnorm_train_idx,:); norm_A(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
% X_test = [abnorm_A(abnorm_test_idx,:); norm_A(norm_test_idx,:)]; % tao tap test tuong tu

Y_train = [true(abn_nt,1); false(n_nt,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_image_abnorm-abn_nt,1); false(n_image_norm-n_nt,1)]; % label cho tap test


model = fitctree(X_train, Y_train); % train svm

% TrainedModel = CVSVMModel.Trained{1};

% SVMModel = fitPosterior(model);

% [Y_pred, score] = predict(SVMModel, X_test);

[Y_pred, score] = predict(model, X_test);

% [Y_pred, score] = predict(TrainedModel, X_test); % predict, Y_pred la label du doan ra voi score tuong ung

Z = (Y_pred == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
Acc = sum(Z(:)) / length(Y_test); % tinh ra phan tram
[Xclt,Yclt,Tclt,AUCclt] = perfcurve(Y_test, score(:,model.ClassNames),'true');
results(i,:)= Acc;
AUCs(i,:) = AUCclt ;
end
% yi= smooth(Yclt);
% plot(Xclt,yi)
% legend('Classtree','Location','SE');
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Classtree')