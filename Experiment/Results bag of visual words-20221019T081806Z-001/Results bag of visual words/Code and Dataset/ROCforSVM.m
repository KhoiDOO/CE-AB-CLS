clear
clc
load('AbnormBagofwords.mat');
abnorm_A = A;
load('NormBagofwords.mat');
norm_A = A;
clear A;
n_image_abnorm = size(abnorm_A,1); % so anh abnorm
n_image_norm = size(norm_A,1); % so anh norm
rng(1,'combRecursive');
abnorm_random_idx = randperm(n_image_abnorm); % random toan bo dong abnorm

norm_random_idx = randperm(n_image_norm); % random toan bo dong norm

s = rng;
rng (s);

abn_nt = 60 ; % so anh abnorm de train
n_nt = 28; % so anh norm de train

% n=10;
% results = zeros(10,1);
% for i= 1:n
abnorm_train_idx = abnorm_random_idx(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = norm_random_idx(1:n_nt); % index cua toan bo anh norm dung de train

abnorm_test_idx = abnorm_random_idx(abn_nt+1:end); % index cua test
norm_test_idx = norm_random_idx(n_nt+1:end); % index cua test

X_train = [abnorm_A(abnorm_train_idx,:); norm_A(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
X_test = [abnorm_A(abnorm_test_idx,:); norm_A(norm_test_idx,:)]; % tao tap test tuong tu

Y_train = [true(abn_nt,1); false(n_nt,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_image_abnorm-abn_nt,1); false(n_image_norm-n_nt,1)]; % label cho tap test

% model = fitcsvm(X_train, Y_train,'KernelFunction', 'LINEAR','OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'));
%  
model = fitcsvm(X_train, Y_train,'KernelFunction', 'LINEAR'); % train svm

% TrainedModel = CVSVMModel.Trained{1};

% SVMModel = fitPosterior(model);

% [Y_pred, score] = predict(SVMModel, X_test);

[Y_pred, score] = predict(model, X_test);

% [Y_pred, score] = predict(TrainedModel, X_test); % predict, Y_pred la label du doan ra voi score tuong ung

Z = (Y_pred == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
Acc = sum(Z(:)) / length(Y_test); % tinh ra phan tram
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Y_test, score(:,model.ClassNames),'true');
% results(i,:)= Acc;
% end
plot(Xsvm,Ysvm)
legend('linearSVM','Location','SE');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by SVM')