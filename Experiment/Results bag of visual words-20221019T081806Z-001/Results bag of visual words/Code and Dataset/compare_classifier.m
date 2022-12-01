% clear
% clc
% load('Abnorm1SURFselect100.mat');
% abnorm1_A = A;
% load('Abnorm2SURFselect100.mat');
% abnorm2_A = A;
% load('Norm1SURFselect100.mat');
% norm1_A = A;
% load('Norm2SURFselect100.mat');
% norm2_A = A;
% clear A;

%%%%%%%%%%%%%%%%%%%%%%%% BoW
clear
clc
load('10%AbnormChungBagofwordsK200.mat');
abnorm1_A = A;
load('10%NormChungBagofwordsK200.mat');
norm1_A = A;
clear A;
% clear  B;
abnorm = abnorm1_A;
norm = norm1_A;
n_image_abnorm = size(abnorm,1); % so anh abnorm
n_image_norm = size(norm,1); % so anh norm

%%%%%%%%%%%%%%%%%%%%%%%%  BoW
%%%%%%%%%%%   SURF
% abnorm = [abnorm1_A; abnorm2_A]; % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
% norm = [norm1_A; norm2_A]; % Tuong tu
% n_image_abnorm = size(abnorm,1); % so anh abnorm
% n_image_norm = size(norm,1); % so anh norm

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rng(1,'combRecursive'); %su dung seed trong random

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% s = rng;%su dung seed trong random
% rng (s);%su dung seed trong random

abnorm_random_idx = randperm(n_image_abnorm); % random toan bo dong abnorm
norm_random_idx = randperm(n_image_norm); % random toan bo dong norm

abn_nt = 60; % so anh abnorm de train
n_nt = 28 ; % so anh norm de train

abnorm_train_idx = abnorm_random_idx(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = norm_random_idx(1:n_nt); % index cua toan bo anh norm dung de train

abnorm_test_idx = abnorm_random_idx(abn_nt+1:end); % index cua test
norm_test_idx = norm_random_idx(n_nt+1:end); % index cua test

%%%%%%%%%%%%%% SURF
X_train = [abnorm(abnorm_train_idx,:); norm(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
X_test = [abnorm(abnorm_test_idx,:); norm(norm_test_idx,:)]; % tao tap test tuong tu

Y_train = [true(abn_nt,1); false(n_nt,1)]; %label: abnorm = 1; norm = 0 % tao label cho tap train
Y_test = [true(n_image_abnorm-abn_nt,1); false(n_image_norm-n_nt,1)]; % label cho tap test
%%%%%%%%%%%%%%%%%%%%%%%%%%% SVM
modelA = fitcsvm(X_train, Y_train,'KernelFunction', 'LINEAR'); % train svm

[Y_pred_svm, score] = predict(modelA, X_test);

Zsvm = (Y_pred_svm == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
AccA = sum(Zsvm(:)) / length(Y_test); % tinh ra phan tram
[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(Y_test, score(:,modelA.ClassNames),'true'  );
yiA= smooth(Ysvm);
%%%%%%%%%%%%%%%%%%%%%%%%%%% KNN 
modelB = fitcknn(X_train, Y_train,'NumNeighbors',5,'DistanceWeight','squaredinverse' ); % train knn
[Y_pred_knn, score] = predict(modelB, X_test);
Zknn = (Y_pred_knn == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
AccB = sum(Zknn(:)) / length(Y_test); % tinh ra phan tram
[Xknn,Yknn,Tknn,AUCknn] = perfcurve(Y_test, score(:,modelB.ClassNames),'true');
yiB= smooth(Yknn);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classification Tree
modelC = fitctree(X_train, Y_train); % train svm
[Y_pred_clt, score] = predict(modelC, X_test);
Zclt = (Y_pred_clt == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
AccC = sum(Zclt(:)) / length(Y_test); % tinh ra phan tram
[Xclt,Yclt,Tclt,AUCclt] = perfcurve(Y_test, score(:,modelC.ClassNames),'true');
yiC= smooth(Yclt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Discriminant analysis model
modelD = fitcdiscr(X_train, Y_train); 
[Y_pred_dis, score] = predict(modelD, X_test);
Zdis = (Y_pred_dis == Y_test); % xem Y_test vs Y_pred co nhung cai nao trung nhau
AccD = sum(Zdis(:)) / length(Y_test); % tinh ra phan tram
[Xdis,Ydis,Tdis,AUCdis] = perfcurve(Y_test, score(:,modelD.ClassNames),'true');
yiD= smooth(Ydis);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(Xsvm,Ysvm,'r', 'LineWidth',3)
hold on
plot(Xknn,Yknn,'b--o','LineWidth',3)
hold on
plot(Xclt,Yclt,'g*-','LineWidth',3)
hold on
plot(Xdis,Ydis,'p-','LineWidth',3)
% plotroc(Xsvm,Ysvm);
legend('Support Vector Machines','Knearest','Classification Tree','LDA','Location','SE');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curves for SVM, KNN, Classification Tree,LDA')
hold off