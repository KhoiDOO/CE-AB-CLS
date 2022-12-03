clear
clc
%%%%%%%%%%%%%%%%%% load SURF
load( 'Abnorm1SURF64select100.mat');
abnorm1SURF_A = A; %sua A thanh B
load( 'Abnorm2SURF64select100.mat');
abnorm2SURF_A = A;
load( 'Norm1SURF64select100.mat');
norm1SURF_A = A;
load( 'Norm2SURF64select100.mat');
norm2SURF_A = A;
%%%%%%%%%%%%%%%%%% load BDIP
load( 'Abnorm1BDIPblock7.mat');
abnorm1BDIP_A = B; 
load( 'Abnorm2BDIPblock7.mat');
abnorm2BDIP_A = B;
load( 'Norm1BDIPblock7.mat');
norm1BDIP_A = B;
load( 'Norm2BDIPblock7.mat');
norm2BDIP_A = B;

%%%%%%%%%%%%%%%%%%% load BVLC 
load( 'Abnorm1BVLCblock8.mat');
abnorm1BVLC_A = B; 
load( 'Abnorm2BVLCblock8.mat');
abnorm2BVLC_A = B;
load( 'Norm1BVLCblock8.mat');
norm1BVLC_A = B;
load( 'Norm2BVLCblock8.mat');
norm2BVLC_A = B;

%%%%%%%%%%%%%%%%%%% load KAZE
load( 'Abnorm1KAZE64select100.mat');
abnorm1KAZE_A = A;
load( 'Abnorm2KAZE64select100.mat');
abnorm2KAZE_A = A; 
load( 'Norm1KAZE64select100.mat');
norm1KAZE_A = A;
load( 'Norm2KAZE64select100.mat');
norm2KAZE_A = A; 

clear A ;
clear B ;
abnormSURF = [abnorm1SURF_A; abnorm2SURF_A]; % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
normSURF = [norm1SURF_A; norm2SURF_A]; % Tuong tu

abnormBDIP = [abnorm1BDIP_A; abnorm2BDIP_A]; % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
normBDIP = [norm1BDIP_A; norm2BDIP_A]; % Tuong tu

abnormKAZE = [abnorm1KAZE_A; abnorm2KAZE_A]; % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
normKAZE = [norm1KAZE_A; norm2KAZE_A]; % Tuong tu

abnormBVLC = [abnorm1BVLC_A; abnorm2BVLC_A]; % Gom tat ca abnorm1 2 vao 1 bien abnorm ve sau se ghep them std vao day
normBVLC = [norm1BVLC_A; norm2BVLC_A]; % Tuong tu

Abnorm = [abnormSURF, abnormBDIP, abnormBVLC, abnormKAZE];
Norm = [normSURF, normBDIP, normBVLC, normKAZE];

n_image_abnorm = size(Abnorm,1); % so anh abnorm
n_image_norm = size(Norm,1); % so anh norm

abn_nt = 60; % so anh abnorm de train
n_nt = 28; % so anh norm de train

abnorm_train_idx = n_image_abnorm(1:abn_nt); % index cua toan bo anh abnorm dung de train
norm_train_idx = n_image_norm(1:n_nt); % index cua toan bo anh norm dung de train

abnorm_test_idx = n_image_abnorm(abn_nt+1:end); % index cua test
norm_test_idx = n_image_norm(n_nt+1:end); % index cua test

%%%%%%%%%%%%%%%%%% t-test on each feature and compare p-value
X_train = [abnorm(abnorm_train_idx,:); norm(norm_train_idx,:)]; % tao tap train bang cac ghep abnorm train vs norm train
X_test = [abnorm(abnorm_test_idx,:); norm(norm_test_idx,:)]; % tao tap test tuong tu
[h,p,ci,stat] = ttest2(X_train,X_test,'Vartype','unequal');
ecdf(p);
xlabel('P value');
ylabel('CDF value')
