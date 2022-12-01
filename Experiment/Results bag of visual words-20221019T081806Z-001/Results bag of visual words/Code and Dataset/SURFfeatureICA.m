addpath('E:\BTL_Y_sinh\BLT_XLAYT\FastICA_25')
n=60;  % So luong anh 
B = zeros(60,12800); % ma tran luu tru n features vector 
for count=1:n 
    % doc anh 
    img = imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Abnormal\Anh_Abnormal',num2str(count),' (2).png']); 
    % detect surf points 
    points = detectSURFFeatures(img,'MetricThreshold',0);
    % extract features tu cac points
    [features, valid_points] = extractFeatures(img, points,'FeatureSize',128);
    % chuyen vi ma tran feature
    Tfeatures = features.' ;
     % tinh tong các hàng cua Tfeature
    sumfeat = sum(Tfeatures, 2);
    % sap xep cac hang theo trat tu tu lon den be
    [sortedSums, sortOrder] = sort(sumfeat, 'Descend');
    % sap xep feature theo thu tu cac hang nhu tren
    sortfeatures = Tfeatures(sortOrder,:);
    
    % Estimate ICs and projections of sortfeatures 
    [icasig, A, W] = fastica(sortfeatures, 'approach', 'defl'); 
    % select 100 columns -> rows 
    finalfeatures = icasig (1:100,:);
    
    sumfinalfeatures = finalfeatures(:) ;
    
    IPfeatures = sumfinalfeatures.' ;
    
    B (count,:)= IPfeatures(1,:);
    
    
end
save('thuAbnorm2SURFicaselect100.mat','B');