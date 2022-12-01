n=28;  % So luong anh 
A = zeros(28,3000); % ma tran luu tru n features vector 
for count=1:n 
    % doc anh 
    img = imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Normal\Anh_Normal',num2str(count),'.png']); 
    % detect surf points 
    points = detectSURFFeatures(img,'MetricThreshold',0);
    % extract features tu cac points
    [features, valid_points] = extractFeatures(img, points,'FeatureSize',128);
    % normalize features vector
%     Nfeat = normalize(features,1);
    
    % Principal component analysis features vector
    PCAfeatures = pca(features);
    % select 30 variables max
    outfeatures = PCAfeatures(:,1:30);
    % 
    infeatures = features*outfeatures ;
    % tinh tong các hàng cua feature
    sumin = sum(infeatures, 2);
    % sap xep cac hang theo trat tu tu lon den be
    [sortedSums, sortOrder] = sort(sumin, 'Descend');
    % sap xep feature theo thu tu cac hang nhu tren
    sortfeatures = infeatures(sortOrder, :);
    finalfeatures = sortfeatures (1:100,:);
    sumfinalfeatures = finalfeatures(:) ;
    IPfeatures = sumfinalfeatures.' ;
    A (count,:)= IPfeatures(1,:);
    
end
save('Norm1SURFpcaselect30notnorm.mat','A');