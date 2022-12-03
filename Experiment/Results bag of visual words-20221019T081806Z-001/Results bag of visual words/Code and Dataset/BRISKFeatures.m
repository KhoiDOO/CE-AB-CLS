n=28;
A = zeros(n,6400);
% B = zeros(60,64);
for count=1:n 
    img = imread(['D:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\them\Normal\Anh_Normal',num2str(count),'.png']);
    points = detectBRISKFeatures(img);
    strongests = selectStrongest(points,100 );
    [features, valid_points] = extractFeatures(img ,strongests,'FeatureSize',64);
%     features = features (1:60,:);     % select 60 feature dau tien
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Nfeat = normalize(features,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     features = pca(features (1:6,1:5)); % PCA feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Nfeat = Nfeat(:) ;
%     IPfeatures = Nfeat.' ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     fmeans = mean(features);
%     fstds = std(features);
%     A (count,:)=fmeans(1,:);
%     B(count,:)= fstds(1,:);
    SUMfeatures = features (:);
    IPfeatures = SUMfeatures.' ;
    A (count,:)= IPfeatures(1,:);
end
% points = detectSURFFeatures(GRAYimg,'MetricThreshold',0);
% [features, valid_points] = extractFeatures(GRAYimg, points);
% fmean = sum(features)/size(features,1);
save('Norm1BRISK64select100.mat','A');