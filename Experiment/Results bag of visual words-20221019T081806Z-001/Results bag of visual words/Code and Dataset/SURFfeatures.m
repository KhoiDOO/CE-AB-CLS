
n=28;  % So luong anh 

% A = zeros(n,64*100);
D = zeros(n,256);
% B = zeros(60,64);
for count=1:n 
    img = imread(['D:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\them\Normal\Anh_Normal',num2str(count),' (2).png']);
    points = detectSURFFeatures(img,'MetricThreshold',0);
    strongests = selectStrongest(points, 100);
    features = extractFeatures(img ,strongests,'FeatureSize',64);
%     Nfeat = normalize(features,1); %%normalize feature
%     PCAfeatures = pca(Nfeatures );
%     IPfeatures = PCAfeatures.' ;
%%%%%%%%%%%
%     Nfeatures = features (:) ;
%     Xfeatures = Nfeatures.' ;
%     A (count,:)= Xfeatures(1,:);
%%%%%%%%%%%%%%   
    histI = imhist(features);
    Hfeatures = histI.' ;
    D (count,:)= Hfeatures(1,:);
end

% save('Abnorm2SURF64select100PCAvar.mat','A');
save('Norm2SURF64select100His.mat','D');

%  figure; imshowpair(img1, img57, 'montage');
% hold on;
%  plot(valid_points1.selectStrongest(10),'showOrientation',true);
%  hold on;
%  plot(valid_points57.selectStrongest(10),'showOrientation',true);