n=28;  % So luong anh 

A = zeros(n,63*64);
nICAfeatures = zeros(63,64);
% B = zeros(60,64);
for count=1:n 
    img = imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\them\Normal\Anh_Normal',num2str(count),' (2).png']);
    points = detectSURFFeatures(img,'MetricThreshold',0);
    strongests = selectStrongest(points, 100);
    features = extractFeatures(img ,strongests,'FeatureSize',64);
%     Nfeat = normalize(features,1); %%normalize feature
%     Nfeatures = features.' ;
    ICAfeatures =  fastica (features);
    if(size (ICAfeatures,1)~= 63)
        m = zeros (1,64);
        nICAfeatures = [ICAfeatures; m] ; 
    end
    Ffeatures = nICAfeatures (:) ;
    
    IPfeatures = Ffeatures.' ;
    A (count,:)= IPfeatures(1,:);
end

save('Norm2SURF64ICA.mat','A');

%  figure; imshowpair(img1, img57, 'montage');
% hold on;
%  plot(valid_points1.selectStrongest(10),'showOrientation',true);
%  hold on;
%  plot(valid_points57.selectStrongest(10),'showOrientation',true);