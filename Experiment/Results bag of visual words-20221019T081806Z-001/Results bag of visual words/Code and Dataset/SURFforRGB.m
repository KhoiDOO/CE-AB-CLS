n= 28;
A = zeros(n,64*100);
for count=1:n 
    img = imread(['D:\BTL_Y_sinh\BLT_XLAYT\RGBoriginal\gg\Normal\CHGastro_Normal_',num2str(count),' (2).png']);
    img = img(:,:,3);
    points = detectSURFFeatures(img,'MetricThreshold',0);
    strongests = selectStrongest(points, 100);
    features = extractFeatures(img ,strongests,'FeatureSize',64);

%%%%%%%%%%%
    Nfeatures = features (:) ;
    Xfeatures = Nfeatures.' ;
    A (count,:)= Xfeatures(1,:);
%%%%%%%%%%%%%%   

end

save('Norm2SURF64select100Blue.mat','A');
