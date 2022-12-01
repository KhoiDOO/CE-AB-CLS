% A = zeros(10,64);
 img = imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Normal\Anh_Normal_horizontal_1 (2).png']);
points = detectSURFFeatures(img,'MetricThreshold',0);
strongests = selectStrongest(points,10);
[features, valid_points] = extractFeatures(img ,strongests,'FeatureSize',64);

save('norm_horizon.mat','features');



    