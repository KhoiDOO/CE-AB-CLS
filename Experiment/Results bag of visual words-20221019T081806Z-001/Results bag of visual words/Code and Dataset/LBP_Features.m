n=60;  % So luong anh 

A = zeros(n,59);
% B = zeros(60,64);
for count=1:n 
    img = imread(['D:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\them\Abnormal\Anh_Abnormal',num2str(count),'.png']);
    features = extractLBPFeatures(img,'Radius',3);
    A (count,:)= features(1,:);
end

save('Abnorm1LBP3Radius.mat','A');

%  figure; imshowpair(img1, img57, 'montage');
% hold on;
%  plot(valid_points1.selectStrongest(10),'showOrientation',true);
%  hold on;
%  plot(valid_points57.selectStrongest(10),'showOrientation',true);