n=60;  % So luong anh 

A = zeros(n,10);
% B = zeros(60,64);
DoUniform = true;
for count=1:n 
   img = imread(['D:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\them\Abnormal\Anh_Abnormal',num2str(count),'.png']);
    [ LBPHistogram ] = LBP(img,DoUniform);
   A (count,:)= LBPHistogram(1,:);
end

save('Abnorm1LBPothersbin10.mat','A');

%  figure; imshowpair(img1, img57, 'montage');
% hold on;
%  plot(valid_points1.selectStrongest(10),'showOrientation',true);
%  hold on;
%  plot(valid_points57.selectStrongest(10),'showOrientation',true);