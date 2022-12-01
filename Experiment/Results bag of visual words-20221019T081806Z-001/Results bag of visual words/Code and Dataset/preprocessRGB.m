n=60;  % So luong anh 
for count=1:n 
% Doc anh goc
input=imread(['E:\BTL_Y_sinh\BLT_XLAYT\fulldata\Abnormal\CHGastro_Abnormal_',num2str(count),'.png']);    
% Doc anh ROI
mask=imread(['E:\BTL_Y_sinh\BLT_XLAYT\fulldata\Abnormal\CHGastro_Abnormal_',num2str(count),'_ROI.png']);     
% Chuyen anh rgb sang anh xam
% input = rgb2gray(input);
%-------------------------------
%        Lay Vung ROI
%-------------------------------
% Loc thong thap gaussian
% k=fspecial('gaussian',[3 3],0.5); 
% input=imfilter(input,k);
%-------------------------------
IM2 = imcomplement(mask);   % Xu ly anh nhi phan ROI: dao nguoc bit
gray = 255*uint8(IM2);      % Chuyen anh nhi phan thanh anh xam
rgbImage = cat(3, gray, gray, gray); % chuyen anh xam thanh anh RGB
out=input-rgbImage;             % Lay vung anh can thiet
%------------------------------
out = imresize(out,0.5);    % Giam kich thuoc anh
filename = (['E:\BTL_Y_sinh\BLT_XLAYT\RGBdata\Abnormal\Anh_Abnormal',num2str(count),'.png']);
imwrite(out, filename );% Luu anh dau ra
end