% function tien_xu_ly(hObject, eventdata, handles)
close all;
clear all;
clc;
n=28;  % So luong anh 
for count=1:n 
% Doc anh goc
input=imread(['E:\BTL_Y_sinh\BLT_XLAYT\fulldata\Rotatenor\Anh_90Normal_',num2str(count),' (2).png']);    
% Doc anh ROI
% mask=imread(['E:\BTL_Y_sinh\BLT_XLAYT\fulldata\Abnormal\CHGastro_Abnormal_',num2str(count),'_ROI (2).png']);     
% Chuyen anh rgb sang anh xam
input = rgb2gray(input);
%-------------------------------
%        Lay Vung ROI
%-------------------------------
% Loc thong thap gaussian
% k=fspecial('gaussian',[3 3],0.5); 
% input=imfilter(input,k);
%-------------------------------
% IM2 = imcomplement(mask);   % Xu ly anh nhi phan ROI: dao nguoc bit
% gray = 255*uint8(IM2);      % Chuyen anh nhi phan thanh anh xam
% out=input-gray;             % Lay vung anh can thiet
%------------------------------
% out = imresize(input,0.5);    % Giam kich thuoc anh
filename = (['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Normal\Anh_90Normal_',num2str(count),' (2).png']);
imwrite(input, filename );% Luu anh dau ra
end