n=28;  % So luong anh 
for count=1:n 
% Doc anh goc
input=imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Normal\Anh_Normal',num2str(count),'.png']); 
out = imresize(input,0.5);    % Giam kich thuoc anh
filename = (['E:\BTL_Y_sinh\BLT_XLAYT\GRAYresize\Normal\Anh_Normalresize_',num2str(count),'.png']);
imwrite(out, filename );% Luu anh dau ra

end