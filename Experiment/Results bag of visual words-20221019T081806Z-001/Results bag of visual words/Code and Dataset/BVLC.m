n= 28; % So luong anh 
for count=1:n 
input_orig=imread(['E:\BTL_Y_sinh\BLT_XLAYT\them\Abnormal\Anh_Abnormal',num2str(count),' (2).png']); 
[row_limit,col_limit]=size(input_orig);
input = im2double(input_orig);
block_size = 2;

end