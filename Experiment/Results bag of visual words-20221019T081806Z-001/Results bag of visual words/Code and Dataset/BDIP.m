n=60; % So luong anh 
% a = 3; % size block
  B = zeros(n,1155);
for count=1:n 
input_orig=imread(['E:\BTL_Y_sinh\BLT_XLAYT\them\Abnormal\Anh_Abnormal',num2str(count),' (2).png']); 
[row_limit,col_limit]=size(input_orig);
input = im2double(input_orig);

block_size = 15;
stride = 1;

feat_row = floor(row_limit/block_size) + 1;
feat_col = floor(col_limit/block_size) + 1;

feat = zeros(feat_row,feat_col);

for i = 1:block_size:row_limit
    for j = 1:block_size:col_limit
        ie = i+block_size-1;
        if (ie > row_limit)
            ie = row_limit;
        end
        je = j+block_size-1;
        if (je > col_limit)
            je = col_limit;
        end
        block = input(i:ie,j:je);
        ib = floor(i/block_size)+1;
        jb = floor(j/block_size)+1;
        M = block_size * block_size;
        sumpixel = sum(block(:));
        maxValue = max(block(:));
        feat(ib,jb) = (1/M)*(M* maxValue - sumpixel)/maxValue;
%         feat(ib,jb) = M - sumpixel/maxValue;
   end 
end
filename = (['E:\BTL_Y_sinh\BLT_XLAYT\BDIP15\Abnormal\BDIP_Abnormal',num2str(count),' (2).png']);
imwrite(feat, filename );% Luu anh dau ra
Nfeatures = feat(:) ;
IPfeatures = Nfeatures.' ;
B (count,:)= IPfeatures(1,:);
end
save('Abnorm2BDIPblock15.mat','B');
% figure(2);
% imshow(feat)
