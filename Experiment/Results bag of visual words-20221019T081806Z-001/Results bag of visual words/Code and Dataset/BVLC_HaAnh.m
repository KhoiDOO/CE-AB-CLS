n=60;
B = zeros(n,1155);
for count=1:n
image=double(imread(['E:\BTL_Y_sinh\BLT_XLAYT\them\Abnormal\Anh_Abnormal',num2str(count),' (2).png']));

%image=randi([1 10],5,5);
%disp(image);


block_size=15;
[row,col]=size(image);

r1=[0 1];
r2=[1 0];
r3=[1 1];
r4=[-1 1];
if mod(row,block_size)==0&&mod(col,block_size)==0
    image_re=image;
elseif  mod(row,block_size)==0&&mod(col,block_size)~=0
    image_re=zeros(row,col+block_size-mod(col,block_size));
elseif mod(row,block_size)~=0&&mod(col,block_size)==0
    image_re=zeros(row+block_size-mod(row,block_size),col);
else
    image_re=zeros(row+block_size-mod(row,block_size),col+block_size-mod(col,block_size));
end

image_re(1:row,1:col)=image;
%disp(image_re);

image_x=padarray(image_re,[block_size-1 block_size-1]);
%disp(image_x);
%[bvlc_row,bvlc_col]=size(image_x);
bvlc_row=floor(row/block_size)+1;
bvlc_col=floor(col/block_size)+1;

BVLC_cal=zeros(bvlc_row,bvlc_col);
for i=block_size:block_size:row+block_size-1
    for j=block_size:block_size:col+block_size-1
        block_o=image_x(i:block_size+i-1,j:block_size+j-1);
  
        block_r1=image_x(i+r1(1):i+block_size-1+r1(1),j+r1(2):j+block_size-1+r1(2));
        block_r2=image_x(i+r2(1):i+block_size-1+r2(1),j+r2(2):j+block_size-1+r2(2));
        block_r3=image_x(i+r3(1):i+block_size-1+r3(1),j+r3(2):j+block_size-1+r3(2));
        block_r4=image_x(i+r4(1):i+block_size-1+r4(1),j+r4(2):j+block_size-1+r4(2));
        
        tong_1=sum(sum(block_o.*block_r1));
        tong_2=sum(sum(block_o.*block_r2));
        tong_3=sum(sum(block_o.*block_r3));
        tong_4=sum(sum(block_o.*block_r4));
       
        mean_o=mean(mean((block_o)));
        mean_r1=mean(mean((block_r1)));
        mean_r2=mean(mean((block_r2)));
        mean_r3=mean(mean((block_r3)));
        mean_r4=mean(mean((block_r4)));
        
        re_x_o=reshape(block_o,1,[]);
        re_x_r1=reshape(block_r1,1,[]);
        re_x_r2=reshape(block_r2,1,[]);
        re_x_r3=reshape(block_r3,1,[]);
        re_x_r4=reshape(block_r4,1,[]);
        
        std_o=std(re_x_o);
        std_r1=std(re_x_r1);
        std_r2=std(re_x_r2);
        std_r3=std(re_x_r3);
        std_r4=std(re_x_r4);
        
        if std_o*std_r1==0
           p_1=0;
        else
        p_1=(tong_1-mean_o*mean_r1)/(block_size*block_size*std_o*std_r1);
        end
        if std_o*std_r2==0
           p_2=0;
        else
        p_2=(tong_2-mean_o*mean_r2)/(block_size*block_size*std_o*std_r2);
        end
        if std_o*std_r3==0
           p_3=0;
        else
        p_3=(tong_3-mean_o*mean_r3)/(block_size*block_size*std_o*std_r3);
        end
        if std_o*std_r4==0
           p_4=0;
        else
        p_4=(tong_4-mean_o*mean_r4)/(block_size*block_size*std_o*std_r4);
        end
        mat=[p_1 p_2 p_3 p_4];
        
        ib = floor(i/block_size);
        jb = floor(j/block_size);
        BVLC_cal(ib,jb)=max(mat)-min(mat);
        %bvlc(i,j)=max(mat)-min(mat);
        
       
    end
end
Nfeatures =  BVLC_cal(:) ;
IPfeatures = Nfeatures.' ;
B (count,:)= IPfeatures(1,:);

% save=(['C:\Users\Pho Ha Anh\Desktop\datn\DATN\BVLC\Abnormal\bvlc_abnormal',num2str(count),'.png']);

%final=BVLC_cal(block_size:block_size+row-1,block_size:block_size-1+col);

% imwrite(BVLC_cal, save );
% disp(BVLC_cal);
end
save('Abnorm2BVLCblock15.mat','B');
% imshow(BVLC_cal);



%fclose(save);

