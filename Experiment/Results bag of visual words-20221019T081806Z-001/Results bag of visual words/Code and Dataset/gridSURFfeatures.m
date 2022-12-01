addpath('E:\BTL_Y_sinh\BLT_XLAYT\FastICA_25')

n=28;  % So luong anh 
A = zeros(28,3840);

for count=1:n 
    img = imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Normal\Anh_Normal',num2str(count),' (2).png']);
%     img = imread(['E:\BTL_Y_sinh\BLT_XLAYT\GRAYoriginal\Abnormal\Anh_Abnormal1.png']);
    [height,width,numChannels] = size(img);
    %Select a point location for feature extraction
    %Use a regular spaced grid of point locations. 
    %Using the grid over the image allows for dense SURF feature extraction. The grid step is in pixels.
    gridStep = 64;
    gridX = 1:gridStep:width;
    gridY = 1:gridStep:height;

    [x,y] = meshgrid(gridX,gridY);

    gridLocations = [x(:) y(:)];
    GridPoints = [SURFPoints(gridLocations)];
    features = extractFeatures(img,GridPoints,'Upright',true);
%     features = features.';
    % Estimate ICs and projections of sortfeatures 
    [icasig, B, W] = fastica(features, 'approach', 'defl');
    featselect = icasig(1:60,:) ;
    sumfinalfeatures = featselect(:) ;
    
    IPfeatures = sumfinalfeatures.' ;
    A (count,:)=IPfeatures(1,:);
end

save('Norm2SURFICAgrid64notT.mat','A');
