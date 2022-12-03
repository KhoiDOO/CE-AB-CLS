n = 120;
% m = 56;
A = zeros(n,200);
% B = zeros(m,500);
% imgSetsNorm = imageSet(fullfile('RGBoriginal', 'Normal')) ;
imgFolder = fullfile('RGBoriginal','gg' , 'Chung');
imgSetsChung = imageSet(imgFolder);
imgFolder1 = fullfile('RGBoriginal','gg' , 'Abnormal');
imgSetsAbnorm = imageSet(imgFolder1);
% imgFolder2 = fullfile('RGBoriginal','gg' , 'Normal');
% imgSetsNorm = imageSet(imgFolder2);
%  imgSets = [imgSetsNorm, imgSetsAbnorm] ;
%  [Setnorm1, Setnorm2] = partition(imgSetsNorm,0.7, 'randomize');
%  [Setabnorm1, Setabnorm2] = partition(imgSetsAbnorm, 0.7,'randomize');
%  trainingSets = [ Setnorm1, Setabnorm1 ] ;
%  testingSets = [ Setnorm2 ,Setabnorm2 ];  
% rootFolder = fullfile('E:\BTL_Y_sinh\BLT_XLAYT', 'imagepre');
% categories = {'Normal', 'Abnormal'};
% image and the category labels associated with each image
% imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
% [trainingSets, testSets] = splitEachLabel(imds, 0.7, 'randomize');
 bag = bagOfFeatures(imgSetsChung,'VocabularySize',200,'StrongestFeatures',0.1,'Verbose',false,'PointSelection','Detector');
 
%   imgs = readall(imgSetsNorm);
%   [featureVector, words] = encode(bag, imgs);
% imshow(imgs{1});
for i = 1:n
 img = readimage(imgSetsAbnorm, i);
 featureVector = encode(bag, img);
 A (i,:)= featureVector(1,:);
end
% figure
% bar(featureVector)
% title('Visual word occurrences for Normal Image')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')
save('10%AbnormChungBagofwordsK200.mat','A');

% for i = 1:m
%  img = readimage(imgSetsNorm, i);
%  featureVector = encode(bag, img);
%   B (i,:)= featureVector(1,:);
%  end
% save('Normchung500Bagofwords5.mat','B');
% fileID = fopen('feature.txt','w');
% fprintf(fileID,'%6s %12s\n','imgs','Feature');
% fprintf(fileID,'%6.2f %12.8f\r\n','featureVector');
% fclose(fileID);
% Plot the histogram of visual word occurrences
% figure
% bar(featureVector)
% title('Visual word occurrences')
% xlabel('Visual word index')
% ylabel('Frequency of occurrence')
 