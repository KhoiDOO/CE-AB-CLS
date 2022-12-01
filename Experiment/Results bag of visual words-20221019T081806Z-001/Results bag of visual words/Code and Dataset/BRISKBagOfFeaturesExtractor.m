
function [features,featureMetrics] = FASTBagOfFeaturesExtractor(img)
setDir  = fullfile(toolboxdir('vision'),'visiondata','imageSets');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
numImages = numel(imds);

for i = 1:numImages
    [height,width,numChannels] = size(img);
if numChannels > 1
    grayImage = rgb2gray(img);
else
    grayImage = img;
end
    img = readimage(imds,i); % read in an image from the set
    points = detectFASTFeatures(grayImage);
    features = extractFeatures(grayImage, points);
%     features = F.Features;
    featureMetrics = points.Metric;
end 

return 
end 