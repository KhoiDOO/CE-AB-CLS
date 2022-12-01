
function [features,featureMetrics] = exampleBagOfFeaturesExtractor(img)
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
    points = detectKAZEFeatures(grayImage);
    features = extractFeatures(grayImage, points);
    featureMetrics = points.Metric;
end 

return 
end 