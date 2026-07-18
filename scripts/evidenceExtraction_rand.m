% work in progress

% Classical Image Evidence Extraction

clc
clear
close all

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir,"..");

addpath(fullfile(projectRoot,"src"));

datasetName = "MVTec AD";
partName = "screw";
datasetFolder = fullfile(projectRoot,"data",datasetName,partName);

imds = loadDataset(datasetFolder);

% Random Good Image
goodImages = find(imds.Labels == "good");
goodIndex = goodImages(randi(length(goodImages)));

% Random Defective Image
defectImages = find(imds.Labels ~= "good");
defectIndex = defectImages(randi(length(defectImages)));

% Select image to inspect
inputImage = readimage(imds,defectIndex);

[rgbImage, grayImage] = preprocessImage(inputImage);


% Part ROI Detection

% 1. Convert image to binary
BW = imbinarize(grayImage);

figure
imshow(BW)
title("Binary Image")

% 2. Invert so the screw is white
BW = ~BW;

figure
imshow(BW)
title("Inverted Binary Image")

% 3. Remove small objects
BW = bwareaopen(BW,50);

figure
imshow(BW)
title("Small Objects Removed")

% 4. Fill holes in screw
BW = imfill(BW,"holes");

figure
imshow(BW)
title("Filled Holes")

% 5. Detect object ROI
roiStats = regionprops(BW,"BoundingBox");
bbox = roiStats.BoundingBox;

figure
imshow(grayImage)
hold on
rectangle("Position",bbox,...
    "EdgeColor","r",...
    "LineWidth",2)
title("Detected ROI")

% 6. Crop object ROI
roiImage = imcrop(grayImage,bbox);

figure
imshow(roiImage)
title("Cropped ROI")


% Classical Evidence Extraction


% 1. Enhance local contrast
enhancedImage = adapthisteq(roiImage);

figure
imshow(enhancedImage)
title("Contrast Enhanced ROI")

% 2. Gradient magnitude
% Highlights regions with rapid intensity changes.
% Used as visual evidence only.

gradientImage = imgradient(enhancedImage);

figure
imagesc(gradientImage)
axis image
colorbar
title("Gradient Magnitude")

% 3. Bottom-hat transform
% Highlights small dark surface features.

bottomHat = imbothat(enhancedImage,...
    strel("disk",8));

figure
imshow(bottomHat,[])
colorbar
title("Bottom Hat Evidence")

% 4. Generate evidence mask
threshold = graythresh(bottomHat) * 1.3; % adjustable

evidenceMask = imbinarize(bottomHat,threshold);

% Remove tiny false positives
evidenceMask = bwareaopen(evidenceMask,40);

% Smooth nearby regions
evidenceMask = imclose(evidenceMask,...
    strel("disk",2));

figure
imshow(evidenceMask)
title("Evidence Mask")

% STEP 5 - Measure Evidence

CC = bwconncomp(evidenceMask);

evidenceStats = regionprops(evidenceMask,"Area");

numComponents = CC.NumObjects;

if isempty(evidenceStats)
    largestArea = 0;
else
    largestArea = max([evidenceStats.Area]);
end

areaRatio = nnz(evidenceMask)/numel(evidenceMask);

fprintf("Number of Components : %d\n",numComponents);
fprintf("Largest Area         : %.0f pixels\n",largestArea);
fprintf("Area Ratio           : %.4f\n",areaRatio); % may need to improve

evidenceStats = regionprops(evidenceMask, "Area",...
    "BoundingBox",...
    "Centroid");

figure
imshow(roiImage)
hold on

minimumArea = 150; % adjustable

for k = 1:length(evidenceStats)

    if evidenceStats(k).Area < minimumArea
        continue
    end

    rectangle("Position",evidenceStats(k).BoundingBox,...
        "EdgeColor","r",...
        "LineWidth",2);

end

title("Classical Evidence Regions")