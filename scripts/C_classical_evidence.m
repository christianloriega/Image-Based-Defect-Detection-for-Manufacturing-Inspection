% Classical Evidence Extraction
%
% This script documents the development of the classical vision pipeline.
% The initial implementation focused on classical computer vision pipeline 
% for detecting front-tip defects in screw images before being being pushed 
% into the complete hybrid inspection system.


% This script:
%   1. Detects the screw ROI.
%   2. Standardizes orientation.
%   3. Extracts the tip inspection region.
%   4. Computes geometric features.
%   5. Applies a rule-based baseline classifier.


clc
clear
close all

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(fullfile(projectRoot, "src"));

% Load Dataset

datasetName = "MVTec AD";
partName = "screw";
datasetFolder = fullfile(projectRoot, "data", datasetName,partName);

imds = loadDataset(datasetFolder);

% Select Random Manipulated Front Sample

manipulatedFrontImages = find(imds.Labels == "manipulated_front");
manipulatedFrontIndex = manipulatedFrontImages(randi(length(manipulatedFrontImages)));

inputImage = readimage(imds,manipulatedFrontIndex);

% Standardize Input Image

[rgbImage, grayImage] = preprocessImage(inputImage);

%% Detect Screw Region

% Convert the grayscale image into a binary image so the screw can be
% separated from the background.
BW = imbinarize(grayImage);

% Invert the binary image because MATLAB stores white pixels as logical 1.
BW = ~BW;

% Filter out small objects from the binary image.
BW = bwareaopen(BW, 50);


% Bounding box is used to crop the inspected region while the orientation
% estimates the angle to later auto rotate screw.
roiStats = regionprops(BW,"BoundingBox", "Orientation");
bbox = roiStats.BoundingBox;
angle = roiStats.Orientation;

figure
imshow(grayImage)
hold on

rectangle("Position", bbox, "EdgeColor", "r", "LineWidth", 2)

title("Detected Screw ROI")

%% Extract Screw Region

% Crop the image to the detected screw region.
roiImage = imcrop(grayImage,bbox);

figure
imshow(roiImage)
title("Inspection Region")

%% Align Screw Horizontally

% Rotate the screw so its major axis is horizontal. This standardization of
% position allows for consistent measurements at the tip of the screw.
alignedGray = imrotate(roiImage, -angle, "bilinear","loose");

roiMask = imcrop(BW,bbox);

alignedMask = imrotate(roiMask, -angle, "nearest", "loose");

%% Standardize Tip Orientation

% Compute screw width profile by summing the white pixels in each image
% column.
columnWidth = sum(alignedMask,1);

% Ignore columns containing artificial zeros that formed from rotation.
validColumns = columnWidth > 0;
columnWidth = columnWidth(validColumns);

% Smooth width profile using a moving average to reduce pixel fluctuations.
columnWidth = movmean(columnWidth, 5);

% Compare the average width at both ends to determine which side contains
% the head.
sampleLength = min(40,length(columnWidth));

leftAverage = mean(columnWidth(1:sampleLength));
rightAverage = mean(columnWidth(end-sampleLength+1:end));

% Rotate 180° if the screw head is detected on the left.
if leftAverage > rightAverage

    alignedGray = rot90(alignedGray,2);
    alignedMask = rot90(alignedMask,2);

    % Recompute width profile after rotation.
    columnWidth = sum(alignedMask,1);

    validColumns = columnWidth > 0;
    columnWidth = columnWidth(validColumns);

    columnWidth = movmean(columnWidth,5);

end


% Verify Orientation.
figure("Name","Width Profile")

plot(columnWidth,"LineWidth",2)
grid on
xlabel("Column")
ylabel("Screw Width (pixels)")
title("Width Profile")

figure

subplot(1,2,1)
imshow(grayImage)
title("Original")

subplot(1,2,2)
imshow(alignedGray)
title("Aligned Screw")


%% Define Tip Inspection Region

% Compute screw width profile.
columnWidth = sum(alignedMask,1);

% Locate the screw within the rotated image.
validColumns = find(columnWidth > 0);

leftEdge = validColumns(1);
rightEdge = validColumns(end);

% Compute total screw length.
screwLength = rightEdge - leftEdge;

% Inspect only the front 25% of the screw because the selected defect class
% ("manipulated_front") occurs at the tip.
frontLength = round(0.25*screwLength);

frontROI = alignedGray(:,leftEdge:leftEdge+frontLength);

frontMask = alignedMask(:,leftEdge:leftEdge+frontLength);

figure

subplot(1,2,1)
imshow(frontROI)
title("Front Inspection Region")

subplot(1,2,2)
imshow(frontMask)
title("Front Inspection Mask")

%% Analyze Tip Width Profile

% Compute the tip width profile by counting the foreground pixels in each
% column of the inspection mask.
frontWidth = sum(frontMask,1);

figure
plot(frontWidth,"LineWidth",2)
grid on
xlabel("Distance From Tip")
ylabel("Width (pixels)")
title("Front Width Profile")

%% Compute Tip Geometry Metrics

% Remove empty columns.
frontWidth = frontWidth(frontWidth > 0);

% Smooth the profile.
frontWidth = movmean(frontWidth,3);

% Calculate tip measurements.
tipWidth = max(frontWidth);

tipArea = nnz(frontMask);

% Compute the change in width between neighboring columns. The larger
% values indicate a more rapidly changing tip profile, while smaller values
% correspond to smoother tapers.
tipSlope = diff(frontWidth);

% Average the rate of taper near the screw tip.
averageSlope = mean(abs(tipSlope(1:min(8,end))));

% Variation in taper.
slopeVariation = std(tipSlope);

fprintf("Tip Width       : %.0f pixels\n",tipWidth);
fprintf("Tip Area        : %.0f pixels\n",tipArea);
fprintf("Average Slope   : %.2f\n",averageSlope);
fprintf("Slope Variation : %.2f\n",slopeVariation);
%% Visualize Tip Inspection Region

figure
imshow(alignedGray)
hold on

rectangle(...
    "Position",[leftEdge,...
    1,...
    frontLength,...
    size(alignedGray,1)],...
    "EdgeColor","g",...
    "LineWidth",2);

title("Front Inspection Region")

%% Rule-Based Baseline Decision

% Tip Centerline Analysis
% Well estimate the geometric center of the screw tip in every image
% column. The collection of center points forms an approximate centerline
% that can be used to evaluate the straightness of the tip after damage.

centerline = nan(1,size(frontMask,2));

for c = 1:size(frontMask,2)

    rows = find(frontMask(:,c)); % Finds all foreground pixels in the current image columns.

    if ~isempty(rows)
        centerline(c) = mean(rows);
    end

end

valid = ~isnan(centerline);

% Calculate standard deviation of the estimated centerline. A straight tip
% produces a nearly constant centerline, and a bent or damaged tip caused
% deviation.
centerDeviation = std(centerline(valid));

% Rule-based thresholds to classify screw.

ruleWidth = tipWidth >= 31 && tipWidth <= 39;

ruleSlope = averageSlope >= 0.75 && averageSlope <= 1.20;

ruleStraight = centerDeviation <= 1; % Adjustable centerline threshold**

ruleScore = sum(~[ruleWidth ruleSlope ruleStraight]);

% Final Decision

if ruleScore == 0

    baselineDecision = "PASS";
    boxColor = "g";

% elseif ruleScore == 1

%    baselineDecision = "WARNING";         % Optional include "WARNING".
%    boxColor = "y";

else

    baselineDecision = "FAIL";
    boxColor = "r";

end


% Results

fprintf("\n");
fprintf("=================================\n");
fprintf(" Rule-Based Inspection Results\n");
fprintf("=================================\n");

fprintf("Tip Width        : %.1f pixels\n",tipWidth);
fprintf("Average Slope    : %.2f\n",averageSlope);
fprintf("Center Deviation : %.2f pixels\n",centerDeviation);

fprintf("\nRule Results\n");
fprintf("----------------------------\n");
fprintf("Tip Width     : %s\n",string(ruleWidth));
fprintf("Tip Taper     : %s\n",string(ruleSlope));
fprintf("Tip Straight  : %s\n",string(ruleStraight));

fprintf("\nFailed Rules : %d / 3\n",ruleScore);
fprintf("Inspection Status : %s\n",baselineDecision);

figure
imshow(alignedGray)
hold on

rectangle(...
    "Position",[leftEdge 1 frontLength size(alignedGray,1)],...
    "EdgeColor",boxColor,...
    "LineWidth",2)

plot(find(valid)+leftEdge-1,...
     centerline(valid),...
     "r","LineWidth",2)

text(leftEdge+5,...
    20,...
    baselineDecision,...
    "Color","yellow",...
    "FontSize",18,...
    "FontWeight","bold");

title(sprintf("Front Inspection Region - %s",baselineDecision))