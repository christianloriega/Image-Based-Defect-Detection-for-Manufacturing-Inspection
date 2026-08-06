function [frontOverlay, frontMetrics, frontDecision] = ...
    detectManipulatedFront(grayImage)
% % detectManipulatedFront Detects deformation near the screw tip.
% The screw is aligned horizontally, the tip region is measured, and
% simple geometry rules are used to produce a PASS or FAIL decision.

%% Prepare Image

if size(grayImage, 3) == 3
    grayImage = rgb2gray(grayImage);
end

grayImage = im2uint8(grayImage);

%% Create Screw Mask

BW = ~imbinarize(grayImage); % Inverse Binary

BW = bwareaopen(BW, 100);
BW = imfill(BW, "holes");

objectStats = regionprops(BW, "Area", "Orientation", "PixelIdxList");

% Return a safe result if no screw is detected.

if isempty(objectStats)

    frontDecision = "PASS";

    frontMetrics = struct( ...
        "TipWidth", 0, ...
        "TipArea", 0, ...
        "MaxSlope", 0, ...
        "SlopeVariation", 0, ...
        "CenterDeviation", 0, ...
        "RuleWidth", true, ...
        "RuleSlope", true, ...
        "RuleStraight", true, ...
        "FailedRules", 0);

    frontOverlay = cat(3, grayImage, grayImage, grayImage);

    return
end

%% Keep Largest Screw Object

[~, largestIndex] = max([objectStats.Area]);

screwMask = false(size(BW));

screwMask(objectStats(largestIndex).PixelIdxList) = true;

rotationAngle = -objectStats(largestIndex).Orientation;

%% Rotate Full Image and Mask

rotatedGray = imrotate(grayImage, rotationAngle, "bilinear", ...
    "loose");

rotatedMask = imrotate(screwMask, rotationAngle, "nearest", ...
    "loose");

%% Keep Largest Object After Rotation

rotatedStats = regionprops(rotatedMask, "Area", "BoundingBox", ...
    "PixelIdxList");

if isempty(rotatedStats)

    frontDecision = "PASS";

    frontMetrics = struct( ...
        "TipWidth", 0, ...
        "TipArea", 0, ...
        "MaxSlope", 0, ...
        "SlopeVariation", 0, ...
        "CenterDeviation", 0, ...
        "RuleWidth", true, ...
        "RuleSlope", true, ...
        "RuleStraight", true, ...
        "FailedRules", 0);

    frontOverlay = cat(3, grayImage, grayImage, grayImage);

    return
end

[~, largestRotatedIndex] = max([rotatedStats.Area]);

cleanMask = false(size(rotatedMask));

cleanMask(rotatedStats(largestRotatedIndex).PixelIdxList) = true;

rotatedMask = cleanMask;

% Rotate another 90 degrees if the screw is still vertical.

rotatedBox = rotatedStats(largestRotatedIndex).BoundingBox;

if rotatedBox(4) > rotatedBox(3)

    rotatedGray = rot90(rotatedGray);
    rotatedMask = rot90(rotatedMask);

end

%% Crop Around Rotated Screw

cropStats = regionprops(rotatedMask, "Area", ...
    "BoundingBox");

[~, cropIndex] = max([cropStats.Area]);

cropBox = cropStats(cropIndex).BoundingBox;

padding = 10;

xStart = max(floor(cropBox(1)) - padding, 1);
yStart = max(floor(cropBox(2)) - padding, 1);

xEnd = min(ceil(cropBox(1) + cropBox(3)) + padding,...
    size(rotatedGray, 2));

yEnd = min(ceil(cropBox(2) + cropBox(4)) + padding, ...
    size(rotatedGray, 1));

alignedGray = rotatedGray(yStart:yEnd, xStart:xEnd);

alignedMask = rotatedMask(yStart:yEnd, xStart:xEnd);

%% Put Screw Tip on the Left

columnWidth = sum(alignedMask, 1);

validColumns = find(columnWidth > 0);

if isempty(validColumns)

    frontDecision = "PASS";

    frontMetrics = struct( ...
        "TipWidth", 0, ...
        "TipArea", 0, ...
        "MaxSlope", 0, ...
        "SlopeVariation", 0, ...
        "CenterDeviation", 0, ...
        "RuleWidth", true, ...
        "RuleSlope", true, ...
        "RuleStraight", true, ...
        "FailedRules", 0);

    frontOverlay = cat(3, alignedGray, alignedGray, alignedGray);

    return
end

firstColumn = validColumns(1);
lastColumn = validColumns(end);

screwWidths = columnWidth(firstColumn:lastColumn);
screwWidths = movmean(screwWidths, 5);

sampleLength = max(round(0.15 * numel(screwWidths)), 5);

sampleLength = min(sampleLength, numel(screwWidths));

leftAverage = mean(screwWidths(1:sampleLength));

rightAverage = mean(screwWidths(end-sampleLength+1:end));

% The narrow tip should be on the left.
if leftAverage > rightAverage

    alignedGray = rot90(alignedGray, 2);
    alignedMask = rot90(alignedMask, 2);

end

%% Select Manipulated-Front Inspection Region

columnWidth = sum(alignedMask, 1);

% Preserve narrow damaged tip pixels.
validColumns = find(columnWidth >= 2);

if isempty(validColumns)

    frontDecision = "PASS";

    frontMetrics = struct( ...
        "TipWidth", 0, ...
        "TipArea", 0, ...
        "MaxSlope", 0, ...
        "SlopeVariation", 0, ...
        "CenterDeviation", 0, ...
        "RuleWidth", true, ...
        "RuleSlope", true, ...
        "RuleStraight", true, ...
        "FailedRules", 0);

    frontOverlay = cat(3, alignedGray, alignedGray, alignedGray);

    return
end

leftEdge = validColumns(1);
rightEdge = validColumns(end);

screwLength = rightEdge - leftEdge + 1;

frontLength = max(round(0.20 * screwLength), 2);

frontEnd = min(leftEdge + frontLength - 1, size(alignedMask, 2));

frontMask = alignedMask(:, leftEdge:frontEnd);

%% Measure Tip Geometry

frontWidth = sum(frontMask, 1);
frontWidth = frontWidth(frontWidth > 0);

if numel(frontWidth) >= 2

    frontWidth = movmean(frontWidth, 3);

    tipWidth = max(frontWidth);
    tipArea = nnz(frontMask);

    tipSlope = diff(frontWidth);

    maxSlope = max(abs(tipSlope(1:min(8, end))));

    slopeVariation = std(tipSlope);

else

    tipWidth = 0;
    tipArea = nnz(frontMask);
    maxSlope = 0;
    slopeVariation = 0;

end

%% Measure Tip Centerline

centerline = nan(1, size(frontMask, 2));

for columnIndex = 1:size(frontMask, 2)

    rows = find(frontMask(:, columnIndex));

    if ~isempty(rows)
        centerline(columnIndex) = mean(rows);
    end

end

validCenterline = ~isnan(centerline);

if nnz(validCenterline) >= 2
    centerDeviation = std(centerline(validCenterline));
else
    centerDeviation = 0;
end

%% Rule-Based Decision

ruleWidth = tipWidth >= 28 && tipWidth <= 39;

ruleSlope = maxSlope <= 2.5;

ruleStraight = centerDeviation <= 1;

ruleScore = sum(~[ruleWidth, ruleSlope, ruleStraight]);

if ruleScore == 0

    frontDecision = "PASS";
    boxColor = "green";

else

    frontDecision = "FAIL";
    boxColor = "red";

end

%% Package Metrics

frontMetrics = struct( ...
    "TipWidth", tipWidth, ...
    "TipArea", tipArea, ...
    "MaxSlope", maxSlope, ...
    "SlopeVariation", slopeVariation, ...
    "CenterDeviation", centerDeviation, ...
    "RuleWidth", ruleWidth, ...
    "RuleSlope", ruleSlope, ...
    "RuleStraight", ruleStraight, ...
    "FailedRules", ruleScore);

%% Create Evidence Overlay

frontOverlay = repmat(alignedGray, [1 1 3]);

frontOverlay = insertShape(frontOverlay, "Rectangle",[leftEdge, ...
        1, frontEnd - leftEdge + 1, size(alignedGray, 1)], ...
    "Color", boxColor, ...
    "LineWidth", 3);

validIndices = find(validCenterline);

for k = 1:numel(validIndices)-1

    firstIndex = validIndices(k);
    secondIndex = validIndices(k + 1);

    x1 = leftEdge + firstIndex - 1;
    x2 = leftEdge + secondIndex - 1;

    y1 = centerline(firstIndex);
    y2 = centerline(secondIndex);

    frontOverlay = insertShape(frontOverlay, ...
        "Line", [x1 y1 x2 y2], ...
        "Color", "red", ...
        "LineWidth", 2);

end

frontOverlay = insertText( frontOverlay,[leftEdge + 5, 10], ...
    char(frontDecision), ...
    "FontSize", 18, ...
    "TextColor", "white", ...
    "BoxOpacity", 0);

end