function [overlay, metrics, baselineDecision] = detectManipulatedFront(grayImage)

% Detects front-tip defects on a screw using classical computer vision.

% Inputs: 
% grayImage - Preprocessed grayscale screw
% Outputs: 
% overlay - RGB image with inspection overlay
% metrics - Structure containing geometric measurements
% baselineDecision - PASS or FAIL

% Detect Screw Region
BW = imbinarize(grayImage);
BW = ~BW;
BW = bwareaopen(BW, 50);

roiStats = regionprops(BW, "BoundingBox", "Orientation");

bbox = roiStats.BoundingBox;
angle = roiStats.Orientation;

% Extract Screw Region
roiImage = imcrop(grayImage, bbox);

% Horizontal Alignment
alignedGray = imrotate(roiImage, -angle, "bilinear", "loose");

roiMask = imcrop(BW,bbox);
alignedMask = imrotate(roiMask, -angle, "nearest", "loose");

% Standardize Tip Orientation

columnWidth = sum(alignedMask,1);

validColumns = columnWidth > 0;
columnWidth = columnWidth(validColumns);

columnWidth = movmean(columnWidth,5);

sampleLength = min(40,length(columnWidth));

leftAverage = mean(columnWidth(1:sampleLength));
rightAverage = mean(columnWidth(end-sampleLength+1:end));

if leftAverage > rightAverage

    alignedGray = rot90(alignedGray,2);
    alignedMask = rot90(alignedMask,2);

end

% Define Tip Inspection Region
columnWidth = sum(alignedMask,1);

validColumns = find(columnWidth > 8); % **Adjustable Treshold

leftEdge = validColumns(1);
rightEdge = validColumns(end);

screwLength = rightEdge-leftEdge;

frontLength = round(0.15*screwLength);

frontMask = alignedMask(:,leftEdge:leftEdge+frontLength-1);

% Compute Tip Geometry
frontWidth = sum(frontMask,1);

frontWidth = frontWidth(frontWidth>0);

frontWidth = movmean(frontWidth,3);

tipWidth = max(frontWidth);

tipArea = nnz(frontMask);

tipSlope = diff(frontWidth);

maxSlope = max(abs(tipSlope(1:min(8,end))));

slopeVariation = std(tipSlope);

% Centerline Analysis
centerline = nan(1,size(frontMask,2));

for c = 1:size(frontMask,2)

    rows = find(frontMask(:,c));

    if ~isempty(rows)
        centerline(c) = mean(rows);
    end

end

valid = ~isnan(centerline);

centerDeviation = std(centerline(valid));

% Rule-Based Decision
ruleWidth = tipWidth >= 28 && tipWidth <= 39;

ruleSlope = maxSlope <= 2.5;

ruleStraight = centerDeviation <= 1;

ruleScore = sum(~[ruleWidth ruleSlope ruleStraight]);

if ruleScore == 0
    baselineDecision = "PASS";
    boxColor = "green";
else
    baselineDecision = "FAIL";
    boxColor = "red";
end

% Package Metrics
metrics.TipWidth = tipWidth;
metrics.TipArea = tipArea;
metrics.MaxSlope = maxSlope;
metrics.SlopeVariation = slopeVariation;
metrics.CenterDeviation = centerDeviation;

metrics.RuleWidth = ruleWidth;
metrics.RuleSlope = ruleSlope;
metrics.RuleStraight = ruleStraight;
metrics.FailedRules = ruleScore;

% Build Overlay
overlay = insertShape(repmat(alignedGray,[1 1 3]),...
    "Rectangle",...
    [leftEdge 1 frontLength size(alignedGray,1)],...
    "Color",boxColor,...
    "LineWidth",3);

% Draw centerline
for k = 1:sum(valid)-1

    x1 = find(valid,1,'first') + k - 1 + leftEdge - 1;
    x2 = x1 + 1;

    y1 = centerline(valid);
    y2 = centerline(valid);

    overlay = insertShape(overlay,...
        "Line",...
        [x1 y1(k) x2 y2(k+1)],...
        "Color","red",...
        "LineWidth",2);

end

overlay = insertText(overlay,...
    [leftEdge+5 20],...
    char(baselineDecision),...
    "FontSize",18, ...
    "TextColor","white",...
    "BoxOpacity",0);

end

