function [scratchDecision, scratchMetrics, scratchOverlay] = detectScratchNeck(grayImage)
% detectScratchNeck Detects scratch-like defects near the screw neck.
% The function measures rough, elongated regions and returns a PASS or FAIL
% decision, evidence metrics, and a red defect overlay.

%% Prepare Image

if size(grayImage, 3) == 3
    grayImage = rgb2gray(grayImage);
end

grayImage = im2uint8(grayImage);

%% Enhance Contrast and Reduce Noise

enhancedImage = adapthisteq(grayImage, "ClipLimit", 0.02);
filteredImage = medfilt2(enhancedImage, [3 3]);

%% Locate the Screw

objectMask = ~imbinarize(filteredImage);
objectMask = bwareaopen(objectMask, 200);
objectMask = imfill(objectMask, "holes");

objectStats = regionprops(objectMask, "Area", "PixelIdxList");

% Return a safe PASS result if no screw is found.

if isempty(objectStats)

    scratchDecision = "PASS";

    scratchMetrics = struct( ...
        "ScratchArea", 0, ...
        "ScratchCount", 0, ...
        "ScratchRatio", 0, ...
        "LargestScratchArea", 0);

    scratchOverlay = cat(3, grayImage, grayImage, grayImage);

    return
end

%% Keep the Largest Object

[~, largestIndex] = max([objectStats.Area]);

screwMask = false(size(objectMask));
screwMask(objectStats(largestIndex).PixelIdxList) = true;

% Remove the outer screw boundary to avoid detecting edge reflections.

insideMask = imerode(screwMask, strel("disk", 4, 0));

%% Find the Main Screw Direction

[yCoordinates, xCoordinates] = find(screwMask);

centerX = mean(xCoordinates);
centerY = mean(yCoordinates);

centeredCoordinates = [xCoordinates - centerX, ...
    yCoordinates - centerY];

coordinateCovariance = cov(centeredCoordinates);

[eigenvectors, eigenvalues] = eig(coordinateCovariance);

[~, majorAxisIndex] = max(diag(eigenvalues));

screwDirection = eigenvectors(:, majorAxisIndex);
crossDirection = [-screwDirection(2); screwDirection(1)];

%% Project Pixels Along the Screw

[xGrid, yGrid] = meshgrid( ...
    1:size(grayImage, 2), ...
    1:size(grayImage, 1));

relativeX = xGrid - centerX;
relativeY = yGrid - centerY;

longPosition = relativeX * screwDirection(1) + ...
    relativeY * screwDirection(2);

crossPosition = relativeX * crossDirection(1) + ...
    relativeY * crossDirection(2);

screwPositions = longPosition(screwMask);

minimumPosition = min(screwPositions);
maximumPosition = max(screwPositions);

normalizedPosition = (longPosition - minimumPosition) / ...
    max(maximumPosition - minimumPosition, eps);

%% Determine the Screw Head End

firstEndMask = screwMask & normalizedPosition <= 0.15;
secondEndMask = screwMask & normalizedPosition >= 0.85;

firstEndWidth = mean(abs(crossPosition(firstEndMask)));
secondEndWidth = mean(abs(crossPosition(secondEndMask)));

% The wider end is treated as the screw head.

if secondEndWidth > firstEndWidth
    normalizedPosition = 1 - normalizedPosition;
end

%% Select the Screw Neck Region

neckMask = screwMask & insideMask & ...
    normalizedPosition >= 0.14 & ...
    normalizedPosition <= 0.40;

%% Detect Rough Texture

textureFeatures = stdfilt(filteredImage);
textureFeatures = mat2gray(textureFeatures);

neckValues = textureFeatures(neckMask);

% Return a safe PASS result if the neck region is empty.

if isempty(neckValues)

    scratchDecision = "PASS";

    scratchMetrics = struct( ...
        "ScratchArea", 0, ...
        "ScratchCount", 0, ...
        "ScratchRatio", 0, ...
        "LargestScratchArea", 0);

    scratchOverlay = cat(3, grayImage, grayImage, grayImage);

    return
end

averageNeckTexture = mean(neckValues);
neckTextureDeviation = std(neckValues);

scratchThreshold = averageNeckTexture + 1.45 * neckTextureDeviation;

scratchThreshold = max(scratchThreshold, 0.17);

defectMask = textureFeatures > scratchThreshold & neckMask;

% Clean small or disconnected detections.

defectMask = bwareaopen(defectMask, 7);
defectMask = bwmorph(defectMask, "clean");
defectMask = imclose(defectMask, strel("disk", 1, 0));

%% Inspect Candidate Regions

connectedComponents = bwconncomp(defectMask);

scratchMask = false(size(defectMask));

scratchCount = 0;
largestScratchArea = 0;

for i = 1:connectedComponents.NumObjects

    pixelIndices = connectedComponents.PixelIdxList{i};

    [componentY, componentX] = ind2sub(size(defectMask), ...
        pixelIndices);

    componentArea = numel(pixelIndices);

    if componentArea < 7
        continue
    end

    componentCoordinates = [componentX - mean(componentX), ...
        componentY - mean(componentY)];

    if size(componentCoordinates, 1) < 2
        continue
    end

    componentCovariance = cov(componentCoordinates);

    [componentVectors, componentValues] = eig(componentCovariance);

    [~, componentAxisIndex] = max(diag(componentValues));

    componentDirection = componentVectors(:, componentAxisIndex);

    directionSimilarity = abs(dot(componentDirection, screwDirection));
    directionSimilarity = min(directionSimilarity, 1);

    angleFromScrewAxis = acosd(directionSimilarity);

    componentMask = false(size(defectMask));
    componentMask(pixelIndices) = true;

    componentStats = regionprops(componentMask, ...
        "MajorAxisLength", ...
        "MinorAxisLength", ...
        "Eccentricity");

    if isempty(componentStats)
        continue
    end

    majorLength = componentStats.MajorAxisLength;
    minorLength = componentStats.MinorAxisLength;
    eccentricity = componentStats.Eccentricity;

    elongation = majorLength / max(minorLength, eps);

    % Normal highlights often run parallel to the screw.
    % Scratch regions should cross the surface or form irregular patches.

    isLargeEnough = componentArea >= 8;
    isNotTooLarge = componentArea <= 350;
    isLongEnough = majorLength >= 4.5;
    isElongated = elongation >= 1.25;
    isLineLike = eccentricity >= 0.58;
    isNotParallelHighlight = angleFromScrewAxis >= 15;

    if isLargeEnough && ...
            isNotTooLarge && ...
            isLongEnough && ...
            isElongated && ...
            isLineLike && ...
            isNotParallelHighlight

        scratchMask = scratchMask | componentMask;

        scratchCount = scratchCount + 1;

        largestScratchArea = max(largestScratchArea, componentArea);
    end
end

%% Calculate Scratch Metrics

totalScratchArea = nnz(scratchMask);
totalNeckArea = max(nnz(neckMask), 1);

scratchRatio = totalScratchArea / totalNeckArea;

scratchMetrics = struct("ScratchArea", totalScratchArea, ...
    "ScratchCount", scratchCount, ...
    "ScratchRatio", scratchRatio, ...
    "LargestScratchArea", largestScratchArea);

%% Make Inspection Decision

minimumScratchArea = 15;
minimumLargestScratchArea = 10;
minimumScratchRatio = 0.004;

scratchDetected = scratchCount >= 1 && ...
    totalScratchArea >= minimumScratchArea && ...
    largestScratchArea >= minimumLargestScratchArea && ...
    scratchRatio >= minimumScratchRatio;

if scratchDetected
    scratchDecision = "FAIL";
else
    scratchDecision = "PASS";
end

%% Create Red Evidence Overlay

scratchOverlay = cat(3, grayImage, grayImage, grayImage);

redChannel = scratchOverlay(:, :, 1);
greenChannel = scratchOverlay(:, :, 2);
blueChannel = scratchOverlay(:, :, 3);

redChannel(scratchMask) = 255;
greenChannel(scratchMask) = 0;
blueChannel(scratchMask) = 0;

scratchOverlay = cat(3, redChannel, greenChannel, blueChannel);

end