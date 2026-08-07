

function [finalLabel, confidenceScore,evidenceOverlay,evidenceMetrics,baselineDecision] = inspectPart(I);

% safty chek to insure that resenet is succefully loaded in
persistent net;
if isempty(net);
    if isfile('traineNet.mat');
        data = load('traineNet.mat');
        fFields = fieldnames(data)
        net = data.(fFields{1});
    else
        error('trainNet.mat not found');
    end
end

% resize images for resnet 18 and apply grey scale, medfilt2, and
% adapthsteq 
if size(I,3) == 3
    grayImg = rgb2gray(I);
else
    grayImg = I;
end
% resizing and adding greyscale 
roi = imresize(grayImg,[224,224]);
% we are applying contrast making sure there are a normal distribution of
% white and balck pixels to get a better quality grey image 
roi_contrast = adapthisteq(roi,"ClipLimit",0.02);
% we are know adding a filter to get rid of noise such as salt and peper 
roi_clean = medfilt2(roi_contrast,[3,3]);

% create mask segmentation for overlays: we are trying to isolate scratches
% from the image 

maskEvidence = defectEvidence(roi_clean);
% B. Calculate 4 Traceability Metrics
cc = bwconncomp(maskEvidence);
numComponents = cc.NumObjects;

stats = regionprops(maskEvidence, 'Area');
if ~isempty(stats)
    maxArea = max([stats.Area]);
else
    maxArea = 0;
end

areaRatio   = (nnz(maskEvidence) / numel(maskEvidence)) * 100;
edgeMask    = edge(roi_clean, 'Sobel');
edgeDensity = nnz(edgeMask) / numel(edgeMask);

% C. Store metrics in output struct
evidenceMetrics = struct(...
    'numComponents', numComponents, ...
    'maxArea',       maxArea, ...
    'areaRatio',     areaRatio, ...
    'edgeDensity',   edgeDensity);

% D. Classical Rule-Based Baseline Decision
if areaRatio > 0.5 || maxArea > 150
    baselineDecision = categorical("FAIL");
else
    baselineDecision = categorical("PASS");
end


% ---------------------------------------------------------------------
% 5. GENERATE VISUAL EVIDENCE OVERLAY
% ---------------------------------------------------------------------
% Create base RGB overlay with red defect mask
overlayImg = cat(3, roi_clean, roi_clean, roi_clean);
redChan = overlayImg(:,:,1);
redChan(maskEvidence) = 200;
overlayImg(:,:,1) = redChan;


% initialize the resnet 18 model and load images into teh nodel for testing
% purposes 

% Prepare 3-channel image for ResNet-18 [224 x 224 x 3]
roiForNet = cat(3, roi_clean, roi_clean, roi_clean);

% Call classify deliverable function
[finalLabel, confidenceScore] = classify(net, roiForNet);







% Create header text detailing AI and Classical outputs
labelStr = sprintf('AI: %s (%.1f%%) | Rule: %s | Spots: %d', ...
    string(finalLabel), confidenceScore * 100, string(baselineDecision), numComponents);

% Burn text overlay into image
evidenceOverlay = insertText(overlayImg, [5, 5], labelStr, ...
    'FontSize', 10, ...
    'BoxColor', 'black', ...
    'BoxOpacity', 0.6, ...
    'TextColor', 'white');

% ---------------------------------------------------------------------
% 6. SAVE REJECTS TO DISK (Optional)
% ---------------------------------------------------------------------
if finalLabel == categorical("FAIL")
    if ~exist('rejects_folder', 'dir')
        mkdir('rejects_folder');
    end
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS-FFF');
    imwrite(evidenceOverlay, fullfile('rejects_folder', sprintf('reject_%s.png', timestamp)));



end
end

function maskEvidence = defectEvidence(rio)
% DEFECTEVIDENCE Generates binary mask for scratches spanning glare/shadows.
%   Input:  roi - Preprocessed grayscale image from customEnhaceReader
%   Output: maskEvidence - Binary mask highlighting suspicious regions

% 1. Combine Top-Hat & Bottom-Hat to capture both bright glints & dark grooves
se = strel('disk', 4);
topHat = imtophat(rio, se);          % 1. Bright scratch glints
botHat = imbothat(rio, se);          % 2. Dark scratch grooves

scratchContrast = topHat + botHat


% 2. Threshold the combined contrast map (where scratches stand out as spikes)
T = adaptthresh(scratchContrast, 0.05,'ForegroundPolarity', 'bright', 'NeighborhoodSize', 15);
bwRaw = imbinarize(rio, T);



% 3. Bridge gaps along scratch lines
bwClosed = imclose(bwRaw, se);

% 4. Clean small background noise specks
minDefect = 5;
maskEvidence = bwareaopen(bwClosed, minDefect);






end


function [aiLabel,aiScore] = classify(net, roiForNet)
% Predict class probabilities 
scores = predict(net,single(roiForNet));

% Map class labels 
classNames = categorical(["FAIL","PASS"]);

% Extract predictive labels 
[aiScore,maxIdx] = max(scores);
aiLabel = classNames(maxIdx);
end 
