function [finalLabel, confidenceScore, evidenceOverlay, evidenceMetrics,...
    baselineDecision] = inspectPart(inputImage)

persistent net

if isempty(net)
    
% Load the pretrained network if not already loaded
functionDir = fileparts(mfilename("fullpath"));
projectRoot = fileparts(functionDir);
load(fullfile(projectRoot, "models", "resnet18_PASS_FAIL.mat"), "net");

end

% Preprocess and standardize input image
[rgbImage, grayImage] = preprocessImage(inputImage);

% AI classification
[finalLabel, confidenceScore] = classifyPart(net, rgbImage);

% Classical Evidence Extraction
[evidenceOverlay, evidenceMetrics, baselineDecision] = extractEvidence(grayImage);


end