function [finalLabel, aiLabel, confidenceScore, evidenceOverlay, ...
    evidenceMetrics, baselineDecision] = inspectPart(inputImage)

persistent net

if isempty(net)

    % Load the pretrained network if not already loaded
    functionDir = fileparts(mfilename("fullpath"));
    projectRoot = fileparts(functionDir);

    load(fullfile(projectRoot, "models", ...
    "resnet18_PASS_FAIL_focused.mat"), "net");

end

% Preprocess and standardize input image

[rgbImage, grayImage] = preprocessImage(inputImage);

% AI Classification

[aiLabel, confidenceScore] = classifyPart(net, rgbImage);

% Classical Evidence Extraction

[evidenceOverlay, evidenceMetrics, baselineDecision] = ...
    extractEvidence(grayImage);

% Hybrid Fusion

uncertainPass = ...
    aiLabel == "PASS" && ...
    confidenceScore < 0.90;

if aiLabel == "FAIL"

    finalLabel = "FAIL";

elseif uncertainPass && baselineDecision == "FAIL"

    finalLabel = "FAIL";

else

    finalLabel = "PASS";

end