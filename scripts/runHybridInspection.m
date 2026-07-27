% Single Image Hybrid System Inspection

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

% Run Inspection
[finalLabel, confidenceScore, evidenceOverlay, evidenceMetrics, ...
    baselineDecision] = inspectPart(inputImage);

% Display Results
imshow(evidenceOverlay)

fprintf("      Hybrid Inspection Results\n");
fprintf("========================================\n\n");
fprintf("AI Inspection\n");
fprintf("------------------------------\n");
fprintf("Decision      : %s\n", finalLabel);
fprintf("Confidence    : %.1f%%\n", confidenceScore*100);
fprintf("\nClassical Inspection\n");
fprintf("------------------------------\n");
fprintf("Decision      : %s\n", baselineDecision);
fprintf("\nSystem Summary\n");
fprintf("------------------------------\n");
if finalLabel == baselineDecision
    fprintf("Status        : AI and classical inspection AGREE ✓\n");
else
    fprintf("Status        : AI and classical inspection DISAGREE ⚠\n");
end
fprintf("\nEvidence Measurements\n");
fprintf("------------------------------\n");
disp(evidenceMetrics)