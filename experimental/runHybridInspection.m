% Single Image Hybrid System Inspection

clc
clear
close all

%% Project Setup

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(fullfile(projectRoot, "src"));

%% Load Dataset

datasetName = "MVTec AD";
partName = "screw";

datasetFolder = fullfile( ...
    projectRoot, ...
    "data", ...
    datasetName, ...
    partName);

imds = loadDataset(datasetFolder);

%% Select Random Sample

defectImages = find(imds.Labels == "scratch_neck");

if isempty(defectImages)
    error("No manipulated_front images were found in the dataset.");
end

defectIndex = defectImages(randi(numel(defectImages)));
inputImage = readimage(imds, defectIndex);

%% Run Inspection

[finalLabel, aiLabel, confidenceScore, evidenceOverlay, ...
    evidenceMetrics, baselineDecision] = inspectPart(inputImage);

%% Display Results

figure
imshow(evidenceOverlay)
title("Final Hybrid Decision: " + string(finalLabel))

fprintf("\n");
fprintf("Hybrid Inspection Results\n");
fprintf("========================================\n\n");

fprintf("AI Inspection\n");
fprintf("------------------------------\n");
fprintf("Decision         : %s\n", string(aiLabel));
fprintf("Confidence       : %.1f%%\n", confidenceScore * 100);

fprintf("\nClassical Inspection\n");
fprintf("------------------------------\n");
fprintf("Decision         : %s\n", string(baselineDecision));
fprintf("Detected Defect  : %s\n", string(evidenceMetrics.DetectedDefect));

if evidenceMetrics.FrontDecision == "FAIL"

    fprintf("\nManipulated-Front Evidence\n");
    fprintf("------------------------------\n");
    fprintf("Tip Width        : %.2f\n", evidenceMetrics.TipWidth);
    fprintf("Tip Area         : %.0f\n", evidenceMetrics.TipArea);
    fprintf("Max Slope        : %.3f\n", evidenceMetrics.MaxSlope);
    fprintf("Slope Variation  : %.3f\n", evidenceMetrics.SlopeVariation);
    fprintf("Center Deviation : %.3f\n", evidenceMetrics.CenterDeviation);
    fprintf("Failed Rules     : %d\n", evidenceMetrics.FailedRules);

end

if evidenceMetrics.ScratchDecision == "FAIL"

    fprintf("\nScratch-Neck Evidence\n");
    fprintf("------------------------------\n");
    fprintf("Scratch Area     : %d\n", evidenceMetrics.ScratchArea);
    fprintf("Scratch Count    : %d\n", evidenceMetrics.ScratchCount);
    fprintf("Scratch Ratio    : %.4f\n", evidenceMetrics.ScratchRatio);
    fprintf("Largest Scratch  : %d\n", ...
        evidenceMetrics.LargestScratchArea);

end

fprintf("\nFinal Hybrid Decision\n");
fprintf("------------------------------\n");
fprintf("Decision         : %s\n", string(finalLabel));

fprintf("\nSystem Summary\n");
fprintf("------------------------------\n");

if aiLabel == baselineDecision

    fprintf("Status           : AI and classical inspection AGREE✓\n");

else

    fprintf("Status           : AI and classical inspection DISAGREE⚠\n");

end