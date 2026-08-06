% Manufacturing Inspection System Evaluation

% This script measures the performance of the AI, classical, and hybrid
% inspection approaches using standard classification metrics.
clc
clear
close all

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(fullfile(projectRoot, "src"));

% Load the focused held-out test set

splitFile = fullfile(projectRoot, "models", "focused_dataset_split.mat");

load(splitFile, "imdsTest");

fprintf("Focused Test Set:\n");
disp(countEachLabel(imdsTest));

% Allocating Memory

numImages = numel(imdsTest.Files);

trueLabels = strings(numImages, 1);
aiLabels = strings(numImages, 1);
classicalLabels = strings(numImages, 1);
hybridLabels = strings(numImages, 1);

frontTriggered = false(numImages, 1);
scratchTriggered = false(numImages, 1);

% Run Inspection

for i = 1:numImages

    inputImage = readimage(imdsTest, i);

    trueLabels(i) = string(imdsTest.Labels(i));

    % Run complete inspection system

    [finalLabel, aiLabel, confidenceScore, evidenceOverlay, ...
        evidenceMetrics, baselineDecision] = inspectPart(inputImage);

    % Store results

    aiLabels(i) = aiLabel;
    classicalLabels(i) = baselineDecision;
    hybridLabels(i) = finalLabel;

    frontTriggered(i) = ...
        evidenceMetrics.FrontDecision == "FAIL";

    scratchTriggered(i) = ...
        evidenceMetrics.ScratchDecision == "FAIL";

end

% Diagnose Classical False Rejects on Good Images

goodImages = trueLabels == "PASS";

frontOnlyRejects = sum( ...
    goodImages & ...
    frontTriggered & ...
    ~scratchTriggered);

scratchOnlyRejects = sum( ...
    goodImages & ...
    ~frontTriggered & ...
    scratchTriggered);

bothRejects = sum( ...
    goodImages & ...
    frontTriggered & ...
    scratchTriggered);

correctGoodPasses = sum( ...
    goodImages & ...
    ~frontTriggered & ...
    ~scratchTriggered);

fprintf("\nGood Image Classical Diagnosis\n");
fprintf("----------------------------------\n");
fprintf("Front Detector Only  : %d\n", frontOnlyRejects);
fprintf("Scratch Detector Only: %d\n", scratchOnlyRejects);
fprintf("Both Detectors       : %d\n", bothRejects);
fprintf("Correctly Passed     : %d\n", correctGoodPasses);

figure
confusionchart(trueLabels, aiLabels);
title("AI Classifier")

figure
confusionchart(trueLabels, classicalLabels);
title("Classical Inspection System")

figure
confusionchart(trueLabels, hybridLabels);
title("Hybrid Inspection System")

% Accuracy

aiAccuracy = mean(aiLabels == trueLabels);
classicalAccuracy = mean(classicalLabels == trueLabels);
hybridAccuracy = mean(hybridLabels == trueLabels);

fprintf("\nInspection Results\n");
fprintf("=============================\n");
fprintf("\nAccuracy\n");
fprintf("-----------------------------\n");
fprintf("AI Accuracy         : %.2f%%\n", 100 * aiAccuracy);
fprintf("Classical Accuracy  : %.2f%%\n", 100 * classicalAccuracy);
fprintf("Hybrid Accuracy     : %.2f%%\n", 100 * hybridAccuracy);

% Yield - Represents the percentage of inspected parts that were accepted
% (classified as PASS) by the inspection system.

% AI Yield

numAIPass = sum(aiLabels == "PASS");
numAIFail = numImages - numAIPass;

aiYield = numAIPass / numImages;

fprintf("\nAI Inspection Summary\n");
fprintf("----------------------------------\n");
fprintf("Passed Parts : %d\n", numAIPass);
fprintf("Failed Parts : %d\n", numAIFail);
fprintf("Yield         : %.2f%%\n", 100 * aiYield);

% Hybrid Yield

numHybridPass = sum(hybridLabels == "PASS");
numHybridFail = numImages - numHybridPass;

hybridYield = numHybridPass / numImages;

fprintf("\nHybrid Inspection Summary\n");
fprintf("----------------------------------\n");
fprintf("Passed Parts : %d\n", numHybridPass);
fprintf("Failed Parts : %d\n", numHybridFail);
fprintf("Yield         : %.2f%%\n", 100 * hybridYield);

% Defect Count Plot

figure

bar([numHybridPass numHybridFail])
xticklabels(["PASS", "FAIL"])
ylabel("Number of Parts")
title("Hybrid Inspection Results")

% Display Hybrid Defect Rate

hybridDefectRate = numHybridFail / numImages;

fprintf("Hybrid Defect Rate: %.2f%%\n", ...
    100 * hybridDefectRate);

% Misclassified Images - AI

wrongAi = aiLabels ~= trueLabels;
wrongAiFiles = imdsTest.Files(wrongAi);

if ~isempty(wrongAiFiles)

    figure
    montage(wrongAiFiles)
    title("AI Misclassifications")

end

% Misclassified Images - Hybrid System

wrongHyb = hybridLabels ~= trueLabels;
wrongHybFiles = imdsTest.Files(wrongHyb);

if ~isempty(wrongHybFiles)

    figure
    montage(wrongHybFiles)
    title("Hybrid System Misclassifications")

end

% True Positive, False Negative, False Positive, True Negative

C = confusionmat( ...
    trueLabels, ...
    hybridLabels, ...
    "Order", ["FAIL", "PASS"]);

TP = C(1,1);
FN = C(1,2);
FP = C(2,1);
TN = C(2,2);

precision = TP / max(TP + FP, 1);
recall = TP / max(TP + FN, 1);

% F1 Score Calculation

f1Score = ...
    2 * precision * recall / ...
    max(precision + recall, eps);

fprintf("\nHybrid Metrics\n");
fprintf("-----------------------------\n");
fprintf("Precision : %.3f\n", precision);
fprintf("Recall    : %.3f\n", recall);
fprintf("F1 Score  : %.3f\n", f1Score);