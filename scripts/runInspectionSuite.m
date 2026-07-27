% Evaluating Inspection System Performance

clc
clear
close all

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(fullfile(projectRoot, "src"));

% Network

load(fullfile(projectRoot,"models","resnet18_PASS_FAIL.mat"),"net");


% Load Dataset

datasetName = "MVTec AD";
partName = "screw";
datasetFolder = fullfile(projectRoot, "data", datasetName,partName);

imds = loadDataset(datasetFolder);

% Split Dataset (Test Images)

[~, imdsTemp] = splitEachLabel(imds, 0.8, "randomized");
[imdsTest, ~] = splitEachLabel(imdsTemp, 0.5, "randomized");

% Allocating Memory

numImages = numel(imdsTest.Files);

trueLabels = strings(numImages, 1);
aiLabels = strings(numImages, 1);
hybridLabels = strings(numImages, 1);

% Run Inspection

for i = 1:numImages

    inputImage = readimage(imdsTest, i);

    label = string(imdsTest.Labels(i));

    if label == "good"
        trueLabels(i) = "PASS";
    else
        trueLabels(i) = "FAIL";
    end

    % AI inspection

    [aiLabel, aiScore] = classifyPart(net, inputImage);

    % Hybrid inspection

    [~, confidenceScore, evidenceOverlay, evidenceMetrics, ...
        baselineDecision] = inspectPart(inputImage);

    % Hybrid fusion logic

    if baselineDecision == "FAIL"
        hybridLabel = "FAIL";
    else
        hybridLabel = aiLabel;
    end

    % Results

    aiLabels(i) = aiLabel;
    hybridLabels(i) = hybridLabel;


end

figure
confusionchart(trueLabels, aiLabels);
title("AI Classifier")

figure
confusionchart(trueLabels, hybridLabels);
title("Hybrid Inspection System")

% Accuracy

aiAccuracy = mean(aiLabels == trueLabels);
hybridAccuracy = mean(hybridLabels == trueLabels);

fprintf("Inspection Results\n");
fprintf("=============================\n");
fprintf("\nAccuracy\n");
fprintf("-----------------------------\n");
fprintf("AI Accuracy      : %.2f%%\n",100*aiAccuracy);
fprintf("Hybrid Accuracy  : %.2f%%\n",100*hybridAccuracy);

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
fprintf("Yield         : %.2f%%\n", 100*aiYield);

% Hybrid Yield

numHybridPass = sum(hybridLabels == "PASS");
numHybridFail = numImages - numHybridPass;

hybridYield = numHybridPass / numImages;

fprintf("\nHybrid Inspection Summary\n");
fprintf("----------------------------------\n");
fprintf("Passed Parts : %d\n", numHybridPass);
fprintf("Failed Parts : %d\n", numHybridFail);
fprintf("Yield         : %.2f%%\n", 100*hybridYield);

% Defect Count Plot
figure

bar([numHybridPass numHybridFail])
xticklabels(["PASS", "FAIL"])
ylabel("Number of Parts")
title("Hybrid Inspection Results")

% Display Hybrid Defect Rate (opposite of yield)
hybridDefectRate = numHybridFail / numImages;
fprintf("Hybrid Defect Rate: %.2f%%\n", 100 * hybridDefectRate);

% Misclassified Images - AI

wrongAi = aiLabels ~= trueLabels;
wrongAiFiles = imdsTest.Files(wrongAi);

figure
montage(wrongAiFiles)
title("AI Misclassifications")


% Misclassified Images - Hybrid System

wrongHyb = hybridLabels ~= trueLabels;
wrongHybFiles = imdsTest.Files(wrongHyb);

figure
montage(wrongHybFiles)
title("Hybrid System Misclassifications")

% True Positive, False Negative, False Positive, True Negative
C = confusionmat(trueLabels, hybridLabels, ...
    "Order", ["FAIL","PASS"]);
TP = C(1,1);
FN = C(1,2);
FP = C(2,1);
TN = C(2,2);

precision = TP / (TP + FP);
recall = TP / (TP + FN);

% F1 Score Calculation
f1Score = 2 * ((precision * recall) / (precision + recall));

fprintf("\nHybrid Metrics\n");
fprintf("-----------------------------\n");
fprintf("Precision : %.3f\n", precision);
fprintf("Recall    : %.3f\n", recall);
fprintf("F1 Score  : %.3f\n", f1Score);