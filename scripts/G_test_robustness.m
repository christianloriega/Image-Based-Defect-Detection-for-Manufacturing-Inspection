% Robustness Evaluation
%
% This script evaluates the AI, classical, and hybrid inspection systems
% under simulated inspection-station variability. The same test is evaluated under 
% original, brightness, contrast, blur, and noise conditions.

clc
clear
close all

% Project Setup

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(fullfile(projectRoot, "src"));

% Load Focused Test Set

splitFile = fullfile(projectRoot, "models", "focused_dataset_split.mat");

load(splitFile, "imdsTest");

fprintf("Focused Test Set:\n");
disp(countEachLabel(imdsTest));

% Define Robustness Conditions

conditions = ["Original", "Brightness", "Contrast",  "Blur", "Noise"];

numConditions = numel(conditions);
numImages = numel(imdsTest.Files);

% Allocate Result Storage

aiAccuracy = zeros(numConditions, 1);
classicalAccuracy = zeros(numConditions, 1);
hybridAccuracy = zeros(numConditions, 1);

aiFalseRejectRate = zeros(numConditions, 1);
classicalFalseRejectRate = zeros(numConditions, 1);
hybridFalseRejectRate = zeros(numConditions, 1);

hybridPrecision = zeros(numConditions, 1);
hybridRecall = zeros(numConditions, 1);
hybridF1Score = zeros(numConditions, 1);

% Evaluate Each Robustness Condition

for conditionIndex = 1:numConditions

    conditionName = conditions(conditionIndex);

    fprintf("\nTesting Condition: %s\n", conditionName);
    fprintf("----------------------------------\n");

    trueLabels = strings(numImages, 1);
    aiLabels = strings(numImages, 1);
    classicalLabels = strings(numImages, 1);
    hybridLabels = strings(numImages, 1);

    for imageIndex = 1:numImages

        % Read Original Image

        inputImage = readimage(imdsTest, imageIndex);

        trueLabels(imageIndex) = string(imdsTest.Labels(imageIndex));

        % Apply Simulated Variation

        switch conditionName

            case "Original"

                distortedImage = inputImage;

            case "Brightness"

                distortedImage = imadjust(inputImage, [], [], 0.7);

            case "Contrast"

                distortedImage = imadjust(inputImage, stretchlim(inputImage));

            case "Blur"

                distortedImage = imgaussfilt(inputImage, 2);

            case "Noise"

                distortedImage = imnoise(inputImage, "gaussian", 0, 0.01);

        end

        % Run Complete Inspection System

        [finalLabel, aiLabel, ~, ~, ~, baselineDecision] = ...
            inspectPart(distortedImage);

        % Store Inspection Decisions

        aiLabels(imageIndex) = aiLabel;
        classicalLabels(imageIndex) = baselineDecision;
        hybridLabels(imageIndex) = finalLabel;

    end

    % Calculate Accuracy

    aiAccuracy(conditionIndex) = mean(aiLabels == trueLabels);

    classicalAccuracy(conditionIndex) = mean(classicalLabels == trueLabels);

    hybridAccuracy(conditionIndex) = mean(hybridLabels == trueLabels);

    % Calculate False-Reject Rates

    % A false reject occurs when a true PASS image is classified as FAIL.

    truePassImages = trueLabels == "PASS";

    aiFalseRejectRate(conditionIndex) = sum(truePassImages & aiLabels == "FAIL") ...
        / max(sum(truePassImages), 1);

    classicalFalseRejectRate(conditionIndex) = ...
        sum(truePassImages & classicalLabels == "FAIL") / ...
        max(sum(truePassImages), 1);

    hybridFalseRejectRate(conditionIndex) = ...
        sum(truePassImages & hybridLabels == "FAIL") / ...
        max(sum(truePassImages), 1);

    % Calculate Hybrid Confusion Matrix

    C = confusionmat(trueLabels, hybridLabels, "Order", ["FAIL", "PASS"]);

    TP = C(1,1);
    FN = C(1,2);
    FP = C(2,1);
    TN = C(2,2);

    hybridPrecision(conditionIndex) = TP / max(TP + FP, 1);

    hybridRecall(conditionIndex) = TP / max(TP + FN, 1);

    hybridF1Score(conditionIndex) = 2 * hybridPrecision(conditionIndex) * ...
        hybridRecall(conditionIndex) / ...
        max(hybridPrecision(conditionIndex) + hybridRecall(conditionIndex), ...
            eps); % Use eps to prevent division by zero in the F1-score calculation.

    % Display Condition Results

    fprintf("AI Accuracy          : %.2f%%\n", 100 * aiAccuracy(conditionIndex));

    fprintf("Classical Accuracy   : %.2f%%\n", 100 * classicalAccuracy(conditionIndex));

    fprintf("Hybrid Accuracy      : %.2f%%\n", 100 * hybridAccuracy(conditionIndex));

    fprintf("Hybrid False Reject  : %.2f%%\n", 100 * hybridFalseRejectRate(conditionIndex));

    fprintf("Hybrid Precision     : %.3f\n", hybridPrecision(conditionIndex));

    fprintf("Hybrid Recall        : %.3f\n", hybridRecall(conditionIndex));

    fprintf("Hybrid F1 Score      : %.3f\n", hybridF1Score(conditionIndex));

end

% Display One Sample Under Every Robustness Condition
%
% This code is outside the evaluation loop so that MATLAB creates only
% one figure containing all five sample variations.

passIndex = find(imdsTest.Labels == "PASS", 1);

sampleImage = readimage(imdsTest, passIndex);

figure
tiledlayout(2, 3)

for sampleIndex = 1:numConditions

    sampleCondition = conditions(sampleIndex);

    switch sampleCondition

        case "Original"

            displayedImage = sampleImage;

        case "Brightness"

            displayedImage = imadjust(sampleImage, [], [], 0.7);

        case "Contrast"

            displayedImage = imadjust(sampleImage, stretchlim(sampleImage));

        case "Blur"

            displayedImage = imgaussfilt(sampleImage, 2);

        case "Noise"

            displayedImage = imnoise(sampleImage, "gaussian", 0, 0.01);

    end

    nexttile
    imshow(displayedImage)
    title(sampleCondition)

end

sgtitle("Sample Images Under Simulated Inspection Variations")

% Create Robustness Results Table

robustnessResults = table( ...
    conditions(:), ...
    100 * aiAccuracy(:), ...
    100 * classicalAccuracy(:), ...
    100 * hybridAccuracy(:), ...
    100 * aiFalseRejectRate(:), ...
    100 * classicalFalseRejectRate(:), ...
    100 * hybridFalseRejectRate(:), ...
    hybridPrecision(:), ...
    hybridRecall(:), ...
    hybridF1Score(:), ...
    VariableNames=[ ...
        "Condition", ...
        "AIAccuracyPercent", ...
        "ClassicalAccuracyPercent", ...
        "HybridAccuracyPercent", ...
        "AIFalseRejectRatePercent", ...
        "ClassicalFalseRejectRatePercent", ...
        "HybridFalseRejectRatePercent", ...
        "HybridPrecision", ...
        "HybridRecall", ...
        "HybridF1Score"]);

fprintf("\nRobustness Evaluation Summary\n");
fprintf("============================================\n");

disp(robustnessResults);

% Compare Accuracy Across Conditions

accuracyData = [100 * aiAccuracy, 100 * classicalAccuracy, ...
    100 * hybridAccuracy];

figure

bar(categorical(conditions), accuracyData)

xlabel("Image Condition")
ylabel("Accuracy (%)")
title("Inspection Accuracy Under Simulated Variations")

legend("AI", "Classical", "Hybrid", "Location", "best")

ylim([0 100])
grid on

% Compare False-Reject Rate Across Conditions

falseRejectData = [ ...
    100 * aiFalseRejectRate, ...
    100 * classicalFalseRejectRate, ...
    100 * hybridFalseRejectRate];

figure

bar(categorical(conditions), falseRejectData)

xlabel("Image Condition")
ylabel("False-Reject Rate (%)")
title("False-Reject Rate Under Simulated Variations")

legend("AI", "Classical", "Hybrid", "Location", "best")

ylim([0 100])
grid on