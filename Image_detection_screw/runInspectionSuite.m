% =========================================================================
% RUN INSPECTION SUITE: BATCH TESTING & FACTORY YIELD REPORT
% =========================================================================
clc; clear; close all;

%% 1. LOAD TEST DATASET & TRAINED MODEL
if ~isfile('dataset_workspace.mat')
    error('dataset_workspace.mat not found. Please run main datastore setup first.');
end
load('dataset_workspace.mat', 'imdsTest');

if ~isfile('trainNet.mat')
    error('trainNet.mat not found. Please train and save your ResNet-18 model first.');
end

numTest = numel(imdsTest.Files);
fprintf('Loaded %d test images for batch evaluation.\n', numTest);

%% 2. BATCH EVALUATION LOOP
groundTruth      = imdsTest.Labels;                    % Actual labels from dataset
aiPredictions    = categorical(repmat("", numTest, 1), ["FAIL", "PASS"]);
baselineResults  = categorical(repmat("", numTest, 1), ["FAIL", "PASS"]);
confidenceScores = zeros(numTest, 1);
flaggedAreaRatios= zeros(numTest, 1);
mismatchFlags    = false(numTest, 1);
overlayStorage   = cell(numTest, 1);

fprintf('Running Hybrid Inspection Pipeline across test set...\n');
for i = 1:numTest
    % Read raw image
   imgPath = imdsTest.Files{i};
if ~isfile(imgPath)
    warning('Image not found, skipping: %s', imgPath);
    continue;
end
I = imread(imgPath);
    
    % Execute hybrid inspector deliverable function
    [finalLabel, confidenceScore, evidenceOverlay, evidenceMetrics, baselineDecision] = inspectpart(I);
    
    % Log metrics for batch reporting
    aiPredictions(i)    = finalLabel;
    baselineResults(i)  = baselineDecision;
    confidenceScores(i) = confidenceScore;
    flaggedAreaRatios(i)= evidenceMetrics.areaRatio;
    overlayStorage{i}   = evidenceOverlay;
    
    % Track disagreement between AI classifier and Classical Rule
    mismatchFlags(i)    = (finalLabel ~= baselineDecision);
end

%% 3. CONFUSION MATRIX & ACCURACY
figure('Name', 'AI Inspection - Confusion Matrix', 'Position', [100, 200, 600, 500]);
cm = confusionchart(groundTruth, aiPredictions, ...
    'Title', 'ResNet-18 Hybrid Inspector Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Compute core metrics
numCorrect = sum(aiPredictions == groundTruth);
accuracy   = (numCorrect / numTest) * 100;

%% 4. FACTORY YIELD & DEFECT METRICS TABLE
numPassedParts = sum(aiPredictions == 'PASS');
numFailedParts = sum(aiPredictions == 'FAIL');

factoryYield   = (numPassedParts / numTest) * 100;    % % of parts marked PASS
defectRate     = (numFailedParts / numTest) * 100;    % % of parts marked FAIL
agreementRate  = (sum(~mismatchFlags) / numTest)*100; % % where AI & Baseline agree

% Build Summary Table
metricNames = { ...
    'Total Test Images Batch'; ...
    'Overall Inspection Accuracy (%)'; ...
    'Passed Parts Count (Yield Count)'; ...
    'Failed Parts Count (Defect Count)'; ...
    'Factory Yield Rate (%)'; ...
    'Factory Defect Rate (%)'; ...
    'AI vs Rule Agreement Rate (%)'};

metricValues = [ ...
    numTest; ...
    round(accuracy, 2); ...
    numPassedParts; ...
    numFailedParts; ...
    round(factoryYield, 2); ...
    round(defectRate, 2); ...
    round(agreementRate, 2)];

summaryTable = table(metricNames, metricValues, ...
    'VariableNames', {'Metric', 'Value'});

disp(' ');
disp('====================================================');
disp('             BATCH INSPECTION SUMMARY               ');
disp('====================================================');
disp(summaryTable);

%% 5. YIELD & METRIC PLOTS
figure('Name', 'Factory Inspection Yield & Metrics', 'Position', [750, 200, 700, 500]);

% Subplot A: Yield vs Defect Pie Chart
subplot(1, 2, 1);
pie([numPassedParts, numFailedParts], {'PASS (Yield)', 'FAIL (Defects)'});
title(sprintf('Batch Yield Split\n(Yield: %.1f%%)', factoryYield));

% Subplot B: Flagged Defect Area Ratio by Prediction
subplot(1, 2, 2);
boxchart(aiPredictions, flaggedAreaRatios);
ylabel('Flagged Defect Area Ratio (%)');
xlabel('AI Classification');
title('Defect Area Distribution');
grid on;

%% 6. FAILURE CASE VISUALIZATION (MONTAGE OF ERRORS)
% Identify incorrect predictions (False Positives & False Negatives)
errorIndices = find(aiPredictions ~= groundTruth);
numErrors    = length(errorIndices);

if numErrors > 0
    fprintf('\nFound %d misclassified images. Displaying error montage...\n', numErrors);
    
    errorOverlays = cell(numErrors, 1);
    for k = 1:numErrors
        idx = errorIndices(k);
        % Annotate image with True vs Predicted for easy debugging
        errText = sprintf('True: %s | AI: %s', string(groundTruth(idx)), string(aiPredictions(idx)));
        
        errorOverlays{k} = insertText(overlayStorage{idx}, [5, 200], errText, ...
            'FontSize', 10, 'BoxColor', 'red', 'TextColor', 'white');
    end
    
    figure('Name', 'Common Failure Cases (Error Montage)', 'Position', [200, 100, 800, 600]);
    montage(errorOverlays, 'Size', [NaN, 4]);
    title(sprintf('Common Failure Cases (Total Misclassifications: %d)', numErrors));
else
    disp('Perfect Batch Inspection! Zero misclassifications found.');
end