% Train the final ResNet-18 classifier using the focused MVTec AD screw dataset.

% This script uses good, manipulated_front, and scratch_neck images, 
% evaluates the model, and saves the network used by the final hybrid inspection system.

clc
clear
close all

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(fullfile(projectRoot, "src"));

datasetName = "MVTec AD";
partName = "screw";

datasetFolder = fullfile(projectRoot, "data", datasetName, partName);

imds = loadDataset(datasetFolder);

%% Keep Only Project Classes

selectedLabels = ["good", "manipulated_front", "scratch_neck"];

keepImages = ismember(string(imds.Labels), selectedLabels);

imds = subset(imds, keepImages);

fprintf("Focused Dataset Before PASS/FAIL Conversion:\n");
disp(countEachLabel(imds));

%% Convert Labels to PASS/FAIL

labels = string(imds.Labels);

labels(labels == "good") = "PASS";
labels(labels ~= "PASS") = "FAIL";

imds.Labels = removecats(categorical(labels));

fprintf("Focused PASS/FAIL Dataset:\n");
disp(countEachLabel(imds));

%% Split Dataset

% 80% Training
% 10% Validation
% 10% Testing

rng(1)

[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.8, "randomized");

[imdsVal, imdsTest] = splitEachLabel(imdsTemp, 0.5, "randomized");

% Dataset Summary

trainTable = countEachLabel(imdsTrain);
testTable = countEachLabel(imdsTest);
valTable = countEachLabel(imdsVal);

fprintf("Training Data:\n");
disp(trainTable);

fprintf("Test Data:\n");
disp(testTable);

fprintf("Validation Data:\n");
disp(valTable);

%% Save Focused Dataset Split

modelsFolder = fullfile(projectRoot, "models");

if ~isfolder(modelsFolder)
    mkdir(modelsFolder);
end

splitFile = fullfile(modelsFolder, "focused_dataset_split.mat");

save(splitFile, "imdsTrain", "imdsVal", "imdsTest");

fprintf("Focused dataset split saved to:\n%s\n", splitFile);

%% Network Input Size

% ResNet-18 expects RGB images with a size of 224 x 224 pixels.

inputSize = [224 224 3];

%% Data Augmentation

augmentor = imageDataAugmenter(RandRotation=[-10 10], ...
    RandXTranslation=[-5 5], RandYTranslation=[-5 5]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    "DataAugmentation", augmentor, ...
    "ColorPreprocessing", "gray2rgb");

augVal = augmentedImageDatastore(inputSize, ...
    imdsVal, "ColorPreprocessing", "gray2rgb");

augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    "ColorPreprocessing", "gray2rgb");

%% Load Pretrained Network

net = imagePretrainedNetwork("resnet18", "NumClasses", 2);

analyzeNetwork(net)

%% Configure Training Options

options = trainingOptions("adam", InitialLearnRate=1e-4, MaxEpochs=20, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    ValidationData=augVal, ...
    ValidationFrequency=10, ...
    Plots="training-progress", ...
    Verbose=true);

%% Train Network

net = trainnet(augTrain, net, "crossentropy", options);

%% Test Network

accuracy = testnet(net, augTest, "accuracy");

fprintf("Test Accuracy: %.2f%%\n", accuracy);

% Predict Test Set Labels

scores = minibatchpredict(net, augTest);

predictedLabels = scores2label(scores, categories(imdsTest.Labels));

trueLabels = imdsTest.Labels;

%% Performance Metrics

figure

C = confusionmat(trueLabels, predictedLabels, "Order",...
    categorical(["FAIL","PASS"]));

confusionchart(trueLabels, predictedLabels);

title("Confusion Matrix - Focused Test Set")

TP = C(1,1);
FN = C(1,2);
FP = C(2,1);
TN = C(2,2);

precision = TP / (TP + FP);
recall = TP / (TP + FN);

f1score = 2 * (precision * recall) / (precision + recall);

fprintf("Precision: %.3f\n", precision);
fprintf("Recall: %.3f\n", recall);
fprintf("F1 Score: %.3f\n", f1score);

%% Save Focused Trained Network

modelFile = fullfile(modelsFolder, ...
    "resnet18_PASS_FAIL_focused.mat"); % good, manipulated_front, and scratched_neck files only

save(modelFile, "net");

fprintf("Focused trained network saved to:\n%s\n", modelFile);