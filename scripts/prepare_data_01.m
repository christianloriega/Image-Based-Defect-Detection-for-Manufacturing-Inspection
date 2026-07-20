clc
clear
close all

%% Project Setup

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir,"..");

addpath(genpath(fullfile(projectRoot,"src")));

%% Dataset Selection

datasetName = "MVTec AD";
partName = "screw";

datasetFolder = fullfile(projectRoot,"data",datasetName,partName);

%% Load Dataset

fprintf("Dataset : %s\n", datasetName);
fprintf("Part    : %s\n", partName);
fprintf("Path    : %s\n\n", datasetFolder);

imds = loadDataset(datasetFolder);

%% Explore Dataset

exploreDataset(imds);

%% Split Dataset

% 80% Training
% 20% Temporary

[imdsTrain,imdsTemp] = splitEachLabel(imds,0.8,"randomized");

% Split remaining 20%

[imdsTest,imdsVal] = splitEachLabel(imdsTemp,0.5,"randomized");

%% Dataset Summary

trainTable = countEachLabel(imdsTrain);
testTable  = countEachLabel(imdsTest);
valTable   = countEachLabel(imdsVal);

fprintf("Training Data:\n");
disp(trainTable);

fprintf("Test Data:\n");
disp(testTable);

fprintf("Validation Data:\n");
disp(valTable);
