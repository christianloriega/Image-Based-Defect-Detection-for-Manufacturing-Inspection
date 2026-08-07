% Load and organize the MVTec AD screw dataset for the inspection system.
% 
% This script explores the dataset.
clc
clear
close all

% Project Setup

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir, "..");

addpath(genpath(fullfile(projectRoot, "src")));

% Dataset Selection

datasetName = "MVTec AD";
partName = "screw";

datasetFolder = fullfile( projectRoot, "data", datasetName, partName);

% Validate Dataset Path

if ~isfolder(datasetFolder)

    error("Dataset folder was not found:\n%s", datasetFolder);

end

% Load Complete Dataset

imdsAll = loadDataset(datasetFolder);

fprintf("\n");
fprintf("Data Preparation Summary\n");
fprintf("========================================\n");
fprintf("Dataset          : %s\n", datasetName);
fprintf("Part Type        : %s\n", partName);
fprintf("Dataset Location : %s\n\n", datasetFolder);


% Explore Complete Dataset Function

exploreDataset(imdsAll);

% Select Final Project Classes

selectedLabels = ["good", "manipulated_front", "scratch_neck"];

keepImages = ismember(string(imdsAll.Labels), selectedLabels);

imdsFocused = subset(imdsAll, keepImages);

fprintf("\nFinal Project Scope\n");
fprintf("========================================\n");
fprintf("PASS Class       : good\n");
fprintf("FAIL Classes     : manipulated_front, scratch_neck\n");
fprintf("Focused Images   : %d\n", numel(imdsFocused.Files));
fprintf("Focused Classes  : %d\n", ...
    numel(categories(removecats(imdsFocused.Labels))));

fprintf("\nFocused Dataset Distribution\n");
fprintf("----------------------------------------\n");
disp(countEachLabel(imdsFocused));

% Convert Focused Labels to PASS and FAIL for Summary

binaryLabels = string(imdsFocused.Labels);

binaryLabels(binaryLabels == "good") = "PASS";
binaryLabels(binaryLabels ~= "PASS") = "FAIL";

imdsBinary = imdsFocused;
imdsBinary.Labels = removecats(categorical(binaryLabels));

binarySummary = countEachLabel(imdsBinary);

numPass = sum(imdsBinary.Labels == "PASS");
numFail = sum(imdsBinary.Labels == "FAIL");
totalFocusedImages = numel(imdsBinary.Files);

passPercent = 100 * numPass / totalFocusedImages;
failPercent = 100 * numFail / totalFocusedImages;
classRatio = numPass / max(numFail, 1);

fprintf("\nBinary Classification Distribution\n");
fprintf("----------------------------------------\n");
disp(binarySummary);

fprintf("PASS Images      : %d (%.2f%%)\n", numPass, passPercent);

fprintf("FAIL Images      : %d (%.2f%%)\n", numFail, failPercent);

fprintf("PASS-to-FAIL Ratio: %.2f to 1\n", classRatio);

