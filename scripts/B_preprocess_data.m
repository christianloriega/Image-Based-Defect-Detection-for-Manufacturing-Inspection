% Image Preprocessing
%
% This script demonstrates how one screw image is standardized for the AI
% classifier and the classical vision branch of the inspection system.

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

datasetFolder = fullfile(projectRoot, "data", datasetName, ...
partName);

imds = loadDataset(datasetFolder);

%% Display Project Scope

selectedLabels = ["good", "manipulated_front", "scratch_neck"];

keepImages = ismember(string(imds.Labels), selectedLabels);

imdsFocused = subset(imds, keepImages);

fprintf("Image Preprocessing\n");
fprintf("========================================\n");
fprintf("Dataset          : %s\n", datasetName);
fprintf("Part Type        : %s\n", partName);
fprintf("Focused Classes  : good, manipulated_front, scratch_neck\n");
fprintf("Focused Images   : %d\n\n", numel(imdsFocused.Files));

fprintf("Focused Dataset Distribution\n");
fprintf("----------------------------------------\n");
disp(countEachLabel(imdsFocused));

%% Read Example Image

inputImage = readimage(imdsFocused, 1);

%% Preprocess Image

% preprocessImage returns:
%   rgbImage  - resized RGB image used by ResNet-18
%   grayImage - grayscale image used by classical vision algorithms

[rgbImage, grayImage] = preprocessImage(inputImage);

%% Display Preprocessing Results

figure("Name", "Preprocessing Pipeline", "Position", [100 100 1200 450]);

subplot(1,3,1)
imshow(inputImage)
title("Original")

subplot(1,3,2)
imshow(rgbImage)
title("RGB for AI")

subplot(1,3,3)
imshow(grayImage)
title("Gray for Vision")

sgtitle("Image Preprocessing Pipeline")