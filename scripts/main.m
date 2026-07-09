clc
clear
close all

% Find the project root
scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir,"..");

% Add source code to MATLAB path
addpath(genpath(fullfile(projectRoot,"src")));

% Choose dataset
datasetName = "MVTec AD";
partName = "screw";

datasetPath = fullfile(projectRoot, "data", datasetName, partName);

% Load
fprintf("Dataset : %s\n", datasetName);
fprintf("Part    : %s\n", partName);
fprintf("Path    : %s\n\n", datasetPath);
imds = loadDataset(datasetPath);
exploreDataset(imds);

