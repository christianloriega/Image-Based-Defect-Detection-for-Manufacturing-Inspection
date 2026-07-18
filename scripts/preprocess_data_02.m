clc
clear
close all

scriptDir = fileparts(mfilename("fullpath"));
projectRoot = fullfile(scriptDir,"..");

addpath(fullfile(projectRoot, "src"));

datasetName = "MVTec AD";
partName = "screw";
datasetFolder = fullfile(projectRoot, "data", datasetName, partName);

imds = loadDataset(datasetFolder);

inputImage = readimage(imds,1);

[rgbImage, grayImage] = preprocessImage(inputImage);

figure

subplot(1,3,1)
imshow(inputImage)
title("Original")

subplot(1,3,2)
imshow(rgbImage)
title("Resized RGB")

subplot(1,3,3)
imshow(grayImage)
title("Grayscale")