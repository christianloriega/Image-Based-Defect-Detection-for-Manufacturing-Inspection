%%%%%%%%%%%%%%%%%%%
%%%%%%%% APPLY NEW SEGMENTATION AND OVERLAYS FOR IMAGE PROCESSING %%%%%%%%
% **********************************************************************
% INDUSTRIAL IMAGE INSPECTOR USING RESNET 18 (WITH DEFECT OVERLAY PIPELINE)
% **********************************************************************
clc; clear; close all;

% Data exploration and setup 
passDir = fullfile('data', 'MVTec AD', 'screw', 'test', 'good');
failDir = fullfile('data', 'MVTec AD', 'screw', 'test', 'scratch_neck');

csvFilename = "factory_inspection_meta.csv";
if isfile(csvFilename)
    delete(csvFilename)
end

buildMetadataCSV(csvFilename, passDir, failDir);
metadataTable = readtable(csvFilename, 'Delimiter', ',');

% Make datastore with labels
imds = imageDatastore(metadataTable.FilePath, 'Labels', categorical(metadataTable.labels));
disp('Dataset Exploration: Label Counts & Balance Summary:');
disp(countEachLabel(imds));

% Split dataset
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.70, 0.15, 0.15, 'randomized');

% Attach custom reader that generates grayscale + contrast + defect overlay (RGB)
imdsTrain.ReadFcn = @customEnhaceReader;
imdsVal.ReadFcn   = @customEnhaceReader;
imdsTest.ReadFcn  = @customEnhaceReader;

save('dataset_workspace.mat', 'imdsTrain', 'imdsVal', 'imdsTest');

inputSize = [224, 224];
augmenter = imageDataAugmenter("RandXReflection", true, "RandYReflection", true);

% Create augmented datastores (Note: images from ReadFcn are already 3-channel RGB)
augTrain = augmentedImageDatastore(inputSize, imdsTrain, "DataAugmentation", augmenter);
augTest  = augmentedImageDatastore(inputSize, imdsTest);
augVal   = augmentedImageDatastore(inputSize, imdsVal);

% **********************************************************************
% TRANSFER LEARNING USING RESNET-18 
% **********************************************************************
numClass = 2;
net = imagePretrainedNetwork('resnet18', NumClasses=numClass);

options = trainingOptions("sgdm", ...
    MiniBatchSize=16, ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=6, ...
    ValidationData=augVal, ...
    ValidationFrequency=5, ...
    Plots="training-progress", ...
    Verbose=false);

disp('Training ResNet-18 Model on Highlighted Defect Images...');
trainedNet = trainnet(augTrain, net, "crossentropy", options);
disp('Training Complete');

% Save updated model
save("trainNet.mat", "trainedNet");

% **********************************************************************
% FUNCTIONS
% **********************************************************************

function [finalLabel, confidenceScore, evidenceOverlay, evidenceMetrics, baselineDecision] = inspectPart(I)
    persistent net;
    if isempty(net)
        if isfile('trainNet.mat')
            data = load('trainNet.mat');
            fFields = fieldnames(data);
            net = data.(fFields{1});
        else
            error('trainNet.mat not found');
        end
    end

    % 1. Preprocessing & Grayscale
    if size(I, 3) == 3
        grayImg = rgb2gray(I);
    else
        grayImg = I;
    end
    
    roi = imresize(grayImg, [224, 224]);
    roi_contrast = adapthisteq(roi, "ClipLimit", 0.02);
    roi_clean = medfilt2(roi_contrast, [3, 3]);

    % 2. Defect Segmentation
    maskEvidence = defectEvidence(roi_clean);

    % 3. Calculate Traceability Metrics
    cc = bwconncomp(maskEvidence);
    numComponents = cc.NumObjects;
    stats = regionprops(maskEvidence, 'Area');
    if ~isempty(stats)
        maxArea = max([stats.Area]);
    else
        maxArea = 0;
    end
    areaRatio   = (nnz(maskEvidence) / numel(maskEvidence)) * 100;
    edgeMask    = edge(roi_clean, 'Sobel');
    edgeDensity = nnz(edgeMask) / numel(edgeMask);

    evidenceMetrics = struct(...
        'numComponents', numComponents, ...
        'maxArea',       maxArea, ...
        'areaRatio',     areaRatio, ...
        'edgeDensity',   edgeDensity);

    % Classical Rule-Based Decision
    if areaRatio > 0.5 || maxArea > 150
        baselineDecision = categorical("FAIL");
    else
        baselineDecision = categorical("PASS");
    end

    % 4. Create RGB Defect Overlay Image for ResNet-18
    overlayImg = cat(3, roi_clean, roi_clean, roi_clean);
    redChan = overlayImg(:,:,1);
    redChan(maskEvidence) = 255; % Highlight defects in bright red
    overlayImg(:,:,1) = redChan;

    % 5. Pass Overlaid Image to ResNet-18
    roiForNet = overlayImg;
    [finalLabel, confidenceScore] = classifyLocal(net, roiForNet);

    % 6. Burn Header Text into Display Overlay
    labelStr = sprintf('AI: %s (%.1f%%) | Rule: %s | Spots: %d', ...
        string(finalLabel), confidenceScore * 100, string(baselineDecision), numComponents);
    
    evidenceOverlay = insertText(overlayImg, [5, 5], labelStr, ...
        'FontSize', 10, ...
        'BoxColor', 'black', ...
        'BoxOpacity', 0.6, ...
        'TextColor', 'white');

    % Save Rejects
    if finalLabel == categorical("FAIL")
        if ~exist('rejects_folder', 'dir')
            mkdir('rejects_folder');
        end
        timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS-FFF');
        imwrite(evidenceOverlay, fullfile('rejects_folder', sprintf('reject_%s.png', timestamp)));
    end
end

function imgOut = customEnhaceReader(filename)
    I = imread(filename);
    if size(I, 3) == 3
        I = rgb2gray(I);
    end
    
    I_enhanced = adapthisteq(I, "ClipLimit", 0.02);
    I_clean = medfilt2(I_enhanced, [3, 3]);
    
    % Segment defects
    mask = defectEvidence(I_clean);
    
    % Create 3-channel RGB image with red overlay
    rgbImg = cat(3, I_clean, I_clean, I_clean);
    redChan = rgbImg(:,:,1);
    redChan(mask) = 255;
    rgbImg(:,:,1) = redChan;
    
    imgOut = rgbImg;
end

function maskEvidence = defectEvidence(rio)
    se = strel('disk', 4);
    topHat = imtophat(rio, se);
    botHat = imbothat(rio, se);
    scratchContrast = topHat + botHat;

    % Fixed: Binarize the contrast map directly
    T = adaptthresh(scratchContrast, 0.05, 'ForegroundPolarity', 'bright', 'NeighborhoodSize', 15);
    bwRaw = imbinarize(scratchContrast, T);

    bwClosed = imclose(bwRaw, se);
    minDefect = 5;
    maskEvidence = bwareaopen(bwClosed, minDefect);
end

function [aiLabel, aiScore] = classifyLocal(net, roiForNet)
    scores = predict(net, single(roiForNet));
    classNames = categorical(["FAIL", "PASS"]);
    [aiScore, maxIdx] = max(scores);
    aiLabel = classNames(maxIdx);
end

function buildMetadataCSV(filename, passDir, failDir) 
    paths = {}; labels = {};
    f1 = dir(fullfile(passDir, "*.png"));
    for i = 1:numel(f1)
        paths{end+1,1} = fullfile(passDir, f1(i).name);
        labels{end+1,1} = "PASS";
    end
    f2 = dir(fullfile(failDir, "*.png"));
    for i = 1:numel(f2)
        paths{end+1,1} = fullfile(failDir, f2(i).name);
        labels{end+1,1} = "FAIL";
    end
    outTable = table(string(paths), string(labels), 'VariableNames', {'FilePath', 'labels'});
    writetable(outTable, filename);
end