% =========================================================================
% ISOLATED SCREW DEFECT SEGMENTATION
% =========================================================================
clc; clear; close all;

% 1. Load Image
csvFilename = 'factory_inspection_meta.csv';
metadataTable = readtable(csvFilename, 'Delimiter', ',');
failPaths = metadataTable.FilePath(metadataTable.labels == "FAIL");
randIdx = randi(numel(failPaths));
imgPath = failPaths{randIdx};

I_raw = imread(imgPath);
if size(I_raw, 3) == 3, grayImg = rgb2gray(I_raw); else, grayImg = I_raw; end
grayImg = imresize(grayImg, [224, 224]);

% =========================================================================
% STEP 1: ISOLATE THE SCREW (CREATE BACKGROUND MASK)
% =========================================================================
% The screw is darker than the bright background
screwRaw = grayImg < 200; 

% Fill holes inside the screw body
screwFilled = imfill(screwRaw, 'holes'); 

% Keep ONLY the single largest object (the screw) and drop background noise
screwROI = bwareafilt(screwFilled, 1); 

% Smooth the outer edges of the screw silhouette
screwROI = imclose(screwROI, strel('disk', 5));

% =========================================================================
% STEP 2: SEGMENT SCRATCHES STRICTLY INSIDE THE SCREW
% =========================================================================
% Enhance contrast inside the image
roi_contrast = adapthisteq(grayImg, "ClipLimit", 0.02);
rio = medfilt2(roi_contrast, [3, 3]);

% Morphological Top-Hat & Bottom-Hat filtering
se = strel('disk', 4);
scratchContrast = imtophat(rio, se) + imbothat(rio, se);

% Adaptive Threshold
sensitivityOffset = 0.05;
T = adaptthresh(scratchContrast, sensitivityOffset, ...
    'ForegroundPolarity', 'bright', ...
    'NeighborhoodSize', 15);
bwRaw = imbinarize(scratchContrast, T);

% --- CRITICAL STEP: Multiply by screwROI to destroy background noise ---
cleanMask = bwRaw & screwROI; 

% Clean up tiny specks (< 5 pixels)
finalMask = bwareaopen(cleanMask, 5);

% =========================================================================
% STEP 3: DISPLAY RESULTS
% =========================================================================
figure('Name', 'Isolated Screw Defect Segmentation', ...
    'Units', 'normalized', 'Position', [0.05, 0.2, 0.9, 0.5]);

% 1. Original
subplot(1, 3, 1);
imshow(grayImg);
title('1. Original Image', 'FontSize', 11, 'FontWeight', 'bold');

% 2. Screw Silhouette ROI
subplot(1, 3, 2);
imshow(screwROI);
title('2. Isolated Screw Silhouette (ROI)', 'FontSize', 11, 'FontWeight', 'bold');

% 3. Final Clean Defect Mask
subplot(1, 3, 3);
imshow(finalMask);
title('3. Clean Defect Mask (Background Removed)', 'FontSize', 11, 'FontWeight', 'bold');