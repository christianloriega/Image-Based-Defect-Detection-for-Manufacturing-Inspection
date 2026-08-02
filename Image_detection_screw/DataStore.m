
% **********************************************************************
% INDUSTRIAL IMAGE INSPECTOR USING RESNET 18 
%*****************************************************************
clc; clear; close all;

% Data exploration and setup 

% folder paths are extracted and placed in varaibles 
passDir = fullfile('data', 'MVTec AD', 'screw', 'test', 'good');
failDir = fullfile('data', 'MVTec AD', 'screw', 'test', 'scratch_neck');

% make a cvs file 
csvFilename = ("factory_inspection_meta.csv");
if isfile(csvFilename)
    delete(csvFilename)
end
% build the new csv file with the file paths of the imgaes we want to send 
buildMetadataCSV(csvFilename,passDir,failDir);
% we are making a table wiht the csv 

metadataTable = readtable(csvFilename,'Delimiter',',');



% making a datastrore so we can acces images 
imds = imageDatastore(metadataTable.FilePath,'Labels',categorical(metadataTable.labels));




disp('Dataset Exploration: Label Counts & Balance Summary:');
disp(countEachLabel(imds));






%******************************************************************
% STEP 2: STANDARIZE IMAGE: using imread(),imresize(),rgb2gray()
% SUB STEP 1: IMAGE PROCESSING: apply preprocessing such as light
% correction imflatfeild or background estimate with imgaussflit
% apply contrast adapthisteq, denoising:medfilt2 or imgaussflit 
% save results to imshowpair for debugging 


% split the data into validation test and training sets 
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.70, 0.15, 0.15, 'randomized');

save('dataset_workspace.mat', 'imdsTrain', 'imdsVal', 'imdsTest');



% since resnet 18 takes input size of 244 by 244 we must resize the images
% into that format. we also need to input 3 of these since the resnet 18
% takes 3 values of RGB we need to imput it 3 times so we do not get in
% unput eorrror 

inputSize = [224,224];




% make an agumentaion data sotre to store the modifications 
% flips the images in the x and y direction that way the model understands
% that a defect is still a defect regarless of the oreintation of the
% object 
augmenter = imageDataAugmenter("RandXReflection",true,"RandYReflection",true)

% applying grey scale and grey scale to the split images 
augTrain = augmentedImageDatastore(inputSize,imdsTrain,"DataAugmentation",augmenter,"ColorPreprocessing","gray2rgb")
augTest = augmentedImageDatastore(inputSize,imdsTest,"ColorPreprocessing","gray2rgb")
augVal = augmentedImageDatastore(inputSize,imdsVal,"ColorPreprocessing","gray2rgb")



%****************************************************************************
% APPLYING PREPROCESSING 
%*************************************************************************************

% uses a function to apply adapthisque and medflit2 to each of the data
% images
imdsTrain.ReadFcn = @customEnhaceReader;
imdsVal.ReadFcn = @customEnhaceReader;
imdsTest.ReadFcn = @customEnhaceReader;




% 4. INSPECTION EVIDENCE & TRACEABILITY METRICS (Debug & Overlay Loop)
% Filter to inspect the first few FAIL images using our defect evidence recipe

failIndices = find(imdsTrain.Labels == 'FAIL');
numToDisplay = min(4, length(failIndices));

figure('Name', 'Inspection Evidence & Traceability Overlays', 'Position', [100 100 1000 800]);













%**********************************************************************
% STEP 3: TRANSFER LEARNING USING RESNET-18 
% **********************************************************************
numClass = 2;
net = imagePretrainedNetwork('resnet18', NumClasses=numClass);

% Define training options
options = trainingOptions("sgdm", ...
    MiniBatchSize=16, ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=6, ...
    ValidationData=augVal, ...
    ValidationFrequency=5, ...
    Plots="training-progress", ...
    Verbose=false);

% Fine-tune the network 
disp('Training ResNet-18 Model...');
trainedNet = trainnet(augTrain, net, "crossentropy", options);
disp('Training Complete');

% Reset test datastore to the beginning 
reset(augTest);

% Read an image batch directly from augTest 
testData = read(augTest);
roiForNet = testData.input{1};

% Call the classify deliverable function 
[aiLabel, aiScore] = classify(trainedNet, roiForNet);

% Display prediction result 
fprintf('AI prediction: %s | Confidence: %.2f%%\n', string(aiLabel), aiScore * 100);

% FIX: Save fine-tuned "trainedNet" (NOT untrained "net") to "trainNet.mat"
save("trainNet.mat", "trainedNet");















% Read any sample image
I = imread(imdsTest.Files{1});

% Call function with the exact requested signature!
%[finalLabel, confidenceScore, evidenceOverlay, evidenceMetrics, baselineDecision] = inspectPart(I);

% Display Overlay
%figure;
%imshow(evidenceOverlay);
%title(sprintf('Final AI Label: %s | Baseline Rule Label: %s', string(finalLabel), string(baselineDecision)));









% **********************************************************************
% LOCAL FUNCTIONS 
%************************************************************



function [aiLabel,aiScore] = classify(net, roiForNet)
% Predict class probabilities 
scores = predict(net,single(roiForNet));

% Map class labels 
classNames = categorical(["FAIL","PASS"]);

% Extract predictive labels 
[aiScore,maxIdx] = max(scores);
aiLabel = classNames(maxIdx);
end 




function buildMetadataCSV(filename,passDir,failDir) 
% thes varibles will store the file path for the specific images 
paths = {}; labels = {};
f1 = dir(fullfile(passDir,"*.png"));
for i = 1: numel(f1)
    paths{end+1,1} = fullfile(passDir, f1(i).name);
    labels{end+1,1} = "PASS";
end

f2 = dir(fullfile(failDir,"*.png"));
for i = 1: numel(f2)
    paths{end+1,1} = fullfile(failDir, f2(i).name);
    labels{end+1,1} = "FAIL";
end

outTable = table(string(paths), string(labels), 'VariableNames',{'FilePath','labels'} );
writetable(outTable,filename);

end

% custom funciton that takes an image data set adn applies a filter to make
% the gray scale more apperent in shades so that we can see the faults alot
%

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
% FUNCTION USED TO APPLY SEGMENTATION FOR EVIDANCE OVERLAY 


% this function incorparates everything we have been practicing it takes
% imgaes applies rgb,resizing, imageprocessing, splits the data and
% test,trains the model and displayes the results 
% finalLabel - this is the AI's decision of PASS OR FAIL 
% confidance score - the AI confidance in its decision 
% evidiance overlays - this is where we apply the overlay techniques to
% make suppicios areas stand out 
% 



