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

%% Convert Labels to PASS/FAIL

labels = imds.Labels;

labels(labels == "good") = "PASS";
labels(labels ~= "PASS") = "FAIL";
imds.Labels = categorical(labels);

% Convert back to categorical and remove all unused categories.

imds.Labels = removecats(categorical(labels));

countEachLabel(imds);

%% Split Dataset

% 80% Training
% 10% Validation
% 10% Testing

[imdsTrain, imdsTemp] = splitEachLabel(imds,0.8,"randomized");

[imdsVal, imdsTest] = splitEachLabel(imdsTemp,0.5,"randomized");

% Dataset Summary

trainTable = countEachLabel(imdsTrain);
testTable  = countEachLabel(imdsTest);
valTable   = countEachLabel(imdsVal);

fprintf("Training Data:\n");
disp(trainTable);

fprintf("Test Data:\n");
disp(testTable);

fprintf("Validation Data:\n");
disp(valTable);

%% Network Input Size

% ResNet-18 is a pretrained neural network trained on ImageNet and expects
% RGB images with a size of 224 x 224 pixels. ( [224 224 3])\

inputSize = [224 224 3];

%% Data Augmentation

% Augmentation creates small random transformation to improve robustness of
% the system. Each image will be slightly different.

augmentor = imageDataAugmenter(RandRotation= [-10 10], RandXTranslation= [-5 5], ...
    RandYTranslation= [-5 5]);

% Create augmented image datastore
augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation',...
    augmentor, 'ColorPreprocessing', 'gray2rgb');
augVal = augmentedImageDatastore(inputSize, imdsVal, 'ColorPreprocessing', 'gray2rgb');
augTest = augmentedImageDatastore(inputSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');

%% Load Pretrained Network

% Load the pretrained ResNet-18 network.
net = imagePretrainedNetwork("resnet18", "NumClasses", 2); % This returns a dlnetwork
analyzeNetwork(net)

%% Configure Training Options

% Training options control how a network learns during transfer learning.
% Such settings are the optimizer, number of training epochs, validation
% data, and visualization of the training process. 

options = trainingOptions("adam", InitialLearnRate= 1e-4, MaxEpochs= 20, MiniBatchSize= 32, ...
    Shuffle= "every-epoch", ValidationData= augVal, ...
    ValidationFrequency= 10, Plots= "training-progress", Verbose= true);

% Adam (Adaptive Moment Estimation): automatically adjusts how much each weight changes during training.
% Instead of using one fixed learning behavior for every weight, it adapts as it learns.

% MaxEpochs: Maximum times the network sees the entire training dataset.

% MiniBatchSize: Number of images processed before updating the network
% weights.

% Shuffle: Shuffle the training images, in this case every epoch.

% ValidationData: Uses augVal as validation dataset to monitor performance.

% ValidationFrequency: Check validation every 10 iterations.

% Plots: Display live training progress window.

% Verbose: Print training information to the Command Window.

%% Train Network

% Cross-entropy is the standard loss function for classification problems.

net = trainnet(augTrain, net, "crossentropy", options);

%% Test Network

% Measure the classification accuracy using the unseen mixed testing data.

accuracy = testnet(net, augTest, "accuracy");

fprintf("Test Accuracy: %.2f%%\n", accuracy);

% Predict Test Set Labels

scores = minibatchpredict(net, augTest);

predictedLabels = scores2label(scores, categories(imdsTest.Labels));

% True Labels

trueLabels = imdsTest.Labels;

% Performance Metrics

% Confusion Matrix

figure;
C = confusionmat(trueLabels, predictedLabels);
confusionchart(trueLabels, predictedLabels);

TN = C(1,1);
FP = C(1,2);
FN = C(2,1);
TP = C(2,2);

precision = TP / (TP + FP);
recall = TP / (TP + FN);

f1score = 2 * (precision * recall) / (precision + recall);

fprintf("Precision: %.3f\n", precision);
fprintf("Recall: %.3f\n", recall);
fprintf("F1 Score: %.3f\n", f1score);

title("Confusion Matrix - Test Set");

%% Save Trained Network

% Save the trained network so it can be loaded later
% without retraining the model.

modelFile = fullfile(projectRoot, ...
    "models", ...
    "resnet18_PASS_FAIL.mat");

save(modelFile, "net");

fprintf("Trained network saved to:\n%s\n", modelFile);