function imds = loadDataset(datasetFolder)

if ~isfolder(datasetFolder)
    error("Dataset folder not found.")
end

trainFolder = fullfile(datasetFolder,"train");
testFolder  = fullfile(datasetFolder,"test");

imdsTrain = imageDatastore(trainFolder, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imdsTest = imageDatastore(testFolder, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imds = imageDatastore([imdsTrain.Files; imdsTest.Files]);

imds.Labels = categorical([imdsTrain.Labels; imdsTest.Labels]);

end