function exploreDataset(imds) % no output: function displays images
fprintf("========================\n")
fprintf("     DATASET SUMMARY\n")
fprintf("========================\n")

numImages = numel(imds.Files);
fprintf("Total Images: %d\n\n", numImages);

labelTable = countEachLabel(imds);
fprintf("Number of Classes: %d\n", height(labelTable));
disp(labelTable)

% Display a sample image from the dataset
img = readimage(imds, 1);
imshow(img);
title(string(imds.Labels(1)));

