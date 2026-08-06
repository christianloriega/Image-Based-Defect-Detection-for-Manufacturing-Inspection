function exploreDataset(imds)
% Displays dataset summary and random sample images.

fprintf("===========================\n")
fprintf("     DATASET SUMMARY\n")
fprintf("===========================\n\n")

numImages = numel(imds.Files);
fprintf("Total Images: %d\n\n", numImages);

labelTable = countEachLabel(imds);

fprintf("Number of Classes: %d\n\n", height(labelTable));

labelTable.Percent = 100 * labelTable.Count / numImages;
disp(labelTable)

% Combined Figure

figure("Name", "Dataset Overview", "Position", [100 100 1500 700]);

mainLayout = tiledlayout(3,4,"Padding", "compact", "TileSpacing", ...
    "compact");

% Class Distribution

nexttile(mainLayout, 1, [3 1])

bar(labelTable.Count)

xticks(1:height(labelTable))
xticklabels(string(labelTable.Label))
xtickangle(45)

xlabel("Class")
ylabel("Number of Images")
title("Class Distribution")
grid on

% Random Dataset Samples

numShow = min(9, numImages);
randomIndices = randperm(numImages, numShow);

for i = 1:numShow

    nexttile(mainLayout)

    idx = randomIndices(i);

    imshow(readimage(imds, idx))

    title( ...
        strrep(string(imds.Labels(idx)), "_", " "), ...
        "FontSize", 10)

end

end