function imds = loadDataset(folder) % folder: input, output: imds

if ~isfolder(folder)
    error("Dataset folder not founnd.")
end

imds = imageDatastore(folder, "IncludeSubfolders", true, "LabelSource",...
    "foldernames");

end
