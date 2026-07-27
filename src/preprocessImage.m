function [rgbImage, grayImage] = preprocessImage(inputImage) 

% Resize image to match network input size
rgbImage = imresize(inputImage, [224 224]);  

% Ensure image has three color channels
if size(rgbImage,3) == 1
    rgbImage = cat(3, rgbImage, rgbImage, rgbImage);
end

% Create grayscale version for classical vision
grayImage = rgb2gray(rgbImage);

end