function [rgbImage, grayImage] = preprocessImage(inputImage) 

% Resize image to match ResNet-18 input of 224 x 224, showing more pixels
rgbImage = imresize(inputImage, [224 224]);  

% Convert to grayscale for classical vision
if size(rgbImage,3) == 3
    grayImage = im2gray(rgbImage);
else
    grayImage = rgbImage;
end


end