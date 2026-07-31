function distortedImage = applyDistortion(inputImage, distortionType)

switch lower(distortionType)

    case "none"
        distortedImage = inputImage;

    case "brightness"
        distortedImage = imadjust(inputImage);
    
    case "blur"
        distortedImage = imgaussfilt(inputImage, 2);

    case "noise"
        distortedImage = imnoise(inputImage, "gaussian", 0, 0.002);

    otherwise
        error("Unknown distortion type.")

end

end

