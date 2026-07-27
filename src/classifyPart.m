function [aiLabel, aiScore] = classifyPart(net, inputImage)
% Inputs:
% net - trained ResNet-18
% roiForNet - Single image
% Outputs:
% aiLabel - Predicted class (PASS/FAIL)
% aiScore - Confidence score

% ResNet-18 expects an RGB image, so only rgbImage is used.
[rgbImage, ~] = preprocessImage(inputImage);

imds = augmentedImageDatastore([224 224 3], rgbImage);

scores = minibatchpredict(net, imds);

aiLabel = scores2label(scores, ["PASS", "FAIL"]);

aiScore = max(scores);

end

