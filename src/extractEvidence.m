function [overlay, metrics, baselineDecision] = extractEvidence(grayImage)

% Detecting manipulated front defects on screws.
[overlay, metrics, baselineDecision] = detectManipulatedFront(grayImage);

% Later as we add more defect functions, we'll add numbers to outputs.
end