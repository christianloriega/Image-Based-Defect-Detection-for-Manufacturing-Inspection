function [overlay, metrics, baselineDecision] = extractEvidence(grayImage)
% extractEvidence Runs the classical defect detectors and combines their
% evidence into one baseline decision, overlay, and metrics structure.

% Manipulated-Front Detection

[frontOverlay, frontMetrics, frontDecision] = ...
    detectManipulatedFront(grayImage);

% Scratch-Neck Detection

[scratchDecision, scratchMetrics, scratchOverlay] = ...
    detectScratchNeck(grayImage);

% Ensure Overlays Are RGB

if size(frontOverlay, 3) == 1
    frontOverlay = cat(3, frontOverlay, frontOverlay, ...
        frontOverlay);
end

if size(scratchOverlay, 3) == 1
    scratchOverlay = cat(3, scratchOverlay, scratchOverlay, ...
        scratchOverlay);
end

% Determine Strong Classical Evidence

strongFrontEvidence = frontDecision == "FAIL" && ...
    frontMetrics.FailedRules >= 2;

strongScratchEvidence = scratchDecision == "FAIL" && ...
    scratchMetrics.ScratchCount >= 1 && ...
    scratchMetrics.LargestScratchArea >= 22 && ...
    scratchMetrics.ScratchArea >= 38 && ...
    scratchMetrics.ScratchRatio >= 0.016;

% Resolve Individual Detector Decisions

if strongFrontEvidence
    resolvedFrontDecision = "FAIL";
else
    resolvedFrontDecision = "PASS";
end

if strongScratchEvidence
    resolvedScratchDecision = "FAIL";
else
    resolvedScratchDecision = "PASS";
end

% Resolve Overall Classical Decision

if strongFrontEvidence || strongScratchEvidence
    baselineDecision = "FAIL";
else
    baselineDecision = "PASS";
end

% Select Defect Type and Evidence Overlay

if strongFrontEvidence && strongScratchEvidence

    detectedDefect = "multiple";

    % Display the manipulated-front overlay when both detectors trigger.
    % Both raw detector results are still preserved in the metrics.
    overlay = frontOverlay;

elseif strongFrontEvidence

    detectedDefect = "manipulated_front";
    overlay = frontOverlay;

elseif strongScratchEvidence

    detectedDefect = "scratch_neck";
    overlay = scratchOverlay;

else

    detectedDefect = "none";

    % Keep the aligned front-detector image for consistent display.
    overlay = frontOverlay;

end

% Combine Evidence Metrics

metrics = struct( ...
    "DetectedDefect", detectedDefect, ...
    "FrontDecision", resolvedFrontDecision, ...
    "ScratchDecision", resolvedScratchDecision, ...
    "RawFrontDecision", frontDecision, ...
    "RawScratchDecision", scratchDecision, ...
    "TipWidth", frontMetrics.TipWidth, ...
    "TipArea", frontMetrics.TipArea, ...
    "MaxSlope", frontMetrics.MaxSlope, ...
    "SlopeVariation", frontMetrics.SlopeVariation, ...
    "CenterDeviation", frontMetrics.CenterDeviation, ...
    "FailedRules", frontMetrics.FailedRules, ...
    "ScratchArea", scratchMetrics.ScratchArea, ...
    "ScratchCount", scratchMetrics.ScratchCount, ...
    "ScratchRatio", scratchMetrics.ScratchRatio, ...
    "LargestScratchArea", scratchMetrics.LargestScratchArea);

end