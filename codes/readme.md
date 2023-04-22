# Description of the Matlab codes we use to implement and study WECS.

The file 'generate_synthetic_sequence.m' generates a sequence of simulated images and the files 'SeqEllipseChangeAnalysis' reproduce the analysis of these simulated data.

## abscor.m
computes the absolute value of the Pearson correlation between two vectors

## F1Scorecurve.m
Produces a curve of F1 scores for a sequence of threshold values in (0,1).

## F1score.m
F1 score of an image given a ground truth image.

## Forest_data_analysis.m
Code used to analyze the forest data.

## generate_synthetic_seq_small_change.m
Code to analyze the performance of WECS for small changes.

## generate_synthetic_sequence.m
Simulates a sequence of SAR images where ellipses correspond to changes.

## kittler.m
Kittler-Illingworth thresholding method.

## MorletWaveletKernel.m
Morlet wavelet function.

## Python
Folder with python codes to apply the following change methods: CVA, MAD, PAC K-means, and ISFA. Codes are in the subfolder Methodology/Traditional.

## ROCcurveNew.m
Produces a ROC curve for a sequence of threshold values in (0,1).

## save_tiff_image.m
Saves an image in the tiff format.

## SeqEllipseChangeAnalysis_methods.m
Code to evaluate different methods of change detection.

## SeqEllipseChangeAnalysis_wavelets.m
Code to evaluate different resolution levels and wavelet families when we apply WECS.

## Small_ChangeAnalysis_methods.m
Code to evaluate the performance of WECS when the images only have small changes.

## wecs.m
Code to apply WECS. Dimensions of the images must be a power of 2.

