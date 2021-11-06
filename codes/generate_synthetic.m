
% Code to generate the synthetic images 

%%
clear all
close all
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading images 
im1=imread('../figs/GroundTruthEllipsoidChanges/ellipse1.tif');
im1=double(~im1); % boolean ~ used to represent change point with 1
im2=imread('../figs/GroundTruthEllipsoidChanges/ellipse2.tif');
im2=double(~im2); 
im3=imread('../figs/GroundTruthEllipsoidChanges/ellipse3.tif');
im3=double(~im3); 
im4=imread('../figs/GroundTruthEllipsoidChanges/ellipse4.tif');
im4=double(~im4); 

% noise level used
sig = 1/10;

% adding Gaussian noise to images
im1_obs = 1+im1/2 + sig*randn(height(im1), width(im1));
im2_obs = 1+(im2+im1)/2 + sig*randn(height(im2), width(im2));
im3_obs = 1+(im1+im2+im3)/2 + sig*randn(height(im3), width(im3));
im4_obs = 1+(im1+im2+im3+im4)/2 + sig*randn(height(im4), width(im4));

colormap(gray(256)); imagesc(im2_obs)
colormap(gray(256)); imagesc(im3_obs)
colormap(gray(256)); imagesc(im4_obs)
colormap(gray(256)); imagesc(im1_obs+im2_obs+im3_obs+im4_obs)

% saving the synthetic images
mImage = cast(100*(im1_obs),'uint8');
imshow(mImage)
%imwrite(mImage,sprintf('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle1.png'))

mImage = cast(100*(im2_obs),'uint8');
imshow(mImage)
%imwrite(mImage,sprintf('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle2.png'))

mImage = cast(100*(im3_obs),'uint8');
imshow(mImage)
%imwrite(mImage,sprintf('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle3.png'))

mImage = cast(100*(im4_obs),'uint8');
imshow(mImage)
%imwrite(mImage,sprintf('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle4.png'))

