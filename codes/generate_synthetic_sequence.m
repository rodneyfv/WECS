
% Code to generate a sequence synthetic images 

%%
% Parameters that must be specified

% % Images are too big, consider subsampling
% subsamplingfactor = 2;
% % noise level used
% sig = 1;
% % size of the sequence (must be multiple of 4)
% n = 80;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loading images 
im1=imread('../figs/GroundTruthEllipsoidChanges/ellipse1.tif');
% boolean ~ used to represent change point with 1
im1=double(~im1(1:subsamplingfactor:end,1:subsamplingfactor:end));

im2=imread('../figs/GroundTruthEllipsoidChanges/ellipse2.tif');
im2=double(~im2(1:subsamplingfactor:end,1:subsamplingfactor:end)); 
im3=imread('../figs/GroundTruthEllipsoidChanges/ellipse3.tif');
im3=double(~im3(1:subsamplingfactor:end,1:subsamplingfactor:end)); 
im4=imread('../figs/GroundTruthEllipsoidChanges/ellipse4.tif');
im4=double(~im4(1:subsamplingfactor:end,1:subsamplingfactor:end)); 
% number of rows and columns
n1 = size(im1,1);
n2 = size(im1,2);


sprintf('signal to noise ratios')
sprintf('im1: %f',sqrt(sum(sum(im1.^2,1))/(n1*n2))/sig)
sprintf('im2: %f',sqrt(sum(sum(im2.^2,1))/(n1*n2))/sig)
sprintf('im3: %f',sqrt(sum(sum(im3.^2,1))/(n1*n2))/sig)
sprintf('im4: %f',sqrt(sum(sum(im4.^2,1))/(n1*n2))/sig)

% array of true images
mI = zeros(n1,n2,4);
mI(:,:,1) = im1; mI(:,:,2) = im2;
mI(:,:,3) = im3; mI(:,:,4) = im4;
clear im1 im2 im3 im4

% array to store sequence of images
mY = zeros(n1, n2, n);
% sequence of images with noise
cont = 1;
while cont<=80
    for m=1:4
        mY(:,:,cont) = mI(:,:,m) + sig*randn(n1, n2);
        cont = cont + 1;
    end
end


