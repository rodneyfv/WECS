
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

%
close all
clear;
%

rng(2021)

% Subsampling of images
subsamplingfactor = 2;
% noise level used
sig = 1;
% size of the sequence (must be multiple of 4)
n = 80;

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

%sprintf('signal to noise ratios')
%sprintf('im1: %f',sqrt(sum(sum(im1.^2,1))/(n1*n2))/sig)
%sprintf('im2: %f',sqrt(sum(sum(im2.^2,1))/(n1*n2))/sig)
%sprintf('im3: %f',sqrt(sum(sum(im3.^2,1))/(n1*n2))/sig)
%sprintf('im4: %f',sqrt(sum(sum(im4.^2,1))/(n1*n2))/sig)

% array of true images
mI = zeros(n1,n2,4);
mI(:,:,1) = im1; mI(:,:,2) = im2;
mI(:,:,3) = im3; mI(:,:,4) = im4;
%clear im1 im2 im3 im4

% reading the first image
t = Tiff(strcat('../../Images/timeSeries/ascending/2015-12-26.tif'),'r');
Y = read(t);

%imagesc(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2))
t = Tiff(strcat('../figs/imgROIs.tif'),'r');
change_map = read(t);

% estimates of shape and scale parameters of a Gamma distribution fitted to
% the observations in the first forest image.

% estimates from channel 1
im = Y(:,:,1) - min(min(Y(:,:,1)));
im_id = change_map(:,:,1)==255; % change pixels
im_mean = mean(im(im_id));
im_var = var(im(im_id));
a_change_c1 = (im_mean^2)/im_var;
b_change_c1 = im_var/im_mean;

im_id = change_map(:,:,2)==255; % non-change pixels
im_mean = mean(im(im_id));
im_var = var(im(im_id));
a_nochange_c1 = (im_mean^2)/im_var;
b_nochange_c1 = im_var/im_mean;

% estimates from channel 2
im = Y(:,:,2) - min(min(Y(:,:,2)));
im_id = change_map(:,:,1)==255; % change pixels
im_mean = mean(im(im_id));
im_var = var(im(im_id));
a_change_c2 = (im_mean^2)/im_var;
b_change_c2 = im_var/im_mean;

im_id = change_map(:,:,2)==255; % non-change pixels
im_mean = mean(im(im_id));
im_var = var(im(im_id));
a_nochange_c2 = (im_mean^2)/im_var;
b_nochange_c2 = im_var/im_mean;


% array to store sequence of images
mY0 = zeros(n1, n2, 2, n);
% Sequence of images with noise. Noisy images are simulated with a Gamma
% distribution.
cont = 1;
while cont<=80
    for m=1:4
        im = mI(:,:,m);
        im_id = mI(:,:,m) > 0; % change pixels
        im(im_id) = im(im_id) + gamrnd(a_change_c1, b_change_c1, size(im(im_id)));        
        im_id = mI(:,:,m) == 0; % non-change pixels
        im(im_id) = im(im_id) + gamrnd(a_nochange_c1, b_nochange_c1, size(im(im_id)));
        % simulated image for channel 1
        mY0(:,:,1,cont) = im;
        
        im = mI(:,:,m);
        im_id = mI(:,:,m) > 0; % change pixels
        im(im_id) = im(im_id) + gamrnd(a_change_c2, b_change_c2, size(im(im_id)));        
        im_id = mI(:,:,m) == 0; % non-change pixels
        im(im_id) = im(im_id) + gamrnd(a_nochange_c2, b_nochange_c2, size(im(im_id)));
        % simulated image for channel 2
        mY0(:,:,2,cont) = im;
        
        cont = cont + 1;
    end
end

% to simulate spatial dependence, we run a sliding windom where each pixel
% is the mean of neighboring values
fun = @(x) mean(x(:));

% array to store sequence of images
mY = zeros(n1, n2, 2, n);
% Sequence of images with noise. Noisy images are simulated with a Gamma
% distribution.
cont = 1;
while cont<=80
    mY(:,:,1,cont) = nlfilter(mY0(:,:,1,cont),[3 3], fun);
    mY(:,:,2,cont) = nlfilter(mY0(:,:,2,cont),[3 3], fun);
    cont = cont + 1;
end
clear mY0;

% the whole time series of simulated images
save('../../Images/simulated/simulated_sequence.mat','mY');
% saving the first and last images of the time series for bi-temporal
% methods
save_tiff_image(mY(:,:,:,1), sprintf('./Python/Dataset/simulated_image_t1.tiff'));
save_tiff_image(mY(:,:,:,80), sprintf('./Python/Dataset/simulated_image_t80.tiff'));


