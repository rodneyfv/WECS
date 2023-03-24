
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
im1=imread('../figs/GroundTruthEllipsoidChanges/ellipse3.tif');
% boolean ~ used to represent change point with 1
im1=double(~im1(1:subsamplingfactor:end,1:subsamplingfactor:end));

% number of rows and columns
n1 = size(im1,1);
n2 = size(im1,2);

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
    im = im1;
    im_id = im1 > 0; % change pixels
    im(im_id) = im(im_id) + gamrnd(a_change_c1, b_change_c1, size(im(im_id)));        
    im_id = im1 == 0; % non-change pixels
    im(im_id) = im(im_id) + gamrnd(a_nochange_c1, b_nochange_c1, size(im(im_id)));
    % simulated image for channel 1
    mY0(:,:,1,cont) = im;

    im = im1;
    im_id = im1 > 0; % change pixels
    im(im_id) = im(im_id) + gamrnd(a_change_c2, b_change_c2, size(im(im_id)));        
    im_id = im1 == 0; % non-change pixels
    im(im_id) = im(im_id) + gamrnd(a_nochange_c2, b_nochange_c2, size(im(im_id)));
    % simulated image for channel 1
    mY0(:,:,2,cont) = im;
    
    cont = cont + 1;
    for m=1:3
        % simulated image for channel 1
        mY0(:,:,1,cont) = gamrnd(a_nochange_c1, b_nochange_c1, n1, n2);
        % simulated image for channel 2
        mY0(:,:,2,cont) = gamrnd(a_nochange_c2, b_nochange_c2, n1, n2);
        
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
save('../../Images/simulated/simulated_sequence_small_change.mat','mY');
% saving the first and last images of the time series for bi-temporal
% methods
save_tiff_image(mY(:,:,:,1), sprintf('./Python/Dataset/simulated_image_small_change_t1.tiff'));
save_tiff_image(mY(:,:,:,80), sprintf('./Python/Dataset/simulated_image_small_change_t80.tiff'));


