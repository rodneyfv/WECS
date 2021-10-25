
%%
close all
clear;
%%
im1 = imread('ImagesWithEllipsoidChanges/ImageEllipseSpeckle1.png');  
im2 = imread('ImagesWithEllipsoidChanges/ImageEllipseSpeckle2.png');  
im3 = imread('ImagesWithEllipsoidChanges/ImageEllipseSpeckle3.png');  
im4 = imread('ImagesWithEllipsoidChanges/ImageEllipseSpeckle4.png');  
%
figure
imshow(im1)
figure
imshow(im2)
figure
imshow(im3)
figure
imshow(im4)
%
% Images are too big, consider subsampling
subsamplingfactor = 2;
im1 = im1(1:subsamplingfactor:end,1:subsamplingfactor:end);
im2 = im2(1:subsamplingfactor:end,1:subsamplingfactor:end);
im3 = im3(1:subsamplingfactor:end,1:subsamplingfactor:end);
im4 = im4(1:subsamplingfactor:end,1:subsamplingfactor:end);
totalchanges = imread('GroundTruthEllipsoidChanges/TotalEllipseChanges.png');  
totalchanges = totalchanges(1:subsamplingfactor:end,1:subsamplingfactor:end);
%
figure
imshow(totalchanges)
title('Total changes')

%%
eps = .00001;
im1log = log(eps+double(im1));
im2log = log(eps+double(im2));
im3log = log(eps+double(im3));
im4log = log(eps+double(im4));
n=4;
imRef = im1log + im2log + im3log + im4log;
    imRef = imRef/n;
%% comparison of wavelet denoising with mean image

wname = 'db2';
J = 2;

tic
[X1,~,~,~] = swt2(im1log,J,wname);
    X1 = X1(:,:,J); % focus only on level J approximations ?
[X2,~,~,~] = swt2(im2log,J,wname);
    X2 = X2(:,:,J); % focus only on level J approximations ?
[X3,~,~,~] = swt2(im3log,J,wname);
    X3 = X3(:,:,J); % focus only on level J approximations ?
[X4,~,~,~] = swt2(im4log,J,wname);
    X4 = X4(:,:,J); % focus only on level J approximations ?
toc

%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d1 = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d1);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
% Not very fast is you want to use multicore processing
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d1);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pD2,pFA2]=ROCcurveNew(R,255*totalchanges); close
%
figure
imshow(R)
title('d(m): wavelet db2, J=2')


%% deep learning denoiser instead of using a wavelet transform
cd deeplearning/
net = denoisingNetwork('DnCNN'); % load a deep neural network that has learned to denoise more efficiently than wavelets !

% X1 = denoiseImageCPU(im1log,net);
% X2 = denoiseImageCPU(im2log,net);
% X3 = denoiseImageCPU(im3log,net);
% X4 = denoiseImageCPU(im4log,net);

tic
X1 = denoiseImageCPU(im1,net);
X2 = denoiseImageCPU(im2,net);
X3 = denoiseImageCPU(im3,net);
X4 = denoiseImageCPU(im4,net);
toc
cd ..
%%
X1 = double(X1);
X2 = double(X2);
X3 = double(X3);
X4 = double(X4);

%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d1 = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d1);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
%%%%%%%%%%%%%%%
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d1);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pDdeep,pFAdeep]=ROCcurveNew(R,255*totalchanges); close
%
figure
imshow(R)
title('d(m): deep learning denoiser')


%% Compare all results

mImage = figure;
hold on
title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(pFA2,pD2,'-.or')
plot(pFAdeep,pDdeep,'y','LineWidth',4)
legend('db2 WECS d(m), J=2', 'd(m): deep learning denoiser', 'Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('dm_comparison_wavelet_deepL.jpg'))



return


