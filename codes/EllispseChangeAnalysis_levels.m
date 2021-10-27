
% Evaluation of different levels when mean approximation coefficients are used

%%
close all
clear;
%%
im1 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle1.png');
im2 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle2.png');  
im3 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle3.png');  
im4 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle4.png');  
%

% Images are too big, consider subsampling
subsamplingfactor = 2;
im1 = im1(1:subsamplingfactor:end,1:subsamplingfactor:end);
im2 = im2(1:subsamplingfactor:end,1:subsamplingfactor:end);
im3 = im3(1:subsamplingfactor:end,1:subsamplingfactor:end);
im4 = im4(1:subsamplingfactor:end,1:subsamplingfactor:end);
totalchanges = imread('../figs/GroundTruthEllipsoidChanges/TotalEllipseChanges.png');  
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

% decomposition level fixed
wname = 'db2';

%% Case 1 : reference image 

J = 1;

[X1,~,~,~] = swt2(im1log,J,wname);
    X1 = X1(:,:,J); % focus only on level J approximations ?
[X2,~,~,~] = swt2(im2log,J,wname);
    X2 = X2(:,:,J); % focus only on level J approximations ?
[X3,~,~,~] = swt2(im3log,J,wname);
    X3 = X3(:,:,J); % focus only on level J approximations ?
[X4,~,~,~] = swt2(im4log,J,wname);
    X4 = X4(:,:,J); % focus only on level J approximations ?

% % mean approximation coefficients
% imRef_wave = X1 + X2 + X3 + X4;
%     imRef_wave = imRef_wave/n;
%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
% Not very fast is you want to use multicore processing
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pD1,pFA1]=ROCcurveNew(R,255*totalchanges); close
[vp,vF1_J1,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db2 wavelet - J=1')

%% 

J = 2;

[X1,~,~,~] = swt2(im1log,J,wname);
    X1 = X1(:,:,J); % focus only on level J approximations ?
[X2,~,~,~] = swt2(im2log,J,wname);
    X2 = X2(:,:,J); % focus only on level J approximations ?
[X3,~,~,~] = swt2(im3log,J,wname);
    X3 = X3(:,:,J); % focus only on level J approximations ?
[X4,~,~,~] = swt2(im4log,J,wname);
    X4 = X4(:,:,J); % focus only on level J approximations ?

% % mean approximation coefficients
% imRef_wave = X1 + X2 + X3 + X4;
%     imRef_wave = imRef_wave/n;
%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
% Not very fast is you want to use multicore processing
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pD2,pFA2]=ROCcurveNew(R,255*totalchanges); close
[vp,vF1_J2,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db2 wavelet - J=2')

%% 

J = 3;

[X1,~,~,~] = swt2(im1log,J,wname);
    X1 = X1(:,:,J); % focus only on level J approximations ?
[X2,~,~,~] = swt2(im2log,J,wname);
    X2 = X2(:,:,J); % focus only on level J approximations ?
[X3,~,~,~] = swt2(im3log,J,wname);
    X3 = X3(:,:,J); % focus only on level J approximations ?
[X4,~,~,~] = swt2(im4log,J,wname);
    X4 = X4(:,:,J); % focus only on level J approximations ?

% % mean approximation coefficients
% imRef_wave = X1 + X2 + X3 + X4;
%     imRef_wave = imRef_wave/n;
%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
% Not very fast is you want to use multicore processing
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pD3,pFA3]=ROCcurveNew(R,255*totalchanges); close
[vp,vF1_J3,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db2 wavelet - J=3')

%% 

J = 4;

[X1,~,~,~] = swt2(im1log,J,wname);
    X1 = X1(:,:,J); % focus only on level J approximations ?
[X2,~,~,~] = swt2(im2log,J,wname);
    X2 = X2(:,:,J); % focus only on level J approximations ?
[X3,~,~,~] = swt2(im3log,J,wname);
    X3 = X3(:,:,J); % focus only on level J approximations ?
[X4,~,~,~] = swt2(im4log,J,wname);
    X4 = X4(:,:,J); % focus only on level J approximations ?

% % mean approximation coefficients
% imRef_wave = X1 + X2 + X3 + X4;
%     imRef_wave = imRef_wave/n;
%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
% Not very fast is you want to use multicore processing
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pD4,pFA4]=ROCcurveNew(R,255*totalchanges); close
[vp,vF1_J4,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db2 wavelet - J=4')

%% 

J = 5;

[X1,~,~,~] = swt2(im1log,J,wname);
    X1 = X1(:,:,J); % focus only on level J approximations ?
[X2,~,~,~] = swt2(im2log,J,wname);
    X2 = X2(:,:,J); % focus only on level J approximations ?
[X3,~,~,~] = swt2(im3log,J,wname);
    X3 = X3(:,:,J); % focus only on level J approximations ?
[X4,~,~,~] = swt2(im4log,J,wname);
    X4 = X4(:,:,J); % focus only on level J approximations ?

% % mean approximation coefficients
% imRef_wave = X1 + X2 + X3 + X4;
%     imRef_wave = imRef_wave/n;
%
D1 = (X1 - imRef).^2;
D2 = (X2 - imRef).^2;
D3 = (X3 - imRef).^2;
D4 = (X4 - imRef).^2;
d = [sum(D1(:)) sum(D2(:)) sum(D3(:)) sum(D4(:))];
disp('Expressiveness of changes per images');
disp(d);
%
[NbRows, NbCols] = size(D1);
NbPixels = NbRows*NbCols;
% Not very fast is you want to use multicore processing
Dtensor1 = cat(2,reshape(D1,NbPixels,1),reshape(D2,NbPixels,1),reshape(D3,NbPixels,1),reshape(D4,NbPixels,1)); % concatenation of sptial and temporal
R = zeros(1, NbPixels);
parfor k=1:NbPixels
        temp = corrcoef(Dtensor1(k,:),d);
        R(k) = temp(1,2);
end
R = R./max(R(:));
R = reshape(R,NbRows, NbCols);
[pD5,pFA5]=ROCcurveNew(R,255*totalchanges); close
[vp,vF1_J5,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db2 wavelet - J=5')

%% Compare all results

mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(pFA1,pD1,':+')
plot(pFA2,pD2,'-.or')
plot(pFA3,pD3,':*')
plot(pFA4,pD4,':^')
plot(pFA5,pD5,'g-d')
legend('$J=1$', '$J=2$', '$J=3$','$J=4$', '$J=5$',...
    'interpreter','latex','Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/levels_comparison.jpg'))

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 13)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 13)
axis([0 1 0 0.7]);
axis square
plot(vp,vF1_J1,':+')
plot(vp,vF1_J2,'-.or')
plot(vp,vF1_J3,':*')
plot(vp,vF1_J4,':^')
plot(vp,vF1_J5,'g-d')
legend('$J=1$', '$J=2$', '$J=3$','$J=4$', '$J=5$',...
    'interpreter','latex','Location','northeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/levels_comparison_F1score.jpg'))


return


