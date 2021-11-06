
% Evaluation of wavelet bases when mean approximation coefficients are used

%%
close all
clear;
%%

% code that generates a sequence of synthetic images

% Subsampling of images
subsamplingfactor = 8;
% noise level used
sig = 1;
% size of the sequence (must be multiple of 4)
n = 80;
% running code
generate_synthetic_sequence

% output
size(mI) % true images
size(mY) % sequence of synthetic images
% number of rows and columns
n1 = size(mY,1);
n2 = size(mY,2);

% all changes that are expected to be detected
totalchanges = (abs(mI(:,:,1)-mI(:,:,2)) + abs(mI(:,:,2)-mI(:,:,3)) + ...
    abs(mI(:,:,3)-mI(:,:,4)) + abs(mI(:,:,4)-mI(:,:,1)))>0;
mImage = figure;
colormap(gray(256)); imagesc(totalchanges)
title('Total changes', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/total_changes.jpg'))

% mean observed image
imRef = mean(mY,3);

% decomposition level fixed
J = 2;

%%

% Wavelet basis
wname = 'haar';

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
[vp,vF1_haar,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('Haar wavelet - J=2')

%% 

wname = 'db2';

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
[vp,vF1_db2,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db2 wavelet - J=2')

%% 

wname = 'db4';

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
[vp,vF1_db4,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('db4 wavelet - J=2')

%% 

wname = 'coif4';

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
[vp,vF1_coif4,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('coif4 wavelet - J=2')

%% 

wname = 'sym4';

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
[vp,vF1_sym4,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('sym4 wavelet - J=2')

%% 

wname = 'sym2';

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
[pD6,pFA6]=ROCcurveNew(R,255*totalchanges); close
[vp,vF1_sym2,~,~] = F1Scorecurve(R,255*totalchanges); close
%
figure
imshow(R)
title('sym2 wavelet - J=2')

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
plot(pFA6,pD6,'k')
legend('haar', 'db2', 'db4',...
    'coif4', 'sym4', 'sym2','Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/families_comparison.jpg'))

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 13)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 13)
axis([0 1 0 0.7]);
axis square
plot(vp,vF1_haar,':+')
plot(vp,vF1_db2,'-.or')
plot(vp,vF1_db4,':*')
plot(vp,vF1_coif4,':^')
plot(vp,vF1_sym4,'g-d')
plot(vp,vF1_sym2,'k')
legend('haar', 'db2', 'db4',...
    'coif4', 'sym4', 'sym2','Location','northeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/families_comparison_F1score.jpg'))



return


