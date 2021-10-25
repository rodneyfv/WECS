
% Evaluation of different methods

%%
close all
clear;
%%
im1 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle1.png');
im2 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle2.png');  
im3 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle3.png');  
im4 = imread('../figs/ImagesWithEllipsoidChanges/ImageEllipseSpeckle4.png');  
%
figure
imshow(im1)
figure
imshow(im2)
figure
imshow(im3)
figure
imshow(im4)

figure
subplot(1,4,1); imshow(im1); title('I(1)')
subplot(1,4,2); imshow(im2); title('I(2)')
subplot(1,4,3); imshow(im3); title('I(3)')
subplot(1,4,4); imshow(im4); title('I(4)')

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
mImage = figure;
imshow(totalchanges)
title('Total changes', 'FontSize', 17)
%saveas(mImage,sprintf('total_changes.jpg'))

%%
eps = .00001;
im1log = log(eps+double(im1));
im2log = log(eps+double(im2));
im3log = log(eps+double(im3));
im4log = log(eps+double(im4));
n=4;
imRef = im1log + im2log + im3log + im4log;
imRef = imRef/n;

%% comparison with mean image

wname = 'db2';
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
% imRef_wave = imRef_wave/n;
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
[pD,pFA]=ROCcurveNew(R,255*totalchanges); close
%
mImage = figure;
imshow(R)
title('db2 WECS d(m), J=2', 'FontSize', 17)
%saveas(mImage,sprintf('corr_changes_dm.jpg'))
    
%% No wavelets for comparison with mean image
X1 = im1log;
X2 = im2log;
X3 = im3log;
X4 = im4log;

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
[pD0,pFA0]=ROCcurveNew(R,255*totalchanges); close
%
mImage = figure;
imshow(R)
title('d(m) without wavelets', 'FontSize', 17)
%saveas(mImage,sprintf('corr_changes_dm_nowavelets.jpg'))


%% Standard change detection

S = abs(im4log - im3log) + abs(im3log - im2log) + abs(im2log - im1log);
S = S./max(R(:));
%
[pD1,pFA1]=ROCcurveNew(S,255*totalchanges); close
%
mImage = figure;
imshow(S)
title('Aggregation of log-ratios'                                                                                                                                                                                                                                                                                                                               , 'FontSize', 17)
saveas(mImage,sprintf('corr_changes_logratios.jpg'))

%% Compare all results

mImage = figure;
hold on
title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(pFA,pD,'k')
plot(pFA0,pD0,'g-d')
plot(pFA1,pD1,':+')
legend('db2 WECS d(m), J=2', 'd(m): without wavelets', ...
    'Aggregation of log-ratios', 'Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
%saveas(mImage,sprintf('methods_comparison.jpg'))



return


