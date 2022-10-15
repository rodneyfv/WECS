
% Evaluation of different methods

%%
close all
clear;
%%

rng(2021)

% generating a sequence of synthetic images

% Subsampling of images
subsamplingfactor = 2;
% noise level used
sig = 1;
% size of the sequence (must be multiple of 4)
n = 80;

% Loading the image with small ellipses
im1=imread('../figs/GroundTruthEllipsoidChanges/ellipse3.tif');
im1=double(~im1(1:subsamplingfactor:end,1:subsamplingfactor:end)); 
% number of rows and columns
n1 = size(im1,1);
n2 = size(im1,2);


sprintf('signal to noise ratios')
sprintf('im1: %f',sqrt(sum(sum(im1.^2,1))/(n1*n2))/sig)

% array to store sequence of images
mY = zeros(n1, n2, n);
% sequence of images with noise
cont = 1;
while cont<=80
    mY(:,:,cont) = im1 + sig*randn(n1, n2);
    cont = cont + 1;
    for m=1:3
        mY(:,:,cont) = sig*randn(n1, n2);
        cont = cont + 1;
    end
end


% output
size(mY) % sequence of synthetic images
% number of rows and columns
n1 = size(mY,1);
n2 = size(mY,2);

% all changes that are expected to be detected
mImage = figure;
colormap(gray(256)); imagesc(im1)
axis off
% saveas(mImage,sprintf('../figs/small_changes.jpg'))
exportgraphics(mImage,sprintf('../figs/small_changes.jpg'),'BackgroundColor','none')

% all changes that are expected to be detected
mImage = figure;
colormap(gray(256)); imagesc(mY(:,:,1))
axis off
exportgraphics(mImage,sprintf('../figs/small_changes_first_image.jpg'),'BackgroundColor','none')


% mean observed image
imRef = mean(mY,3);

%% WECS

wname = 'db2'; % wavelet basis used
J = 2; % resolution level of wavelet transform

% computing the wavelet coefficients of normalized images
mX = zeros(n1,n2,n);
for m=1:n
    mY(:,:,m) = mY(:,:,m)/norm(imRef);
    [tmp,~,~,~] = swt2(mY(:,:,m),J,wname);
    mX(:,:,m) = tmp(:,:,J); % focus only on level J approximations
end
% normalized mean image
imRef = imRef/norm(imRef);

% matrix of squared mean differences
mD = zeros(n1,n2,n);
for m=1:n
    mD(:,:,m) = (mX(:,:,m) - imRef).^2;
end

% vector of overall changes
mImage = figure;
vd = reshape(sum(sum(mD,1),2),n,1);
plot(1:n,vd,'LineWidth',2)
xlabel('$m$','interpreter','latex','FontSize',20); xlim([0 n])
ylabel('$\textbf{d}(m)$','interpreter','latex','FontSize',20);
set(gca,'FontSize',13)
for m = 1:4:80
    xline(m)
end
% saveas(mImage,sprintf('../figs/small_changes_detected_instants.jpg'))
exportgraphics(mImage,sprintf('../figs/small_changes_detected_instants.jpg'),'BackgroundColor','none')

%
R = zeros(n1,n2);
for ii=1:n1
   for jj=1:n2
       tmp = abs(corrcoef(reshape(mD(ii,jj,:),n,1),vd));
       R(ii,jj) = tmp(1,2);
   end
end
R_wecs = R./max(R(:));

%
mImage = figure;
imshow(R_wecs)
% title('db2 WECS d(m), J=2', 'FontSize', 17)
% saveas(mImage,sprintf('../figs/small_changes_corr_dm.jpg'))
exportgraphics(mImage,sprintf('../figs/small_changes_corr_dm.jpg'),'BackgroundColor','none')


%% No wavelets for comparison with mean image (ECS)

% matrix of squared mean differences
mD = zeros(n1,n2,n);
for m=1:n
    mD(:,:,m) = (mY(:,:,m) - imRef).^2;
end

% vector of overall changes
vd = reshape(sum(sum(mD,1),2),n,1);
plot(1:n,vd)

%
NbPixels = n1*n2;
R = zeros(n1,n2);
for ii=1:n1
   for jj=1:n2
       tmp = abs(corrcoef(reshape(mD(ii,jj,:),n,1),vd));
       R(ii,jj) = tmp(1,2);
   end
end
R_nowecs = R./max(R(:));

%
mImage = figure;
imshow(R_nowecs)
title('d(m) without wavelets', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/small_changes_corr_dm_nowavelets.jpg'))


%% Standard change detection

% matrix of aggregated absolute differences
S = zeros(n1,n2);
for m=2:n
    S = S + abs(mY(:,:,m) - mY(:,:,m-1));
end
S = S./max(S(:));

%
mImage = figure;
imshow(S)
title('Aggregation of absolute differences'                                                                                                                                                                                                                                                                                                                               , 'FontSize', 17)
%saveas(mImage,sprintf('../figs/small_changes_corr_logratios.jpg'))

%% Compare all results

[TP_wecs,FP_wecs]=ROCcurveNew(R_wecs,255*im1); close
[TP_nowecs,FP_nowecs] = ROCcurveNew(R_nowecs,255*im1); close
[TP_agg,FP_agg]=ROCcurveNew(S,255*im1); close

% saving results in a csv file
mResults = array2table([TP_wecs; FP_wecs; TP_nowecs; FP_nowecs; TP_agg; FP_agg]');
mResults.Properties.VariableNames = {'TP_wecs' 'FP_wecs' 'TP_nowecs' 'FP_nowecs' 'TP_agg' 'FP_agg'};
%writetable(mResults,'SeqEllipse_methods_ROC.csv')

mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(FP_wecs,TP_wecs,'k')
plot(FP_nowecs,TP_nowecs,'g-d')
plot(FP_agg,TP_agg,':+')
legend('db2 WECS $\textbf{d}(m)$, $J=2$', '$\textbf{d}(m)$: without wavelets', ...
    'Aggregation of absolute differences','interpreter','latex', 'Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
% saveas(mImage,sprintf('../figs/small_changes_methods_comparison.jpg'))
exportgraphics(mImage,sprintf('../figs/small_changes_methods_comparison.jpg'),'BackgroundColor','none')

%

[vp_wecs,FP_wecs,~,~] = F1Scorecurve(R_wecs,255*im1); close
[vp_nowecs,FP_nowecs,~,~] = F1Scorecurve(R_nowecs,255*im1); close
[vp_agg,FP_agg,~,~] = F1Scorecurve(S,255*im1); close

mResults = array2table([vp_wecs; FP_wecs; vp_nowecs; FP_nowecs; vp_agg; FP_agg]');
mResults.Properties.VariableNames = {'vp_wecs' 'FP_wecs' 'vp_nowecs' 'FP_nowecs' 'vp_agg' 'FP_agg'};
%writetable(mResults,'SeqEllipse_methods_F1score.csv')

% saving CSV for ROC curves of WECS, TAAD and ECS
mResults = array2table([FP_wecs; vp_wecs; FP_agg; vp_agg; FP_nowecs; vp_nowecs]');
mResults.Properties.VariableNames = {'FP_wecs' 'f1_score_wecs' 'FP_taad' 'f1_score_taad' 'FP_ecs' 'f1_score_ecs'};
writetable(mResults,'small_changes_ROC_curves_forest.csv')


mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 13)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 13)
axis([0 1 0 0.99]);
axis square
plot(vp_wecs,FP_wecs,'k')
plot(vp_nowecs,FP_nowecs,'g-d')
plot(vp_agg,FP_agg,':+')
legend('db2 WECS $\textbf{d}(m)$, $J=2$', '$\textbf{d}(m)$: without wavelets', ...
    'Aggregation of absolute differences','interpreter','latex', 'Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
%saveas(mImage,sprintf('../figs/small_changes_methods_comparison_F1score.jpg'))




return


