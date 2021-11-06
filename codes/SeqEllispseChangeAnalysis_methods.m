
% Evaluation of different methods

%%
close all
clear;
%%

% code that generates a sequence of synthetic images

% Subsampling of images
subsamplingfactor = 2;
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
saveas(mImage,sprintf('../figs/total_changes.jpg'))

% mean observed image
imRef = mean(mY,3);

%% WECS

wname = 'db2'; % wavelet basis used
J = 2; % resolution level of wavelet transform

mX = zeros(n1,n2,n);
for m=1:n
    [tmp,~,~,~] = swt2(mY(:,:,m),J,wname);
    mX(:,:,m) = tmp(:,:,J); % focus only on level J approximations
end

% matrix of squared mean differences
mD = zeros(n1,n2,n);
for m=1:n
    mD(:,:,m) = (mX(:,:,m) - imRef).^2;
end

% vector of overall changes
vd = reshape(sum(sum(mD,1),2),n,1);
plot(1:n,vd)

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
title('db2 WECS d(m), J=2', 'FontSize', 17)
saveas(mImage,sprintf('../figs/corr_changes_dm.jpg'))


%% No wavelets for comparison with mean image

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
saveas(mImage,sprintf('../figs/corr_changes_dm_nowavelets.jpg'))


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
saveas(mImage,sprintf('../figs/corr_changes_logratios.jpg'))

%% Compare all results

[TP_wecs,FP_wecs]=ROCcurveNew(R_wecs,255*totalchanges); close
[TP_nowecs,FP_nowecs] = ROCcurveNew(R_nowecs,255*totalchanges); close
[TP_agg,FP_agg]=ROCcurveNew(S,255*totalchanges); close

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
saveas(mImage,sprintf('../figs/methods_comparison.jpg'))

%

[vp_wecs,FP_wecs,~,~] = F1Scorecurve(R_wecs,255*totalchanges); close
[vp_nowecs,FP_nowecs,~,~] = F1Scorecurve(R_nowecs,255*totalchanges); close
[vp_agg,FP_agg,~,~] = F1Scorecurve(S,255*totalchanges); close

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
saveas(mImage,sprintf('../figs/methods_comparison_F1score.jpg'))



return


