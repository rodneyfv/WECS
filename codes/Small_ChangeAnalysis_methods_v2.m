
% Evaluation of different methods

%%
close all
clear;
%%

rng(2021)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load previously simulated images
tmp = matfile('../../Images/simulated/simulated_sequence_small_change.mat');
mY0 = tmp.mY;
[n1, n2, ~, n] = size(mY0);
clear tmp

% array to store sequence of images
mY = zeros(n1, n2, n);
cont = 1;
while cont<=80
    mY(:,:,cont) = double(sqrt(mY0(:,:,1,cont).^2 + mY0(:,:,2,cont).^2));
    cont = cont + 1;
end
clear mY0;

% mean observed image
imRef = mean(mY,3);

%%
% original images

% Subsampling of images
subsamplingfactor = 2;
% size of the sequence (must be multiple of 4)
n = 80;

% Loading images 
im1=imread('../figs/GroundTruthEllipsoidChanges/ellipse3.tif');
% boolean ~ used to represent change point with 1
im1=double(~im1(1:subsamplingfactor:end,1:subsamplingfactor:end));
% number of rows and columns
n1 = size(im1,1);
n2 = size(im1,2);

% all changes that are expected to be detected
mImage = figure;
%colormap(gray(256)); imagesc(im1)
imshow(im1/max(max(im1)))
axis off
% saveas(mImage,sprintf('../figs/small_changes.jpg'))
%exportgraphics(mImage,sprintf('../figs/v2/small_changes.jpg'),'BackgroundColor','none')
exportgraphics(mImage,sprintf('../figs/v2/small_changes.jpg'),'Resolution',300)

% example of an image with small changes
mImage = figure;
%colormap(gray(256)); imagesc(mY(:,:,1))
imshow(mY(:,:,1)/max(max(mY(:,:,1))))
axis off
%exportgraphics(mImage,sprintf('../figs/v2/small_changes_first_image.jpg'),'BackgroundColor','none')
exportgraphics(mImage,sprintf('../figs/v2/small_changes_first_image.jpg'),'Resolution',300)

%% WECS

wname = 'db2'; % wavelet basis used
J = 3; % resolution level of wavelet transform

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
set(gca,'FontSize',15)
for m = 1:4:80
    xline(m)
end
% saveas(mImage,sprintf('../figs/small_changes_detected_instants.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/small_changes_detected_instants.jpg'),'Resolution',300)

% saving CSV for ROC curves of WECS, TAAD and ECS
mResults = array2table([1:n; vd']');
mResults.Properties.VariableNames = {'id' 'dm'};
writetable(mResults,'small_changes_dm_instants.csv')


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
axis off
% title('db2 WECS d(m), J=2', 'FontSize', 17)
% saveas(mImage,sprintf('../figs/small_changes_corr_dm.jpg'))
%exportgraphics(mImage,sprintf('../figs/v2/small_changes_Rwecs.jpg'),'BackgroundColor','none')
exportgraphics(mImage,sprintf('../figs/v2/small_changes_Rwecs.jpg'),'Resolution',300)


%% Approach using correlations but without wavelets (ECS)

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
R_ecs = R./max(R(:));

%
mImage = figure;
imshow(R_ecs)
axis off
title('d(m) without wavelets', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/small_changes_corr_dm_nowavelets.jpg'))


%% Standard change detection

% matrix of aggregated absolute differences
tmp = zeros(n1,n2);
for m=2:n
    tmp = tmp + abs(mY(:,:,m) - mY(:,:,m-1));
end
A_taad = tmp./max(tmp(:));

%
mImage = figure;
imshow(A_taad)
title('Aggregation of absolute differences'                                                                                                                                                                                                                                                                                                                               , 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/small_changes_corr_logratios.jpg'))

%% Compare all results

[TP_wecs,FP_wecs]=ROCcurveNew(R_wecs,255*im1); close
[TP_ecs,FP_ecs] = ROCcurveNew(R_ecs,255*im1); close
[TP_agg,FP_agg]=ROCcurveNew(A_taad,255*im1); close

% saving results in a csv file
mResults = array2table([TP_wecs; FP_wecs; TP_ecs; FP_ecs; TP_agg; FP_agg]');
mResults.Properties.VariableNames = {'TP_wecs' 'FP_wecs' 'TP_ecs' 'FP_ecs' 'TP_agg' 'FP_agg'};
%writetable(mResults,'SeqEllipse_methods_ROC.csv')

mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 15)
ylabel('True positive rate', 'FontSize', 15)
axis([0 1 0 1]);
axis square
plot(FP_wecs,TP_wecs,'k','LineWidth',2)
plot(FP_ecs,TP_ecs,'g-d','LineWidth',2)
plot(FP_agg,TP_agg,':+','LineWidth',2)
grid on
legend('WECS', 'ECS', ...
    'TAAD','interpreter','latex', 'Location','southeast', 'FontSize', 15)
legend('boxoff')
hold off
% saveas(mImage,sprintf('../figs/small_changes_methods_comparison.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/small_changes_methods_ROC.jpg'),'Resolution',300)


% saving CSV for ROC curves of WECS, TAAD and ECS
mResults = array2table([TP_wecs; FP_wecs; TP_ecs; FP_ecs; TP_agg; FP_agg]');
mResults.Properties.VariableNames = {'TP_wecs' 'FP_wecs' 'TP_ecs' 'FP_ecs' 'TP_agg' 'FP_agg'};
writetable(mResults,'small_changes_ROC_curves.csv')

%



return


