
% Evaluation of different levels and families on WECS and comparison with a
% deep-learning denoiser

%%
close all
clear;
%%

rng(2021)

% code that generates a sequence of simulated images

% Subsampling of images
subsamplingfactor = 2;
% noise level used
sig = 1;
% size of the sequence (must be multiple of 4)
n = 80;
% running code
generate_synthetic_sequence

% mean SNR of simulated images
meanSNR = 0;
for m=1:4
    meanSNR = meanSNR + norm(mI(:,:,m),'fro')/(4*norm(mY(:,:,m),'fro'));
end
meanSNR

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
imshow(totalchanges)
title('Total changes', 'FontSize', 17)
saveas(mImage,sprintf('../figs/total_changes.jpg'))

% first four observed images
for m=1:4
    mImage = figure;
    imshow(mY(:,:,m))
    saveas(mImage,sprintf('../figs/ellipses_t%d.jpg',m))
end

%% Analysis of decomposition level

% wavelet basis used
wname = 'db2'; 

R_levels = zeros(n1,n2,5);

for J=1:5
    [tmp,~] = wecs(mY,wname,J);
    R_levels(:,:,J) = tmp/max(tmp(:));
    mImage = figure;
    imshow(R_levels(:,:,J))
    title(sprintf('WECS: %s, J=%d',wname,J))
end


% Compare all results

[TP_J1,FP_J1] = ROCcurveNew(R_levels(:,:,1),255*totalchanges); close
[TP_J2,FP_J2] = ROCcurveNew(R_levels(:,:,2),255*totalchanges); close
[TP_J3,FP_J3] = ROCcurveNew(R_levels(:,:,3),255*totalchanges); close
[TP_J4,FP_J4] = ROCcurveNew(R_levels(:,:,4),255*totalchanges); close
[TP_J5,FP_J5] = ROCcurveNew(R_levels(:,:,5),255*totalchanges); close

% saving results in a csv file
mResults = array2table([TP_J1 ; FP_J1; TP_J2 ; FP_J2; TP_J3 ; FP_J3;...
    TP_J4 ; FP_J4; TP_J5 ; FP_J5]');
mResults.Properties.VariableNames = {'TP_J1' 'FP_J1' 'TP_J2' 'FP_J2'...
    'TP_J3' 'FP_J3' 'TP_J4' 'FP_J4' 'TP_J5' 'FP_J5'};
writetable(mResults,'SeqEllipse_levels_ROC.csv')


mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(FP_J1,TP_J1,':+')
plot(FP_J2,TP_J2,'-.or')
plot(FP_J3,TP_J3,':*')
plot(FP_J4,TP_J4,':^')
plot(FP_J5,TP_J5,'g-d')
legend('$J=1$', '$J=2$', '$J=3$','$J=4$', '$J=5$',...
    'interpreter','latex','Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/levels_comparison.jpg'))

[vp_J1,FP_J1] = F1Scorecurve(R_levels(:,:,1),255*totalchanges); close
[vp_J2,FP_J2] = F1Scorecurve(R_levels(:,:,2),255*totalchanges); close
[vp_J3,FP_J3] = F1Scorecurve(R_levels(:,:,3),255*totalchanges); close
[vp_J4,FP_J4] = F1Scorecurve(R_levels(:,:,4),255*totalchanges); close
[vp_J5,FP_J5] = F1Scorecurve(R_levels(:,:,5),255*totalchanges); close

mResults = array2table([vp_J1 ; FP_J1; vp_J2 ; FP_J2; vp_J3 ; FP_J3;...
    vp_J4 ; FP_J4; vp_J5 ; FP_J5]');
mResults.Properties.VariableNames = {'vp_J1' 'FP_J1' 'vp_J2' 'FP_J2'...
    'vp_J3' 'FP_J3' 'vp_J4' 'FP_J4' 'vp_J5' 'FP_J5'};
writetable(mResults,'SeqEllipse_levels_F1score.csv')

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 13)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 13)
axis([0 1 0 0.99]);
axis square
plot(vp_J1,FP_J1,':+')
plot(vp_J2,FP_J2,'-.or')
plot(vp_J3,FP_J3,':*')
plot(vp_J4,FP_J4,':^')
plot(vp_J5,FP_J5,'g-d')
legend('$J=1$', '$J=2$', '$J=3$','$J=4$', '$J=5$',...
    'interpreter','latex','Location','northeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/levels_comparison_F1score.jpg'))


%% Analysis of wavelet basis

% decomposition level fixed
J = 2;

R_families = zeros(n1,n2,6);
cont = 1;
for wname = {'haar','db2','db4','coif4','sym2','sym4'}
    [tmp,~] = wecs(mY,char(wname),J);
    R_families(:,:,cont) = tmp/max(tmp(:));
    mImage = figure;
    imshow(R_families(:,:,cont))
    title(sprintf('WECS: %s, J=%d',char(wname),J))
    cont = cont + 1;
end


% Compare all results

[TP_ha,FP_ha] = ROCcurveNew(R_families(:,:,1),255*totalchanges); close
[TP_d2,FP_d2] = ROCcurveNew(R_families(:,:,2),255*totalchanges); close
[TP_d4,FP_d4] = ROCcurveNew(R_families(:,:,3),255*totalchanges); close
[TP_c4,FP_c4] = ROCcurveNew(R_families(:,:,4),255*totalchanges); close
[TP_s2,FP_s2] = ROCcurveNew(R_families(:,:,5),255*totalchanges); close
[TP_s4,FP_s4] = ROCcurveNew(R_families(:,:,6),255*totalchanges); close

mResults = array2table([TP_ha ; FP_ha; TP_d2 ; FP_d2; TP_d4 ; FP_d4;...
    TP_c4 ; FP_c4; TP_s2 ; FP_s2; TP_s4 ; FP_s4]');
mResults.Properties.VariableNames = {'TP_ha' 'FP_ha' 'TP_d2' 'FP_d2' ...
    'TP_d4' 'FP_d4' 'TP_c4' 'FP_c4' 'TP_s2' 'FP_s2' 'TP_s4' 'FP_s4'};
writetable(mResults,'SeqEllipse_family_ROC.csv')

mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(FP_ha,TP_ha,':+')
plot(FP_d2,TP_d2,'-.or')
plot(FP_d4,TP_d4,':*')
plot(FP_c4,TP_c4,':^')
plot(FP_s2,TP_s2,'g-d')
plot(FP_s4,TP_s4,'k')
legend('haar', 'db2', 'db4',...
    'coif4', 'sym2', 'sym4','Location','northeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/families_comparison.jpg'))


[vp_ha,FP_ha] = F1Scorecurve(R_families(:,:,1),255*totalchanges); close
[vp_d2,FP_d2] = F1Scorecurve(R_families(:,:,2),255*totalchanges); close
[vp_d4,FP_d4] = F1Scorecurve(R_families(:,:,3),255*totalchanges); close
[vp_c4,FP_c4] = F1Scorecurve(R_families(:,:,4),255*totalchanges); close
[vp_s2,FP_s2] = F1Scorecurve(R_families(:,:,5),255*totalchanges); close
[vp_s4,FP_s4] = F1Scorecurve(R_families(:,:,6),255*totalchanges); close

mResults = array2table([vp_ha ; FP_ha; vp_d2 ; FP_d2; vp_d4 ; FP_d4;...
    vp_c4 ; FP_c4; vp_s2 ; FP_s2; vp_s4 ; FP_s4]');
mResults.Properties.VariableNames = {'vp_ha' 'FP_ha' 'vp_d2' 'FP_d2' ...
    'vp_d4' 'FP_d4' 'vp_c4' 'FP_c4' 'vp_s2' 'FP_s2' 'vp_s4' 'FP_s4'};
writetable(mResults,'SeqEllipse_family_F1score.csv')

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 13)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 13)
axis([0 1 0 0.99]);
axis square
plot(vp_ha,FP_ha,':+')
plot(vp_d2,FP_d2,'-.or')
plot(vp_d4,FP_d4,':*')
plot(vp_c4,FP_c4,':^')
plot(vp_s2,FP_s2,'g-d')
plot(vp_s4,FP_s4,'k')
legend('haar', 'db2', 'db4',...
    'coif4', 'sym2', 'sym4','Location','northeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/families_comparison_F1score.jpg'))



%% Analysis of wavelet basis

% WECS

% decomposition level fixed
J = 2;
% wavelet basis
wname = 'db2';

R_wecs = zeros(n1,n2);
[R_wecs,~] = wecs(mY,char(wname),J);
R_wecs = R_wecs/max(R_wecs(:));
mImage = figure;
imshow(R_wecs)
title(sprintf('WECS: %s, J=%d',wname,J))

% Deep-learning denoiser

% load a deep neural network that has learned to denoise more efficiently than wavelets
net = denoisingNetwork('DnCNN'); 

% mean observed image
imRef = mean(mY,3);

mX = zeros(n1,n2,n);
mD = zeros(n1,n2,n);
tic
for m=1:n
    tmp = denoiseImageCPU(mY(:,:,m),net);
    mX(:,:,m) = double(tmp);
    mD(:,:,m) = (mX(:,:,m) - imRef).^2;
end
toc

% vector of overall changes
vd = reshape(sum(sum(mD,1),2),n,1);

% matrix of correlations
R = zeros(n1,n2);
for ii=1:n1
   for jj=1:n2
       tmp = abs(corrcoef(reshape(mD(ii,jj,:),n,1),vd));
       R(ii,jj) = tmp(1,2);
   end
end
R_deep = R./max(R(:));
mImage = figure;
imshow(R)
title('d(m): deep learning denoiser')

% Compare all results

[TP_wecs,FP_wecs] = ROCcurveNew(R_wecs,255*totalchanges); close
[TP_deep,FP_deep] = ROCcurveNew(R_deep,255*totalchanges); close

mResults = array2table([TP_wecs ; FP_wecs; TP_deep ; FP_deep]');
mResults.Properties.VariableNames = {'TP_wecs' 'FP_wecs' 'TP_deep' 'FP_deep'};
writetable(mResults,'SeqEllipse_deep_ROC.csv')

mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(FP_wecs,TP_wecs,':+')
plot(FP_deep,TP_deep,'-.or')
legend('db2 WECS $\textbf{d}(m)$, $J=2$', '\textbf{d}(m): deep learning denoiser', ...
    'interpreter','latex','Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/dm_comparison_wavelet_deepL.jpg'))


[vp_wecs,FP_wecs] = F1Scorecurve(R_wecs,255*totalchanges); close
[vp_deep,FP_deep] = F1Scorecurve(R_deep,255*totalchanges); close

mResults = array2table([vp_wecs ; FP_wecs; vp_deep ; FP_deep]');
mResults.Properties.VariableNames = {'vp_wecs' 'FP_wecs' 'vp_deep' 'FP_deep'};
writetable(mResults,'SeqEllipse_deep_F1score.csv')

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 13)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 13)
axis([0 1 0 0.99]);
axis square
plot(vp_wecs,FP_wecs,':+')
plot(vp_deep,FP_deep,'-.or')
legend('db2 WECS $\textbf{d}(m)$, $J=2$', '\textbf{d}(m): deep learning denoiser', ...
    'interpreter','latex','Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('../figs/dm_comparison_wavelet_deepL_F1score.jpg'))


return


