
% Evaluation of different levels and families on WECS and comparison with a
% deep-learning denoiser

%
close all
clear;
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load previously simulated images
tmp = matfile('../../Images/simulated/simulated_sequence.mat');
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% original images

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
clear im1 im2 im3 im4


% mean SNR of simulated images
meanSNR = 0;
for m=1:4
    tmp = norm(sqrt(mI(:,:,m).^2 + mI(:,:,m).^2),'fro')/(norm(mY(:,:,m),'fro'));
    sprintf('SNR of simulated images for m=%d',m)
    tmp
    meanSNR = meanSNR + tmp/4;
end
fprintf('Mean SNR of simulated images')
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
imshow(totalchanges / max(max(totalchanges)))
axis off
%title('Total changes', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/total_changes_v2.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/total_changes_v2.jpg'),'Resolution',300)

% first four observed images
for m=1:4
    mImage = figure;
    imshow( (mY(:,:,m)- min(min(mY(:,:,m)))) /( max(max(mY(:,:,m))) - min(min(mY(:,:,m)))))
    axis off
    %saveas(mImage,sprintf('../figs/v2/ellipses_t%d_v2.jpg',m))
    exportgraphics(mImage,sprintf('../figs/v2/ellipses_t%d_v2.jpg',m),'Resolution',300)
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
writetable(mResults,'SeqEllipse_levels_ROC_v2.csv')


mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 15)
ylabel('True positive rate', 'FontSize', 15)
axis([0 1 0 1]);
axis square
plot(FP_J1,TP_J1,':+','LineWidth',2)
plot(FP_J2,TP_J2,'-.or','LineWidth',2)
plot(FP_J3,TP_J3,':*','LineWidth',2)
plot(FP_J4,TP_J4,':^','LineWidth',2)
plot(FP_J5,TP_J5,'g-d','LineWidth',2)
grid on
legend('$J=1$', '$J=2$', '$J=3$','$J=4$', '$J=5$',...
    'interpreter','latex','Location','southeast', 'FontSize', 18)
legend('boxoff')
hold off
%saveas(mImage,sprintf('../figs/v2/levels_ROC_v2.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/levels_ROC_v2.jpg'),'Resolution',300)


[vp_J1,FP_J1] = F1Scorecurve(R_levels(:,:,1),255*totalchanges); close
[vp_J2,FP_J2] = F1Scorecurve(R_levels(:,:,2),255*totalchanges); close
[vp_J3,FP_J3] = F1Scorecurve(R_levels(:,:,3),255*totalchanges); close
[vp_J4,FP_J4] = F1Scorecurve(R_levels(:,:,4),255*totalchanges); close
[vp_J5,FP_J5] = F1Scorecurve(R_levels(:,:,5),255*totalchanges); close

mResults = array2table([vp_J1 ; FP_J1; vp_J2 ; FP_J2; vp_J3 ; FP_J3;...
    vp_J4 ; FP_J4; vp_J5 ; FP_J5]');
mResults.Properties.VariableNames = {'vp_J1' 'FP_J1' 'vp_J2' 'FP_J2'...
    'vp_J3' 'FP_J3' 'vp_J4' 'FP_J4' 'vp_J5' 'FP_J5'};
writetable(mResults,'SeqEllipse_levels_F1score_v2.csv')

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 15)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 15)
axis([0 1 0 0.99]);
axis square
plot(vp_J1,FP_J1,':+','LineWidth',2)
plot(vp_J2,FP_J2,'-.or','LineWidth',2)
plot(vp_J3,FP_J3,':*','LineWidth',2)
plot(vp_J4,FP_J4,':^','LineWidth',2)
plot(vp_J5,FP_J5,'g-d','LineWidth',2)
grid on
legend('$J=1$', '$J=2$', '$J=3$','$J=4$', '$J=5$',...
    'interpreter','latex','Location','northeast', 'FontSize', 18)
legend('boxoff')
hold off
%saveas(mImage,sprintf('../figs/v2/levels_F1score_v2.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/levels_F1score_v2.jpg'),'Resolution',300)


%% Analysis of wavelet basis

% decomposition level fixed
J = 3;

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
writetable(mResults,'SeqEllipse_family_ROC_v2.csv')

mImage = figure;
hold on
%title('ROC Curve', 'FontSize', 17)
xlabel('False positive rate', 'FontSize', 15)
ylabel('True positive rate', 'FontSize', 15)
axis([0 1 0 1]);
axis square
plot(FP_ha,TP_ha,':+','LineWidth',2)
plot(FP_d2,TP_d2,'-.or','LineWidth',2)
plot(FP_d4,TP_d4,':*','LineWidth',2)
plot(FP_c4,TP_c4,':^','LineWidth',2)
plot(FP_s2,TP_s2,'g-d','LineWidth',2)
plot(FP_s4,TP_s4,'k','LineWidth',2)
grid on
legend('haar', 'db2', 'db4',...
    'coif4', 'sym2', 'sym4','Location','southeast', 'FontSize', 15)
legend('boxoff')
hold off
%saveas(mImage,sprintf('../figs/v2/families_ROC_v2.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/families_ROC_v2.jpg'),'Resolution',300)


[vp_ha,FP_ha] = F1Scorecurve(R_families(:,:,1), 255*totalchanges); close
[vp_d2,FP_d2] = F1Scorecurve(R_families(:,:,2), 255*totalchanges); close
[vp_d4,FP_d4] = F1Scorecurve(R_families(:,:,3), 255*totalchanges); close
[vp_c4,FP_c4] = F1Scorecurve(R_families(:,:,4), 255*totalchanges); close
[vp_s2,FP_s2] = F1Scorecurve(R_families(:,:,5), 255*totalchanges); close
[vp_s4,FP_s4] = F1Scorecurve(R_families(:,:,6), 255*totalchanges); close

mResults = array2table([vp_ha ; FP_ha; vp_d2 ; FP_d2; vp_d4 ; FP_d4;...
    vp_c4 ; FP_c4; vp_s2 ; FP_s2; vp_s4 ; FP_s4]');
mResults.Properties.VariableNames = {'vp_ha' 'FP_ha' 'vp_d2' 'FP_d2' ...
    'vp_d4' 'FP_d4' 'vp_c4' 'FP_c4' 'vp_s2' 'FP_s2' 'vp_s4' 'FP_s4'};
writetable(mResults,'SeqEllipse_family_F1score_v2.csv')

mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 15)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 15)
axis([0 1 0 0.99]);
axis square
plot(vp_ha,FP_ha,':+','LineWidth',2)
plot(vp_d2,FP_d2,'-.or','LineWidth',2)
plot(vp_d4,FP_d4,':*','LineWidth',2)
plot(vp_c4,FP_c4,':^','LineWidth',2)
plot(vp_s2,FP_s2,'g-d','LineWidth',2)
plot(vp_s4,FP_s4,'k','LineWidth',2)
grid on
legend('haar', 'db2', 'db4',...
    'coif4', 'sym2', 'sym4','Location','northeast', 'FontSize', 15)
legend('boxoff')
hold off
%saveas(mImage,sprintf('../figs/v2/families_F1score_v2.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/families_F1score_v2.jpg'),'Resolution',300)



return


