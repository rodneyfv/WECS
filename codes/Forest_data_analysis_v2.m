
clear 
close all
clc

%% Compute the spatial coefficient of variation per polarization for every acqusition

% dates corresponding to each image
dates = readtable('../../Images/timeSeries/ascending/saida.txt','ReadVariableNames',false);
[n,~] = size(dates);
n = n - 2; % the last two elements in the table are not dates
% Sorting the dates according to time
tmp = dates;
for m=1:n
    tmp2 = char(tmp{m,:});
    if tmp2(7)=='-'
        tmp{m,:} = {insertAfter(tmp2,5,'0')};
    end
end
[~, dates_idsort] = sortrows(tmp(1:n,1));
dates = dates(dates_idsort,:);

% reading the first image
tic
t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{1,:})),'r');
Y = read(t);
toc
imagesc(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2))

%
[Nx,Ny,Nc] = size(Y);
% Nx * Ny : spatial dimension
% Nc : number of channels, here: VV and VH

%% Configurations for wavelet decompositions

wname='db2';

% saving the current boundary handling and then changing it to periodic.
% The default mode was giving different norms of vC for different
% decomposition levels J
origMode = dwtmode('status','nodisplay');
dwtmode('per');

% % this value will be added to images when taking logs
% eps = min(min(min(min(data(:,:,:,:)))));
% eps = abs(eps)*(eps<0)  + .001;


%% Analysis of squared mean deviations


%********************************************%
%******** analysis for combined chanel ************%

% this matrix will store the mean squared differences of approximation
% coefficients, but to save time, we first store the observed amplitudes of
% the images on it
mD_wecs = zeros([Nx, Ny, n]);

tic
% mean image
imRef = zeros(Nx,Ny);
for m=1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    % observed amplitudes
    mD_wecs(:,:,m) = double(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2));
    imRef = imRef + double(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2));
end
tmp = mean(mD_wecs, 3);
% constant to make the mean matrix have norm one
normconst = norm(tmp);
imRef = tmp/normconst;

% resolution level used on wavelet decompositions
J = 3;

for m=1:n
    % we normalize the image by the norm of the mean image
    data = mD_wecs(:,:,m)/normconst;
    % extending image to be able to use swt2
    pwr2 = ceil(log2(min([Nx Ny])));
    extens = ceil((2^pwr2 - min([Nx Ny]))/2);
    tmp = wextend('2D','per',data,extens); % matrix whose dim is power of 2
    % wavelet approximation for log of VV chanel
    [tmp2,~,~,~] = swt2(tmp(1:2^pwr2,1:2^pwr2),J,wname);
    % sqared mean deviations of level J approx. coeff.
    mD_wecs(:,:,m) = (tmp2(extens+1:extens+Nx,extens+1:extens+Ny,J) - imRef).^2;
end

% vector with total sum of squared mean deviations for each image
vd_wecs = reshape(sum(sum(mD_wecs,1),2),1,n);

% computing correlations
R_wecs = zeros(Nx,Ny);
for ii=1:Nx
    for jj=1:Ny
        tmp = abs(corrcoef(reshape(mD_wecs(ii,jj,:),1,n),vd_wecs));
        R_wecs(ii,jj) = tmp(1,2);
    end
end
toc
clear Y t tmp tmp2 mD_wecs

% plot of mean squared deviations
mImage = figure;
plot(1:n, vd_wecs,'LineWidth',2);
hold on
tmp = mad(vd_wecs);
plot(1:n, (2*tmp + median(vd_wecs))*ones(1,n),'LineWidth',2)
xlabel('$m$','interpreter','latex','FontSize',20); xlim([0 n])
ylabel('$\textbf{d}(m)$','interpreter','latex','FontSize',20);
set(gca,'FontSize',13)
hold off
saveas(mImage,sprintf('forest_vSumDifCoefSq_v2.jpg'))

% checking the images corresponding to change points in time
for m = [1 30 59]
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data = double(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2))/normconst;
    
    % extending image to be able to use swt2
    pwr2 = ceil(log2(min([Nx Ny])));
    extens = ceil((2^pwr2 - min([Nx Ny]))/2);
    tmp = wextend('2D','per',data,extens); % matrix whose dim is power of 2
    % wavelet approximation for log of VV chanel
    [tmp2,~,~,~] = swt2(tmp(1:2^pwr2,1:2^pwr2),J,wname);

    save_tiff_image(tmp2(extens+1:extens+Nx,extens+1:extens+Ny,J),...
        sprintf('forest_changes_time_m%2d_v2.tiff',m));
end
clear tmp tmp2
save_tiff_image(imRef,...
        sprintf('forest_changes_time_mean_v2.tiff'));

    
% Saving the images of correlations
save_tiff_image(R_wecs,...
        sprintf('forest_wecs_abscorr_v2.tiff'));
mImage = figure;
imagesc(R_wecs)
axis off; colorbar
set(gca,'FontSize',13)
%saveas(mImage,sprintf('../figs_v2/forest_wecs_abscorr.jpg'))

% Change image using threshold suggested in feature screening literature
%histogram(mCorr(:))
mImage = figure;
cutoff = quantile(R_wecs(:),1-1/log(Nx*Ny));
imshow(R_wecs > cutoff)
%saveas(mImage,sprintf('../figs/forest_wecs_change_space.jpg'))
save_tiff_image(double(R_wecs > cutoff),...
        sprintf('forest_wecs_abscorr_screethrs_v2.tiff'));

cutoff_KI_wecs = kittler(R_wecs); % Kittler-Illingworth threshold
cutoff_Otsu_wecs = graythresh(R_wecs); % Otsu's threshold

tmp = double(R_wecs > cutoff_KI_wecs);
save_tiff_image(tmp,...
        sprintf('forest_wecs_change_space_KI_v2.tiff'));
tmp = double(R_wecs > cutoff_Otsu_wecs);
save_tiff_image(tmp,...
        sprintf('forest_wecs_change_space_otsu_v2.tiff'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Aggregation of differences (TAAD)
A_taad = zeros(Nx,Ny);

tic
t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{1,:})),'r');
Y = read(t);
data1 = double(Y(:,:,1).^2 + Y(:,:,2).^2)/normconst;
for m=2:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data2 = double(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2))/normconst;    
    A_taad = A_taad + abs(data2 - data1);
    data2 = data1;
end
clear data1 data2
A_taad = A_taad./max(A_taad(:));
toc

% saving change measures obtained with TAAD
save_tiff_image(A_taad,...
        sprintf('forest_taad_v2.tiff'));

mImage = figure;
imagesc(A_taad)
axis off; colorbar
set(gca,'FontSize',13)
%saveas(mImage,sprintf('../figs_v2/forest_aggreg_ratios.jpg'))

cutoff_KI_taad = kittler(A_taad); % Kittler-Illingworth threshold
cutoff_Otsu_taad = graythresh(A_taad); % Otsu's threshold

mImage = figure;
imshow(A_taad > cutoff_Otsu_taad)
%saveas(mImage,sprintf('../figs_v2/forest_aggreg_change_space.jpg'))

tmp = double(A_taad > cutoff_KI_taad);
save_tiff_image(tmp,...
        sprintf('forest_taad_change_space_KI_v2.tiff'));
tmp = double(A_taad > cutoff_Otsu_taad);
save_tiff_image(tmp,...
        sprintf('forest_taad_change_space_otsu_v2.tiff'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% No wavelets for comparison with mean image (ECS)

% matrix of squared mean differences
mD_ecs = zeros([Nx, Ny, n]);

tic
% mean image
imRef = zeros(Nx,Ny);
for m=1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    imRef = imRef + double(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2));
end
imRef = imRef/n;
% constant to make the mean matrix have norm one
normconst = norm(imRef);
imRef = imRef/normconst;

for m=1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data = double(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2))/normconst;
    % sqared mean deviations
    mD_ecs(:,:,m) = (data - imRef).^2;
end
clear Y t data

% vector of overall changes
vd_ecs = reshape(sum(sum(mD_ecs,1),2),n,1);

% computing correlations
R_ecs = zeros(Nx,Ny);
for ii=1:Nx
    for jj=1:Ny
        tmp = abs(corrcoef(reshape(mD_ecs(ii,jj,:),1,n),vd_ecs));
        R_ecs(ii,jj) = tmp(1,2);
    end
end
toc
clear mD_ecs

% vector of overall changes
plot(1:n,vd_ecs)

% saving image of correlations from ECS
save_tiff_image(R_ecs,...
        sprintf('forest_ecs_v2.tiff'));
imagesc(R_ecs)

cutoff_KI_ecs = kittler(R_ecs); % Kittler-Illingworth threshold
cutoff_Otsu_ecs = graythresh(R_ecs); % Otsu's threshold

tmp = double(R_ecs > cutoff_KI_ecs);
save_tiff_image(tmp,...
        sprintf('forest_ecs_change_space_KI_v2.tiff'));
tmp = double(R_ecs > cutoff_Otsu_ecs);
save_tiff_image(tmp,...
        sprintf('forest_ecs_change_space_otsu_v2.tiff'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% reading the reference image with change locations
%t = Tiff('../figs/change_nonchange.tif','r');
%Y = read(t);

t = Tiff('../figs/imgROIs.tif','r');
Y = read(t);

mChange = double(Y(:,:,1));
mImage = figure;
imshow(mChange)
saveas(mImage,sprintf('forest_change_v2.jpg'))

% Computing F1-score for WECS and image using aggregation

% file to write the scores
file_cd_scores = fopen('F1scores_cd_methods_v2.txt','w');

% F1-score for changing regions
[F1_wecs_ki, Pr_wecs_ki, Re_wecs_ki] = F1score(R_wecs > cutoff_KI_wecs, mChange);
[F1_wecs_otsu, Pr_wecs_otsu, Re_wecs_otsu] = F1score(R_wecs > cutoff_Otsu_wecs, mChange);
[F1_taad_ki, Pr_taad_ki, Re_taad_ki] = F1score(A_taad > cutoff_KI_taad, mChange);
[F1_taad_otsu, Pr_taad_otsu, Re_taad_otsu] = F1score(A_taad > cutoff_Otsu_taad, mChange);
[F1_ecs_ki, Pr_ecs_ki, Re_ecs_ki] = F1score(R_ecs > cutoff_KI_ecs, mChange);
[F1_ecs_otsu, Pr_ecs_otsu, Re_ecs_otsu] = F1score(R_ecs > cutoff_Otsu_ecs, mChange);

% change map computed with change vector analysis
mCD_CVA = imread('../figs/CVA_BrazilGuiana.png');
[F1_cva, Pr_cva, Re_cva] = F1score(mCD_CVA./255, mChange);
% change map computed with multivariate alteration detection
mCD_MAD = imread('../figs/IRMAD_BrazilGuiana.png');
[F1_mad, Pr_mad, Re_mad] = F1score(mCD_MAD./255, mChange);
% change map computed with PCA-Kmeans
mCD_PCA = imread('../figs/PCAKmeans_BrazilGuiana.png');
[F1_pca, Pr_pca, Re_pca] = F1score(mCD_PCA./255, mChange);
% change map computed with slow feature analysis
mCD_SFA = imread('../figs/ISFA_BrazilGuiana.png');
[F1_sfa, Pr_sfa, Re_sfa] = F1score(mCD_SFA./255, mChange);

fprintf(file_cd_scores, '%10s %6s %6s %6s\n', 'method', 'F1', 'Prec', 'Rec');
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'WECS_KI', F1_wecs_ki, Pr_wecs_ki, Re_wecs_ki);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'WECS_Otsu', F1_wecs_otsu, Pr_wecs_otsu, Re_wecs_otsu);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'TAAD_KI', F1_taad_ki, Pr_taad_ki, Re_taad_ki);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'TAAD_Otsu', F1_taad_otsu, Pr_taad_otsu, Re_taad_otsu);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'ECS_KI', F1_ecs_ki, Pr_ecs_ki, Re_ecs_ki);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'ECS_Otsu', F1_ecs_otsu, Pr_ecs_otsu, Re_ecs_otsu);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'CVA', F1_cva, Pr_cva, Re_cva);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'MAD', F1_mad, Pr_mad, Re_mad);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'PCA', F1_pca, Pr_pca, Re_pca);
fprintf(file_cd_scores, '%10s %5.4f %5.4f %5.4f\n', 'SFA', F1_sfa, Pr_sfa, Re_sfa);

fclose(file_cd_scores);

imgdiff = 2*255*(R_wecs > cutoff) - mChange;
tp = numel(find(imgdiff==255));

% detection of nonchange regions
[D_wecs,FA_wecs] = ROCcurveNew(R_wecs/max(R_wecs(:)), mChange); close
[D_taad,FA_taad] = ROCcurveNew(A_taad/max(A_taad(:)), mChange); close
[D_ecs,FA_ecs] = ROCcurveNew(R_ecs/max(R_ecs(:)),mChange); close

mImage = figure;
hold on
xlabel('False positive rate', 'FontSize', 13)
ylabel('True positive rate', 'FontSize', 13)
axis([0 1 0 1]);
axis square
plot(FA_wecs,D_wecs,'k')
plot(FA_taad,D_taad,'g-d')
plot(FA_ecs,D_ecs,':+')
legend('WECS','TAAD','ECS',...
    'Location','southeast', 'FontSize', 12)
legend('boxoff')
hold off
saveas(mImage,sprintf('forest_roc_change_v2.jpg'))

% saving CSV for measures in time
mResults = array2table([reshape(vd_wecs,1,n); reshape(vd_ecs,1,n)]');
mResults.Properties.VariableNames = {'vd_wecs' 'vd_ecs'};
writetable(mResults,'change_measures_time_v2.csv')

% saving CSV for ROC curves of WECS, TAAD and ECS
mResults = array2table([FA_wecs; D_wecs; FA_taad; D_taad; FA_ecs; D_ecs]');
mResults.Properties.VariableNames = {'FA_wecs' 'D_wecs' 'FA_taad' 'D_taad' 'FA_ecs' 'D_ecs'};
writetable(mResults,'ROC_curves_forest_v2.csv')



