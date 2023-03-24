
% Evaluation of different methods

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

% mean observed image
imRef = mean(mY,3);

%%
% original images

% Subsampling of images
subsamplingfactor = 2;
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
mI(:,:,1) = im1 * 10; mI(:,:,2) = im2 * 10;
mI(:,:,3) = im3 * 10; mI(:,:,4) = im4 * 10;
clear im1 im2 im3 im4

% all changes that are expected to be detected
totalchanges = (abs(mI(:,:,1)-mI(:,:,2)) + abs(mI(:,:,2)-mI(:,:,3)) + ...
    abs(mI(:,:,3)-mI(:,:,4)) + abs(mI(:,:,4)-mI(:,:,1)))>0;
mImage = figure;
colormap(gray(256)); imagesc(totalchanges)
axis off
title('Total changes', 'FontSize', 17)
clear mI;

%% WECS

wname = 'db2'; % wavelet basis used
J = 3; % resolution level of wavelet transform

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
axis off
%title('db2 WECS d(m), J=2', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/Rwecs_v2.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/Rwecs_v2.jpg'),'Resolution',300)


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
%title('d(m) without wavelets', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/R_ECS.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/R_ECS.jpg'),'Resolution',300)


%% Standard change detection

% matrix of aggregated absolute differences
A_taad = zeros(n1,n2);
for m=2:n
    A_taad = A_taad + abs(mY(:,:,m) - mY(:,:,m-1));
end
A_taad = A_taad./max(A_taad(:));

%
mImage = figure;
imshow(A_taad)
axis off
%title('Aggregation of absolute differences'                                                                                                                                                                                                                                                                                                                               , 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/A_taad.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/A_taad.jpg'),'Resolution',300)


%% Celik et al (2010) - CD DTCWT

% bilinear interpolation
phi_im1 = imresize(mY(:,:,1), [2*n1, 2*n2], 'Method', 'bilinear');
phi_im2 = imresize(mY(:,:,n), [2*n1, 2*n2], 'Method', 'bilinear');

% Compute the dual-tree complex wavelet transform
nlevels = 3;
[~, Dim1] = dualtree2(phi_im1,'Level',nlevels,'LevelOneFilter','nearsym5_7');
[~, Dim2] = dualtree2(phi_im2,'Level',nlevels,'LevelOneFilter','nearsym5_7');
clear phi_im1;
clear phi_im2;

D_s = cell([nlevels, 1]);
B_s = cell([nlevels, 1]);

for ii=1:nlevels
    D_s{ii} = abs(Dim1{ii} - Dim2{ii});
    B_s{ii} = zeros(size(D_s{ii}));
    for jj=1:6
        [nr_ds, nc_ds] = size(D_s{ii}(:,:,jj));
        vx = reshape(D_s{ii}(:,:,jj), [nr_ds*nc_ds, 1]);
        gm_model = fitgmdist(vx, 2);

        [~, id] = sort(gm_model.ComponentProportion,'ascend');
        pdf_g1 = normpdf(vx, gm_model.mu(id(1)), gm_model.Sigma(id(1)));
        pdf_g2 = normpdf(vx, gm_model.mu(id(2)), gm_model.Sigma(id(2)));
        vx_id = gm_model.ComponentProportion(id(1))*pdf_g1 > gm_model.ComponentProportion(id(2))*pdf_g2;

        B_s{ii}(:,:,jj) = reshape(vx_id, [nr_ds, nc_ds]); 
    end
end


bcm_s = cell([nlevels, 1]);
for ii=1:nlevels
    bcm_s{ii} = sum(D_s{ii}, 3)>0;
end
clear D_s;
clear B_s;

bcm = ones(n1, n2);
for ii=1:nlevels
    bcm = bcm .* imresize(bcm_s{ii}, [n1, n2], 'Method', 'nearest');
end


mImage = figure;
imshow( (bcm - min(min(bcm)))/ (max(max(bcm)) - min(min(bcm))) )
axis off
%title('Fusion of multiple wavelet kernels', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/DTCWT_Celik_etal2010.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/DTCWT_Celik_etal2010.jpg'),'Resolution',300)


%% Jia et al (2016, JSTARS) - FMW kernel

% subtraction and ratio images
eps_im = 0.01;
mS = abs(mY(:,:,1) - mY(:,:,n));
mR = max(mY(:,:,1), mY(:,:,n))./(min(mY(:,:,1), mY(:,:,n)) + eps_im);

% normalizing the images to [0,1]
im1_norm = (mY(:,:,1) - min(min(mY(:,:,1))))./(max(max(mY(:,:,1))) - min(min(mY(:,:,1))));
im2_norm = (mY(:,:,n) - min(min(mY(:,:,n))))./(max(max(mY(:,:,n))) - min(min(mY(:,:,n))));
mS_norm = (mS - min(min(mS)))./(max(max(mS)) - min(min(mS)));
mR_norm = (mR - min(min(mR)))./(max(max(mR)) - min(min(mR)));
clear mS;
clear mR;

% length of the nonoverlapping blocks
len_block = 8;

% the blocks make an image of nr_block rows and nc_block columns
nr_block = n1/len_block; % number of rows
nc_block = n2/len_block; % number of columns

% Initial change detection to create pseudo-training samples. Notice that
% it only uses the subtraction image

% the average block
W_S_avg = zeros(len_block);
for ii=1:nr_block
    for jj=1:nc_block
        range_row = ((ii-1)*len_block+1):(ii*len_block);
        range_col = ((jj-1)*len_block+1):(jj*len_block);
        
        W_S_avg = W_S_avg + mS_norm(range_row, range_col);
    end
end
W_S_avg = W_S_avg./(nr_block * nc_block);

% the covariance matrix of these blocks
Co_S_avg = zeros(len_block);
for ii=1:nr_block
    for jj=1:nc_block
        range_row = ((ii-1)*len_block+1):(ii*len_block);
        range_col = ((jj-1)*len_block+1):(jj*len_block);
        
        mDelta = mS_norm(range_row, range_col) - W_S_avg;
        Co_S_avg = Co_S_avg + mDelta*mDelta';
    end
end
Co_S_avg = Co_S_avg./(nr_block * nc_block);

% eigenvectors and corresponding eigenvalues (ordered)
[mV_S, mD] = eig(Co_S_avg);
mD = diag(mD);
[~,tmp] = sort(mD,'descend');
mD = mD(tmp);
mV_S = mV_S(:, tmp);

% loadings of the blocks corresponding to the first principal component
mVecSpace = zeros(nc_block*nr_block, 2 + len_block);
kk = 1;
for ii=1:nr_block
    for jj=1:nc_block
        range_row = ((ii-1)*len_block+1):(ii*len_block);
        range_col = ((jj-1)*len_block+1):(jj*len_block);
        % position corresponding to the current row
        mVecSpace(kk, 1) = (ii-1)*len_block+1 + len_block/2;
        mVecSpace(kk, 2) = (jj-1)*len_block+1 + len_block/2;
        
        % loadings
        tmp = mV_S(:,1)' * mS_norm(range_row, range_col);
        mVecSpace(kk, 3:(2+len_block)) = tmp';

        kk = kk + 1;
    end
end

% k-means with two groups, where the data are the loadings of each block
[id_kmeans, ~, ~, D_kmeans] = kmeans(mVecSpace(:,3:(2+len_block)), 2);
tabulate(id_kmeans)

% indices of pixels among those with 1% lowest distances to the centroids
tmp = find(id_kmeans==1);     % group 1
[~, id_tmp] = sort(D_kmeans(tmp,1),'ascend');
id_perc_g1 = tmp(id_tmp(1:ceil(length(tmp)/100)));

tmp = find(id_kmeans==2);     % group 2
[~, id_tmp] = sort(D_kmeans(tmp,2),'ascend');
id_perc_g2 = tmp(id_tmp(1:ceil(length(tmp)/100)));


% plot of the image and central points at each block with their respective
% classification from the k-means results
subplot(1,2,1)
imagesc(mS_norm);
title('Subtraction image')
hold on
tmp1 = mVecSpace(id_perc_g1,1);
tmp2 = mVecSpace(id_perc_g1,2);
plot(tmp2, tmp1, 'x', 'MarkerSize', 5, 'Color', 'green', 'LineWidth', 2)

tmp1 = mVecSpace(id_perc_g2,1);
tmp2 = mVecSpace(id_perc_g2,2);
plot(tmp2, tmp1, 'x', 'MarkerSize', 5, 'Color', 'red', 'LineWidth', 2)

subplot(1,2,2)
imagesc(mR_norm);
title('Ratio image')
hold on
tmp1 = mVecSpace(id_perc_g1,1);
tmp2 = mVecSpace(id_perc_g1,2);
plot(tmp2, tmp1, 'x', 'MarkerSize', 5, 'Color', 'green', 'LineWidth', 2)

tmp1 = mVecSpace(id_perc_g2,1);
tmp2 = mVecSpace(id_perc_g2,2);
plot(tmp2, tmp1, 'x', 'MarkerSize', 5, 'Color', 'red', 'LineWidth', 2)


% Scale values
va = linspace(0.01, .99, 10);
% wavelet coefficients for each scale value
mK_S_g1 = zeros(n1, n2, length(va));
mK_S_g2 = zeros(n1, n2, length(va));

mK_R_g1 = zeros(n1, n2, length(va));
mK_R_g2 = zeros(n1, n2, length(va));


for ii=1:n1
    for jj=1:n2
        for a=1:length(va)            
            for m=1:length(id_perc_g1)
                i_tmp = mVecSpace(id_perc_g1(m),1);
                j_tmp = mVecSpace(id_perc_g1(m),2);
                d_tmp = MorletWaveletKernel(mS_norm(ii, jj), mS_norm(i_tmp, j_tmp), va(a));
                mK_S_g1(ii, jj, a) = mK_S_g1(ii, jj, a) + d_tmp;
                
                d_tmp = MorletWaveletKernel(mR_norm(ii, jj), mR_norm(i_tmp, j_tmp), va(a));
                mK_R_g1(ii, jj, a) = mK_R_g1(ii, jj, a) + d_tmp;
            end
            mK_S_g1(ii, jj, a) = mK_S_g1(ii, jj, a)/length(id_perc_g1);
            mK_R_g1(ii, jj, a) = mK_R_g1(ii, jj, a)/length(id_perc_g1);
            
            for m=1:length(id_perc_g2)
                i_tmp = mVecSpace(id_perc_g2(m),1);
                j_tmp = mVecSpace(id_perc_g2(m),2);
                d_tmp = MorletWaveletKernel(mS_norm(ii, jj), mS_norm(i_tmp, j_tmp), va(a));
                mK_S_g2(ii, jj, a) = mK_S_g2(ii, jj, a) + d_tmp;
                
                d_tmp = MorletWaveletKernel(mR_norm(ii, jj), mR_norm(i_tmp, j_tmp), va(a));
                mK_R_g2(ii, jj, a) = mK_R_g2(ii, jj, a) + d_tmp;
            end
            mK_S_g2(ii, jj, a) = mK_S_g2(ii, jj, a)/length(id_perc_g2);
            mK_R_g2(ii, jj, a) = mK_R_g2(ii, jj, a)/length(id_perc_g2);
            
        end
    end
end

% correlation of the wavelet coefficients at each scale with corresponding
% difference images
vCor_S_g1 = zeros(length(va), 1);
vCor_S_g2 = zeros(length(va), 1);

vCor_R_g1 = zeros(length(va), 1);
vCor_R_g2 = zeros(length(va), 1);

for a=1:length(va)
    tmp = mK_S_g1(:, :, a);
    vCor_S_g1(a) = corr(tmp(:), mS_norm(:));
    tmp = mK_S_g2(:, :, a);
    vCor_S_g2(a) = corr(tmp(:), mS_norm(:));
    
    tmp = mK_R_g1(:, :, a);
    vCor_R_g1(a) = corr(tmp(:), mR_norm(:));
    tmp = mK_R_g2(:, :, a);
    vCor_R_g2(a) = corr(tmp(:), mR_norm(:));
end

% index of the scale whose correlation above is highest at the change group
% (optimal scale)
[~, id_S_a_opt] = max(vCor_S_g2);
[~, id_R_a_opt] = max(vCor_R_g2);

subplot(1,2,1)
imagesc(mK_S_g2(:, :, id_S_a_opt))
title('Subtraction image at optimal scale')
subplot(1,2,2)
imagesc(mK_R_g2(:, :, id_R_a_opt))
title('Ratio image at optimal scale')


% matrix of the multiple wavelet fusion (MWF) kernel
mFK_g1 = zeros(n1, n2, length(id_perc_g1));
mFK_g2 = zeros(n1, n2,length(id_perc_g2));
lamb = 0.01;
% the pixel class is set as the group with highest mean MWF value
mClass = zeros(n1, n2);

for ii=1:n1
    for jj=1:n2
        for m=1:length(id_perc_g1)
           mFK_g1(ii, jj, m) = mean(mK_R_g1(ii, jj, :));
           if ii==mVecSpace(id_perc_g1(m), 1) && jj==mVecSpace(id_perc_g1(m), 2)
               mFK_g1(ii, jj, m) = mFK_g1(ii, jj, m) + lamb;
           end
        end
        
        for m=1:length(id_perc_g2)
           mFK_g2(ii, jj, m) = mean(mK_S_g2(ii, jj, :));
           if ii==mVecSpace(id_perc_g2(m), 1) && jj==mVecSpace(id_perc_g2(m), 2)
               mFK_g2(ii, jj, m) = mFK_g2(ii, jj, m) + lamb;
           end
        end
        
        tmp1 = mean(mFK_g1(ii, jj, :))^2;
        tmp2 = mean(mFK_g2(ii, jj, :))^2;
        if tmp1 > tmp2
           mClass(ii, jj) = 1;
        else
            mClass(ii, jj) = 2;
        end
    end
end
clear mFK_g1;
clear mFK_g2;
clear mK_S_g1;
clear mK_S_g2;
clear mK_R_g1;
clear mK_R_g2;


%save('../figs/v2/data_FMW_Jia_etal2016.mat','mClass');
%mClass = matfile('../figs/v2/data_FMW_Jia_etal2016.mat');
%mClass = abs(mClass - 2);
mImage = figure;
imshow((mClass - min(min(mClass)))./(max(max(mClass)) - min(min(mClass))))
axis off
%title('Fusion of multiple wavelet kernels', 'FontSize', 17)
%saveas(mImage,sprintf('../figs/v2/CD_FMW_Jia_etal2016.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/CD_FMW_Jia_etal2016.jpg'),'Resolution',300)


%% Johnson and Kasischke (1998) - Change vector analysis (CVA)

t = Tiff(strcat('./Python/Methodology/Traditional/CVA/CVA_simul_im.tiff'),'r');
cva_cd_map = read(t);

mImage = figure;
imshow(cva_cd_map/max(max(cva_cd_map)))
axis off
%saveas(mImage,sprintf('../figs/v2/CVA_simul.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/CVA_simul.jpg'),'Resolution',300)


%% Nielsen (2007) - regularized iteratively reweighted MAD method

t = Tiff(strcat('./Python/Methodology/Traditional/MAD/IRMAD_simul_im.tiff'),'r');
mad_cd_map = read(t);

mImage = figure;
imshow(mad_cd_map/max(max(mad_cd_map)))
axis off
%saveas(mImage,sprintf('../figs/v2/mad_simul.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/mad_simul.jpg'),'Resolution',300)


%% Celik (2009) - CD via principal component analysis and k-means

t = Tiff(strcat('./Python/Methodology/Traditional/PCAKmeans/PCAKmeans_simulated_images.tiff'),'r');
pca_cd_map = read(t);

mImage = figure;
imshow(pca_cd_map/max(max(pca_cd_map)))
axis off
%saveas(mImage,sprintf('../figs/v2/pca_simul.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/pca_simul.jpg'),'Resolution',300)


%% Wu, Du, Zhang (2014) - slow feature analysis for change detection

t = Tiff(strcat('./Python/Methodology/Traditional/SFA/ISFA_simul_im.tiff'),'r');
isfa_cd_map = read(t);

mImage = figure;
imshow(isfa_cd_map/max(max(isfa_cd_map)))
axis off
exportgraphics(mImage,sprintf('../figs/v2/isfa_simul.jpg'),'Resolution',300)


%% Compare all results

[TP_wecs,FP_wecs]=ROCcurveNew(R_wecs,255*totalchanges); close
[TP_ecs,FP_ecs] = ROCcurveNew(R_ecs,255*totalchanges); close
[TP_agg,FP_agg]=ROCcurveNew(A_taad,255*totalchanges); close

% saving results in a csv file
mResults = array2table([TP_wecs; FP_wecs; TP_ecs; FP_ecs; TP_agg; FP_agg]');
mResults.Properties.VariableNames = {'TP_wecs' 'FP_wecs' 'TP_ecs' 'FP_ecs' 'TP_agg' 'FP_agg'};
writetable(mResults,'SeqEllipse_methods_ROC_v2.csv')

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
%saveas(mImage,sprintf('../figs/v2/methods_ROC.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/methods_ROC.jpg'),'Resolution',300)

%

[vp_wecs,FP_wecs,~,~] = F1Scorecurve(R_wecs,255*totalchanges); close
[vp_ecs,FP_ecs,~,~] = F1Scorecurve(R_ecs,255*totalchanges); close
[vp_agg,FP_agg,~,~] = F1Scorecurve(A_taad,255*totalchanges); close

mResults = array2table([vp_wecs; FP_wecs; vp_ecs; FP_ecs; vp_agg; FP_agg]');
mResults.Properties.VariableNames = {'vp_wecs' 'FP_wecs' 'vp_ecs' 'FP_ecs' 'vp_agg' 'FP_agg'};
writetable(mResults,'SeqEllipse_methods_F1score_v2.csv')


mImage = figure;
hold on
%title('F1-score', 'FontSize', 17)
xlabel('$p$','interpreter','latex', 'FontSize', 15)
ylabel('$F_1$-score','interpreter','latex', 'FontSize', 15)
axis([0 1 0 0.99]);
axis square
plot(vp_wecs,FP_wecs,'k','LineWidth',2)
plot(vp_ecs,FP_ecs,'g-d','LineWidth',2)
plot(vp_agg,FP_agg,':+','LineWidth',2)
grid on
legend('WECS', 'ECS', ...
    'TAAD','interpreter','latex', 'Location','southeast', 'FontSize', 15)
legend('boxoff')
hold off
%saveas(mImage,sprintf('../figs/v2/methods_comparison_F1score.jpg'))
exportgraphics(mImage,sprintf('../figs/v2/methods_comparison_F1score.jpg'),'Resolution',300)



return


