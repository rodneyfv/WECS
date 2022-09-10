
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
t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{1,:})),'r');
Y = read(t);
imagesc(sqrt(Y(:,:,1).^2 + Y(:,:,2).^2))

%
[Nx,Ny,Nc] = size(Y);
% Nx * Ny : spatial dimension
% Nc : number of channels, here: VV and VH


%% JUST


% matrix of mean squared differences of approximation coefficients
tmp = zeros(n, 1);

% mean image
imRef = zeros(Nx,Ny);
for m=1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    imRef = imRef + double(Y(:,:,1).^2 + Y(:,:,2).^2);
    tmp(m) = Y(100,100,1).^2 + Y(100,100,2).^2;
end
imRef = imRef/n;
% constant to make the mean matrix have norm one
normconst = norm(imRef);
imRef = imRef/normconst;
tmp1 = tmp/normconst;


% loading the JUST functions
addpath(genpath('/home/rodney/Documents/JUST_11Mar2021/JUST'))


LocIndMagDir = JUSTjumps(t, f, 'P', P, 'save', 1); 

LocIndMagDir = JUSTjumps(tmp2, normalize(tmp),'size',40);
JumpInd = LocIndMagDir(:,2);
JUSTdecompose(t, f, 'ind', JumpInd);


% resolution level used on wavelet decompositions
J = 2;
meanSNR = 0;

for m=1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data = double(Y(:,:,1).^2 + Y(:,:,2).^2)/normconst;
    % extending image to be able to use swt2
    pwr2 = ceil(log2(min([Nx Ny])));
    extens = ceil((2^pwr2 - min([Nx Ny]))/2);
    tmp = wextend('2D','per',data,extens); % matrix whose dim is power of 2
    % wavelet approximation for log of VV chanel
    [tmp2,~,~,~] = swt2(tmp(1:2^pwr2,1:2^pwr2),J,wname);
    % sqared mean deviations of level J approx. coeff.
    mD_wecs(:,:,m) = (tmp2(extens+1:extens+Nx,extens+1:extens+Ny,J) - imRef).^2;
    % mean SNR of observed images
    meanSNR = meanSNR + ...
        norm(tmp2(extens+1:extens+Nx,extens+1:extens+Ny,J),'fro')/(norm(data,'fro')*n);
end
clear Y t tmp tmp2

% vector with total sum of squared mean deviations for each image
vd_wecs = reshape(sum(sum(mD_wecs,1),2),1,n);

% computing correlations
mCorr_wecs = zeros(Nx,Ny);
for ii=1:Nx
    for jj=1:Ny
        tmp = abs(corrcoef(reshape(mD_wecs(ii,jj,:),1,n),vd_wecs));
        mCorr_wecs(ii,jj) = tmp(1,2);
    end
end
toc

% mean SNR for observed images
meanSNR

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
saveas(mImage,sprintf('../figs/forest_vSumDifCoefSq.jpg'))

% checking the images corresponding to change points in time
for m = [25 27 30]
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data = double(Y(:,:,1).^2 + Y(:,:,2).^2)/normconst;
    
    % extending image to be able to use swt2
    pwr2 = ceil(log2(min([Nx Ny])));
    extens = ceil((2^pwr2 - min([Nx Ny]))/2);
    tmp = wextend('2D','per',data,extens); % matrix whose dim is power of 2
    % wavelet approximation for log of VV chanel
    [tmp2,~,~,~] = swt2(tmp(1:2^pwr2,1:2^pwr2),J,wname);

    save_tiff_image(tmp2(extens+1:extens+Nx,extens+1:extens+Ny,J),...
        sprintf('forest_changes_time_m%2d.tiff',m));
end
save_tiff_image(imRef,...
        sprintf('forest_changes_time_mean.tiff'));

% Saving the images of correlations
save_tiff_image(mCorr_wecs,...
        sprintf('forest_wecs_abscorr.tiff'));
mImage = figure;
imagesc(mCorr_wecs)
axis off; colorbar
set(gca,'FontSize',13)
% saveas(mImage,sprintf('../figs/forest_wecs_abscorr.jpg'))

% Change image using threshold suggested in feature screening literature
%histogram(mCorr(:))
mImage = figure;
cutoff = quantile(mCorr_wecs(:),1-1/log(Nx*Ny));
imshow(mCorr_wecs>cutoff)
%saveas(mImage,sprintf('../figs/forest_wecs_change_space.jpg'))
save_tiff_image(double(mCorr_wecs>cutoff),...
        sprintf('forest_wecs_abscorr_screethrs.tiff'));

cutoff_KI_wecs = kittler(mCorr_wecs); % Kittler-Illingworth threshold
cutoff_Otsu_wecs = graythresh(mCorr_wecs); % Otsu's threshold

tmp = double(mCorr_wecs>cutoff_KI_wecs);
save_tiff_image(tmp,...
        sprintf('forest_wecs_change_space_KI.tiff'));
tmp = double(mCorr_wecs>cutoff_Otsu_wecs);
save_tiff_image(tmp,...
        sprintf('forest_wecs_change_space_otsu.tiff'));




