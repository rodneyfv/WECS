
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

%% Configurations for wavelet decompositions

wname='sym8';
%wname='db2';

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

% matrix of mean squared differences of approximation coefficients
mDifCoefSq = zeros([Nx, Ny, n]);

% mean image
imRef = zeros(Nx,Ny);
for m=1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    imRef = imRef + double(Y(:,:,1).^2 + Y(:,:,2).^2);
end
imRef = imRef/n;
normconst = norm(imRef);
imRef = imRef/normconst;

% resolution level used on wavelet decompositions
J = 2;

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
    mDifCoefSq(:,:,m) = (tmp2(extens+1:extens+Nx,extens+1:extens+Ny,J) - imRef).^2;
end
clear Y t tmp tmp2

% vector with total sum of squared mean deviations for each image
vSumDifCoefSq = reshape(sum(sum(mDifCoefSq,1),2),1,n);

% plot of mean squared deviations
mImage = figure;
plot(1:n, vSumDifCoefSq,'LineWidth',2);
hold on
tmp = mad(vSumDifCoefSq);
plot(1:n, (2*tmp + median(vSumDifCoefSq))*ones(1,n),'LineWidth',2)
xlabel('$m$','interpreter','latex','FontSize',20); xlim([0 n])
ylabel('$\textbf{d}(m)$','interpreter','latex','FontSize',20);
set(gca,'FontSize',13)
hold off
saveas(mImage,sprintf('../figs/forest_vSumDifCoefSq.jpg'))

% checking the images corresponding to change points in time
mImage = figure;
cont = 1;
for m = [25 27 30]
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data = double(Y(:,:,1).^2 + Y(:,:,2).^2);
    subplot(2,2,cont); cont = cont + 1;
    imagesc(data)
    title(sprintf('m=%2d',m)); axis off;
end
subplot(2,2,4)
imagesc(imRef); title('Mean image'); axis off;
saveas(mImage,sprintf('../figs/forest_changes_time.jpg'))


% computing correlations
mCorr = zeros(Nx,Ny);
for ii=1:Nx
    for jj=1:Ny
        tmp = abs(corrcoef(reshape(mDifCoefSq(ii,jj,:),1,n),vSumDifCoefSq));
        mCorr(ii,jj) = tmp(1,2);
    end
end

mImage = figure;
imagesc(mCorr)
axis off; colorbar
set(gca,'FontSize',13)
saveas(mImage,sprintf('../figs/forest_wecs_abscorr.jpg'))


%histogram(mCorr(:))
mImage = figure;
cutoff = quantile(mCorr(:),1-1/log(Nx*Ny));
imshow(mCorr>cutoff)
saveas(mImage,sprintf('../figs/forest_wecs_change_space.jpg'))

% Aggregation of differences
S = zeros(Nx,Ny);

t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{1,:})),'r');
Y = read(t);
data1 = double(Y(:,:,1).^2 + Y(:,:,2).^2)/normconst;
for m=2:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data2 = double(Y(:,:,1).^2 + Y(:,:,2).^2)/normconst;    
    S = S + abs(data2 - data1);
    data2 = data1;
end
S = S./max(S(:));

mImage = figure;
imagesc(S)
axis off; colorbar
set(gca,'FontSize',13)
saveas(mImage,sprintf('../figs/forest_aggreg_ratios.jpg'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% making a video with the image time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create the video writer with 1 fps
writer_im = VideoWriter('images-ts.avi');
% number of frames shown in a second
writer_im.FrameRate = 2;
% open the video writer
open(writer_im);
% write the frames to the video
for m = 1:n
    t = Tiff(strcat('../../Images/timeSeries/ascending/',char(dates{m,:})),'r');
    Y = read(t);
    data = double(Y(:,:,1).^2 + Y(:,:,2).^2);
    imagesc(data)
    title(sprintf('m=%2d',m)); axis off;
    frame = getframe(gcf) ;
    writeVideo(writer_im, frame);    
end
% close the writer object
close(writer_im);


