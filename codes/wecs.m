function [R,vd] = wecs(mY,wname,J)
% Function to apply the WECS to a time series of images
% Input
% mY: array of observed images (dimensions must be a power of 2)
% wname: wavelet basis
% J: resolution level of wavelet decomposition

% Output
% R: matrix of correlations of local squared differences with vd
% vd: overall squared mean differences

% image dimension and sample size
[n1,n2,n] = size(mY);

% mean observed image
imRef = mean(mY,3);

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

% matrix of correlations
R = zeros(n1,n2);
for ii=1:n1
   for jj=1:n2
       tmp = abs(corrcoef(reshape(mD(ii,jj,:),n,1),vd));
       R(ii,jj) = tmp(1,2);
   end
end


end

