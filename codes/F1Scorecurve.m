function [t,F1,pD,Re]=F1Scorecurve(img,im0)
% Input
% img: matrix of computed using data with 0 (nonchanges) and 1 (change)
% im0: matrix of true change regions, with values 0 (nonchange) and 255
% (change)
% Output
% t: threshold values considered
% F1: F1 scores for each t value
% Pr: Precision for each t value
% Re: Recall for each t value

% Nombre reel de pixel ayant change
CP=find(im0~=0);
nCP=numel(CP);
%%%%%%%%%%%%%%%%
u=0.01;
v=.99;
pas=(v-u)/100;
t=u:pas:v;
vQuant = quantile(img(:),1-t); % quantiles in decreasing order
%t=u+2*pas:pas:v-2*pas;
lt=numel(t);
pD=zeros(1,lt);
Re=zeros(1,lt);
F1=zeros(1,lt);
for k=1:lt
    % the p% largest correlations, are those larger that the (1-p)th
    % quantile of R
    imtemp=255.*double(img>vQuant(k));
    % Detection
    D=find(imtemp==255);
    nD=numel(D);
    % True positive
    imgdiff=2*imtemp-im0;
    TP=find(imgdiff==255);
    nTP=numel(TP);
    % Precision
    pD(k)= nTP/nD;
    % Recall
    FN=find(imgdiff==-255);
    nFN=numel(FN);
    Re(k)= nTP/(nTP + nFN);
    % F1-score
    F1(k)=2*pD(k)*Re(k)/(pD(k) + Re(k));
end

%y=reshape(imtemp,1,lxy);
%z=reshape(im0,1,lxy);

%%% Plot ROC curves
figure
plot(t,F1);
%title('F1-score')
xlabel('p')
ylabel('F1-score')
axis([0 1 0 1]);
axis square