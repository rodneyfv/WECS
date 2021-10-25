function [pD,pFA]=F1Scorecurve(img,im0)

lxy=numel(im0);
% Nombre reel de pixel ayant change
CP=find(im0~=0);
nCP=numel(CP);
%%%%%%%%%%%%%%%%
u=0;
v=1;
pas=(v-u)/100;
t=u:pas:v;
%t=u+2*pas:pas:v-2*pas;
lt=numel(t);
pD=zeros(1,lt);
pFA=pD;
for k=1:lt
    %imtemp=255.*im2bw(img,k/T);
    imtemp=255.*double(img>t(k));
    % Detection
    D=find(imtemp==255);
    nD=numel(D);
    % True positive
    imgdiff=2*imtemp-im0;
    TP=find(imgdiff==255);
    nTP=numel(TP);
    % Precision
    pD(k)=nTP/nD;
    % Recall
    FN=find(imgdiff==-255);
    nFN=numel(FN);
    pD(k)=nTP/(nBD + );
    % Probabilite de fausse alarm
    pFA(k)=(nD-nTP)/(lxy-nCP);
end

%y=reshape(imtemp,1,lxy);
%z=reshape(im0,1,lxy);

%%% Plot ROC curves
figure
plot(pFA,pD);
title('ROC Curve')
xlabel('False positive rate')
ylabel('True positive rate')
axis([0 1 0 1]);
axis square