function [F1,Pr,Re] = F1score(img,im0)
% Input
% img: matrix of computed using data with 0 (nonchanges) and 1 (change)
% im0: matrix of true change regions, with values 0 (nonchange) and 255
% (change)
% Output
% F1: F1 score
% Pr: Precision
% Re: Recall

% total number of positives
imtemp=255.*double(img);
D=find(imtemp==255);
nD=numel(D);
% True positive
imgdiff=2*imtemp-im0;
TP=find(imgdiff==255);
nTP=numel(TP);
% Precision
Pr= nTP/nD;
% Recall
FN=find(imgdiff==-255); % false negatives
nFN=numel(FN);
Re = nTP/(nTP + nFN);
% F1-score
F1 = 2*Pr*Re/(Pr + Re);


end

