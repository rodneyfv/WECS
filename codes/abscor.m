function r = abscor(vx, vy)
% computes the absolute value of the Pearson correlation between vectors vx
% and vy
n = length(vx);
tmp = abs(corrcoef(reshape(vx,1,n),reshape(vy,1,n)));
r = tmp(1,2);
end

