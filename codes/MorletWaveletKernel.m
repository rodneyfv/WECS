function Ka = MorletWaveletKernel(x, y, a)
    % Input:
    % x, y: real values where the kernel is evaluated at
    % a: scale of the kernel
    % Output
    % Ka: Morlet wavelet kernel defined in Eq. (6) of 
    % Jia et al (2016, Remote-Sensing Image CD With Fusion of Multiple
    % Wavelet Kernels)
    d_tmp = (x-y)/a;
    Ka = cos(1.75*d_tmp)*exp(-(d_tmp^2)/2);
end

