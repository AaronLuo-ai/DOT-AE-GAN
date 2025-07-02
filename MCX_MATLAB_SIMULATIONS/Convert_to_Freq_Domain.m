function dref_FFT = Convert_to_Freq_Domain(path)
%load('C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Usc_data_testing\-15\ref\dref0_1.mat'); 
load(path); 

tend = 7.1428e-09; %2e-9;
tg = 200;
tstep = tend/tg;
% DOT frequency  
freq=140e6;
%% Extract component of 140 MHz (learnt from Shuying's code: MC_Measure_target.m)
Fs=1/tstep;
df=Fs/tg;
frequency=-Fs/2:df:Fs/2-df;
% freqIdx=find(abs(frequency-freq)<3e06);
% Find the element closest to the DOT frequency
[min_diff, freqIdx] = min(abs(frequency-freq));

% since the detector is around 2.5 pixels in radius 
filter_kernel = ones(9, 9, 1);
filter_kernel = filter_kernel / sum(filter_kernel(:));

summed_response = imfilter(dref, filter_kernel, 'replicate');

% Component strength of 140MHz
dref_FFT = zeros(size(dref,1),size(dref,2));
for xdex = 1:size(dref,1)
    for ydex = 1:size(dref,2)
        tempFFT=fftshift(fft(squeeze(tstep.*summed_response(xdex,ydex,:))));
        dref_FFT(xdex,ydex) = tempFFT(freqIdx);
    end
end
end

