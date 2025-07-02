% Load the .mat file
clear, close, clc; 
load('Q3.Labtest2.mat'); % Replace 'your_image.mat' with the path to your .mat file

% Display the variables loaded from the .mat file
whos

% Access the image data directly
% Assuming there's only one variable loaded and it's the image data
originalImage = cameraman; % Replace 'your_loaded_variable' with the actual variable name

% Convert the image to double precision for processing
originalImage = im2double(originalImage);

% Display the original image
figure;
subplot(1,2,1);
imshow(originalImage);
title('Original Image');


% Perform Fourier Transform
fftImage = fft2(originalImage);

% Shift the zero-frequency component to the center
fftImageShifted = fftshift(fftImage);

% Display the magnitude spectrum
subplot(1,2,2);
imagesc(log(abs(fftImageShifted)+1)); % Log transform for better visualization
colormap('gray');
title('Magnitude Spectrum');

% Define a notch filter to remove the sinusoidal noise at 10 Hz
[x_freq, y_freq] = meshgrid(-size(originalImage,2)/2:size(originalImage,2)/2-1, -size(originalImage,1)/2:size(originalImage,1)/2-1);
notchFilter = ones(size(originalImage));
%notchFilter((x_freq == 10) & (y_freq == 0)) = 0; % Centered at 10 Hz in x direction
notchFilter((x_freq == 10 ) ) = 0;
figure;
imshow(abs(notchFilter), []);
title('Notch Image');

% Apply the notch filter to the Fourier transformed image
filteredFFTImage = fftImageShifted .* notchFilter;

% Perform inverse Fourier Transform to get the filtered image
filteredImage = ifft2(ifftshift(filteredFFTImage));

% Display the filtered image
figure;
imshow(abs(filteredImage), []);
title('Filtered Image');
