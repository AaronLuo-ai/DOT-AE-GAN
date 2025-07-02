load("d_cord.mat");
mainFolder = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar22_2024 - Copy\757\-2\tar\dep_2\rad_1.35\ua_0.0195';
files = dir(fullfile(mainFolder, 'dref*.mat'));

measured_data = complex(zeros(9, 14)); 
for j = 1:9 % Iterate over the 9 sources 

    sortedFileNames = sort({files.name});
    path = fullfile(mainFolder, sortedFileNames{j});
    dref_FFT = Convert_to_Freq_Domain(path); 
    
    for k = 1:14 % Iterate over the 14 detectors for each source
        mes = dref_FFT(d_cord(k,1), d_cord(k, 2)); 
        measured_data(j, k) = mes; 
    end
end 
%800\-2\tar\dep_0.5\rad_0.75\ua_0.015
%800\-2\tar\dep_2\rad_1.35\ua_0.0195
file_name = sprintf('sample_data/measured_data_757_-2_tar2.mat'); 
save(file_name, "measured_data"); 