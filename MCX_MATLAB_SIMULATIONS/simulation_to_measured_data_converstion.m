clear all, close all, clc; 
%d_cord = load('d_cord.mat');
load('d_cord.mat') % detector positions are loaded here 
Folder757 = fullfile(pwd,'757');
Folder800 = fullfile(pwd,'800'); 
Folder850 = fullfile(pwd,'850');
wavelength_dirs = [Folder757; Folder800; Folder850]; 

for m= 1:3
    Folder_s_wavelength = wavelength_dirs(m, :);

    subFold_angles = dir(Folder_s_wavelength); 
    subFold_angles = subFold_angles([subFold_angles.isdir] & ~startsWith({subFold_angles.name}, '.'));

    for i = 1:numel(subFold_angles)
        ref_folder = fullfile(Folder_s_wavelength, subFold_angles(i).name, "ref"); %sprintf('%s\ref',subFold_angles(i).folder); 
        files = dir(fullfile(ref_folder, 'dref*.mat'));
        measured_data = complex(zeros(9, 14)); 
        for j = 1: 9 % interate over the 9 source 
            fileNames = {files.name};
            sortedFileNames = sort(fileNames);

            path = fullfile(files(j).folder, sortedFileNames(j)); 

            dref_FFT = Convert_to_Freq_Domain(path{1}); 
            for k = 1: 14 % iterate over the 14 detector for each source
                mes = dref_FFT(d_cord(k,1), d_cord(k, 2)); 
                measured_data(j, k) = mes; 
            end
            %measured_data(j,:) =  mes; 
        end 
        subFold_angles(i).name; 
        figure, plot(abs(measured_data(:)));
        measured_data_all_angles{i} = measured_data;
    end
    measured_data_all_wavelengths{m} = measured_data_all_angles; 
end


% clear all, close all, clc; 
% load('d_cord.mat') % Load detector positions
% 
% Folder757 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar14_2024 - Copy\757';
% Folder800 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar14_2024 - Copy\800'; 
% Folder850 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar14_2024 - Copy\850';
% wavelength_dirs = {Folder757, Folder800, Folder850}; 
% angles_of_phantom = {'-2', '-1', '0', '1', '2'}; 
% 
% measured_data_all_wavelengths = cell(1, 3); % Initialize cell array to store measured data
% 
% for m = 1:3
%     Folder_s_wavelength = wavelength_dirs{m};
% 
%     % subFold_angles = dir(Folder_s_wavelength); 
%     % subFold_angles = subFold_angles([subFold_angles.isdir]);
%     % 
%     measured_data_all_angles = cell(1, numel(angles_of_phantom)); % Initialize cell array to store measured data for each angle
% 
%     for i = 1:numel(angles_of_phantom)
%         ref_folder = fullfile(Folder_s_wavelength, angles_of_phantom{i}, 'ref');
%         files = dir(fullfile(ref_folder, 'dref*.mat'));
% 
%         measured_data = complex(zeros(9, 14)); 
% 
%         for j = 1:9 % Iterate over the 9 sources 
%             sortedFileNames = sort({files.name});
%             path = fullfile(ref_folder, sortedFileNames{j}); 
%             dref_FFT = Convert_to_Freq_Domain(path); 
% 
%             for k = 1:14 % Iterate over the 14 detectors for each source
%                 mes = dref_FFT(d_cord(k,1), d_cord(k, 2)); 
%                 measured_data(j, k) = mes; 
%             end
%         end 
% 
%         % Plot the measured data
%         % figure; 
%         % plot(abs(measured_data(:)));
%         % title(sprintf('Measured Data - Wavelength %d, Angle %d', m, i));
%         % 
%         measured_data_all_angles{i} = measured_data;
%     end
% 
%     measured_data_all_wavelengths{m} = measured_data_all_angles; 
% 
% end


    

