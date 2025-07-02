% clear all, close all, clc; 
% %d_cord = load('d_cord.mat');
% load('d_cord.mat') % detector positions are loaded here 
% Folder757 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar14_2024 - Copy\757';
% Folder800 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar14_2024 - Copy\800'; 
% Folder850 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar14_2024 - Copy\850';
% wavelength_dirs = [Folder757; Folder800; Folder850]; 
% 
% for m= 1:3
%     Folder_s_wavelength = wavelength_dirs(m, :);
% 
%     subFold_angles = dir(Folder_s_wavelength); 
%     subFold_angles = subFold_angles([subFold_angles.isdir]);
% 
%     for i = 1:numel(subFold_angles)
%         ref_folder = fullfile(Folder_s_wavelength, subFold_angles(i).name, "ref"); %sprintf('%s\ref',subFold_angles(i).folder); 
%         files = dir(fullfile(ref_folder, 'dref*.mat'));
%         measured_data = complex(zeros(9, 14)); 
%         for j = 1: 9 % interate over the 9 source 
%             fileNames = {files.name};
%             sortedFileNames = sort(fileNames);
% 
%             path = fullfile(files(j).folder, sortedFileNames(j)); 
% 
%             dref_FFT = Convert_to_Freq_Domain(path); 
%             for k = 1: 14 % iterate over the 14 detector for each source
%                 mes = dref_FFT(d_cord(k,1), d_cord(k, 2)); 
%                 measured_data(j, k) = mes; 
%             end
%             %measured_data(j,:) =  mes; 
%         end 
%         subFold_angles(i).name; 
%         figure, plot(abs(measured_data(:)));
%         measured_data_all_angles(i) = measured_data;
%     end
%     measured_data_all_wavelengths(m) = measured_data_all_angles; 
% end


clear all, close all, clc; 
load('d_cord.mat') % Load detector positions

% Folder757 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar22_2024 - Copy\757';
% Folder800 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar22_2024 - Copy\800'; 
% Folder850 = 'C:\Users\mdiqb\Desktop\Matlab_DOT_Reconstruction\Lukai\Mar22_2024 - Copy\850';

% Define the main folders
mainFolders = {'757', '800', '850'};

% Loop through each main folder
for f = 1:numel(mainFolders)
    mainFolder = mainFolders{f};
    
    % Traverse through the directory structure
    angleDirs = dir(fullfile(mainFolder, '*'));
    angleDirs = angleDirs([angleDirs.isdir] & ~startsWith({angleDirs.name}, '.')); % Exclude hidden folders
    
    % Loop through each angle directory
    for a = 1:numel(angleDirs)
        angleDir = angleDirs(a).name;
        
        % Traverse through the rest of the directory structure
        depthDirs = dir(fullfile(mainFolder, angleDir, 'tar', '*'));
        depthDirs = depthDirs([depthDirs.isdir] & ~startsWith({depthDirs.name}, '.')); % Exclude hidden folders
        
        % Loop through each depth directory
        for d = 1:numel(depthDirs)
            depthDir = depthDirs(d).name;
            
            radiusDirs = dir(fullfile(mainFolder, angleDir, 'tar', depthDir, '*'));
            radiusDirs = radiusDirs([radiusDirs.isdir] & ~startsWith({radiusDirs.name}, '.')); % Exclude hidden folders
            
            % Loop through each radius directory
            for r = 1:numel(radiusDirs)
                radiusDir = radiusDirs(r).name;
                
                uaDirs = dir(fullfile(mainFolder, angleDir, 'tar', depthDir, radiusDir, '*'));
                uaDirs = uaDirs([uaDirs.isdir] & ~startsWith({uaDirs.name}, '.')); % Exclude hidden folders
                
                % Loop through each ua directory
                for u = 1:numel(uaDirs)
                    uaDir = uaDirs(u).name;
                    
                    % Create the new folder for the main folder
                    newFolder = fullfile('new',angleDir, 'tar', depthDir, radiusDir, uaDir, mainFolder);
                    if ~exist(newFolder, 'dir')
                        mkdir(newFolder);
                    end
                    
                    % Move the measured_data_files to the new folder
                    oldFiles = fullfile(mainFolder, angleDir, 'tar', depthDir, radiusDir, uaDir, '*.mat');
                    newFiles =newFolder; % fullfile(newFolder, 'measured_data_files');
                    if ~exist(newFiles, 'dir')
                        mkdir(newFiles);
                    end
                    copyfile(oldFiles, newFolder);
                end
            end
        end
    end
end


    

