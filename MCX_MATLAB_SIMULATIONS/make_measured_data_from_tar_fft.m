clear all, close all, clc; 
load('d_cord.mat') % Load detector positions
 
% angles_of_phantom = {'-2', '-1', '0', '1', '2'}; 
% 
% measured_data_all_angles = cell(1, numel(angles_of_phantom)); % Initialize cell array to store measured data for each angle
% 
% for i = 1:numel(angles_of_phantom)
%     ref_folder = fullfile(Folder_s_wavelength, angles_of_phantom{i}, 'ref');
%     files = dir(fullfile(ref_folder, 'dref*.mat'));
% 
%     measured_data = complex(zeros(9, 14)); 
% 
%     for j = 1:9 % Iterate over the 9 sources 
%         sortedFileNames = sort({files.name});
%         path = fullfile(ref_folder, sortedFileNames{j}); 
%         dref_FFT = Convert_to_Freq_Domain(path); 
% 
%         for k = 1:14 % Iterate over the 14 detectors for each source
%             mes = dref_FFT(d_cord(k,1), d_cord(k, 2)); 
%             measured_data(j, k) = mes; 
%         end
%     end 
% 
%     % Plot the measured data
%     % figure; 
%     % plot(abs(measured_data(:)));
%     % title(sprintf('Measured Data - Wavelength %d, Angle %d', m, i));
%     % 
%     measured_data_all_angles{i} = measured_data;
% end
% 


mainFolder = 'new'; 

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
                waveDirs = dir(fullfile(mainFolder, angleDir, 'tar', depthDir, radiusDir, uaDir, '*'));
   
                waveDirs = waveDirs([waveDirs.isdir] & ~startsWith({waveDirs.name}, '.'));
                
                measured_data = complex(zeros(3, 9, 14));

                for w= 1:numel(waveDirs)
                    waveDir = waveDirs(w).name;

                    files = dir(fullfile(mainFolder, angleDir, 'tar', depthDir, radiusDir, uaDir, waveDir, 'dref*.mat'));
                    
                    %measured_data = complex(zeros(9, 14)); 
    
                    for j = 1:9 % Iterate over the 9 sources 
                        sortedFileNames = sort({files.name});
                        path = fullfile(mainFolder, angleDir, 'tar', depthDir, radiusDir, uaDir, waveDir, sortedFileNames{j}); 
                        dref_FFT = Convert_to_Freq_Domain(path); 
                        
                        for k = 1:14 % Iterate over the 14 detectors for each source
                            mes = dref_FFT(d_cord(k,1), d_cord(k, 2)); 
                            measured_data(w, j, k) = mes; 
                        end
                    end 
                end

                % Plot the measured data
                % for temp=1:3
                %     figure;
                %     measured = measured_data(temp,:,:);
                %     plot(abs(measured(:)));
                %     title(sprintf('Measured Data - Wavelength %s, Angle %s', waveDir, angleDir));
                % end 
                filename = sprintf('measured_data_%s_%s_%s_%s.mat', uaDir, radiusDir, depthDir, angleDir);
                save(fullfile('All_Data', filename), 'measured_data', 'uaDir', 'radiusDir', 'depthDir', 'angleDir');
             
            end
        end
    end
end


