% Modified based on bbreast_phantom_DOT_2.m to loop

close all; clear all;
% Define basic parameters
% voxel size (need to confirm with Iqbal, need to consider the
% downsampling)
unitinmm = 5/8;
% time parameter
tend = 7.1428e-09; %2e-9;
tg = 200;


wavelength = [757,800,850];
% epithelium index for each wavelength 
ep = [5,4,3];
% depth of target [voxel]
depth = (.5:.5:3) ;
% radius of the target [voxel]
radius = (.75:.15:1.2);
% optical property of the target [cm-1]
ua = [0.12:0.015:0.21]/10;
% % my office computer
% ParentFolder = 'C:\Users\Admin\OneDrive - Washington University in St. Louis\simulation data\Iqbal DOT project\Mar14_2024';
% % Iqbal's computer
ParentFolder = pwd;
% Home computer
% ParentFolder = 'C:\Users\wlk9\OneDrive - Washington University in St. Louis\simulation data\Iqbal DOT project\Mar7_2024';
% Taylor's computer
% ParentFolder = 'C:\Users\lukai\OneDrive - Washington University in St. Louis\simulation data\Iqbal DOT project\Mar7_2024';
for iw = 1:size(wavelength,2) 
    for angle = 0:0.5
        % Simulation with reference
        % Load phantom data
        RefFolder = fullfile(ParentFolder,num2str(wavelength(1,iw)),num2str(angle),'ref');
        filename = fullfile(RefFolder,['phan',num2str(wavelength(1,iw)),'_rot_compressed_cut_',num2str(angle),'.mat']);
        load(filename);
        % Load optical properties
        filename = fullfile(RefFolder,['optical_prop_',num2str(wavelength(1,iw)),'_',num2str(angle),'.mat']);
        load(filename);
        % position of sources and detectors (need to be changed later)
        % src_pos = round([size(phan_rot_compressed_cut,1)/2 size(phan_rot_compressed_cut,2)/2 1]);
        % src_pos(1,1:2) = src_pos(1,1:2) + 0.5;
        % det_pos = round([5+size(phan_rot_compressed_cut,1)/2 5+size(phan_rot_compressed_cut,2)/2 1]);
        load('src_pos.mat'); load('det_pos.mat');
        % number of source and detector
        Ns = size(src_pos,1);
        Nd = size(det_pos,1);

        for ss=1:Ns
            [dref]=MC_Measure3(unitinmm,src_pos(ss,:),optical_prop,tend,tg,phan_rot_compressed_cut);
            % Rd_tar{ss} = dref;
            filename = fullfile(RefFolder,['dref0_',num2str(ss),'.mat']);
            save(filename,'dref');
        end
        

        % Simulation with target
        for id = 1:size(depth,2)
            for ir = 1:size(radius,2)
                for iua = 1:size(ua,2)
                    TarFolder = fullfile(ParentFolder,num2str(wavelength(1,iw)),num2str(angle),'tar',['dep_',num2str(depth(id))],['rad_',num2str(radius(ir))],['ua_',num2str(ua(iua))]);
                    filename = fullfile(TarFolder,['phan',num2str(wavelength(1,iw)),'_rot_compressed_cut_',num2str(angle),'.mat']);
                    load(filename);
                    filename = fullfile(TarFolder,['optical_prop_',num2str(wavelength(1,iw)),'_',num2str(angle),'.mat']);
                    load(filename);
                    for ss=1:Ns
                        [dref]=MC_Measure3(unitinmm,src_pos(ss,:),optical_prop,tend,tg,phan_rot_compressed_cut);
                        filename = fullfile(TarFolder,['dref_',num2str(ss),'.mat']);
                        save(filename,'dref');

                    end
                    % save();
                end
            end
        end

    end
end


