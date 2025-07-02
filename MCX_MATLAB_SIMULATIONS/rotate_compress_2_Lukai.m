% Modified based on rotate_compress_Lukai
% Generate the reference


wavelength = [757,800,850];
% epithelium index for each wavelength
ep = [5,4,3];
for iw = 1:size(wavelength,2)
    for angle = 0:0.5
        % create folder
        DataFolder = fullfile(pwd,num2str(wavelength(1,iw)),num2str(angle),'ref');
        if ~exist(DataFolder)
            mkdir(DataFolder)
        end
        % all name as medium_indices and optical_prop
        medium_indices = load(['medium_indices_3d_',num2str(wavelength(1,iw)),'.mat']).(['medium_indices_3d_',num2str(wavelength(1,iw))]);
        optical_prop = load(['optical_prop_',num2str(wavelength(1,iw)),'.mat']).(['optical_prop_',num2str(wavelength(1,iw))]);

        %figure; imagesc(squeeze(medium_indices(:,136,:))); %colorbar; clim([0 10]);

        % dropping the empty space above the breast
        phan = medium_indices(:, :, 39:end);
        %figure; imagesc(squeeze(phan(:,193,:))); %colorbar; clim([0 10]);


        % rotate and compress
        phan_rot = imrotate3(phan,angle,[1,0,0],'nearest','crop','FillValues',1);
        %figure; imagesc(squeeze(phan_rot(:,136,:))); %colorbar; clim([0 10]);

        % 1360*1360*680 each voxel volume (0.125 mm)^3 & center is (-85, -85, -85)
        % 272*272*136 each voxel volume (0.125*5)^3 == 0.6250^3

        % reduce the height of the breast by half. then the x and y has the values
        % equal to dx', dy' = 0.625^3/0.625 where z'= z/2 ;
        dx = 0.625; dy = 0.625; dz = 0.625;
        ratio_z = 1/2;
        [nx, ny, nz ] = size (phan_rot);
        dx_new = sqrt(0.625^2*2);
        dy_new = sqrt(0.625^2*2);
        x_new = nx*dx_new;
        y_new = ny*dy_new;
        x_old = nx*dx;
        y_old = ny*dy;

        ratiox = x_new/x_old; ratioy =  y_new/y_old;


        phan_rot_compressed =phan_rot; % imresize3(phan_rot,[round(nx*ratiox), round(ny*ratioy),round(nz*ratio_z)]);
        phan_rot_compressed = round(phan_rot_compressed);
        phan_rot_compressed(:,:,1:5) = 1;
        phan_rot_compressed = phan_rot_compressed - 1;
        phan_rot_compressed(phan_rot_compressed == -1) = 0;
        phan_rot_compressed(phan_rot_compressed > (size(optical_prop,1)-1)) = (size(optical_prop,1)-1);
        figure; imagesc(squeeze(phan_rot_compressed(:,136,:))); %colorbar; clim([0 10]);
        %pause
        indx1 = find(phan_rot_compressed(68,136,:)~=0);
        indx2 = find(phan_rot_compressed(204,136,:)~= 0);

        cut_slice = max(indx1(1), indx2(1));

        phan_rot_compressed_cut = phan_rot_compressed(:,:,cut_slice-1:end);
        phan_rot_compressed_cut(:,:,1)=0;
        % add epthelium
        % Fill all zero elements in phan_rot_compressed_cut(:,:,2) with 6
        for i = 1:size(phan_rot_compressed_cut,1)
            for j = 1:size(phan_rot_compressed_cut,2)
                if (phan_rot_compressed_cut(i,j,2) ~= 0)
                    phan_rot_compressed_cut(i,j,2) = ep(1,iw);
                end
            end
        end

        figure; imagesc(squeeze(phan_rot_compressed_cut(:,136,:)));
        %pause; 
        filename = fullfile(DataFolder,['phan',num2str(wavelength(1,iw)),'_rot_compressed_cut_',num2str(angle),'.mat']);
        save(filename,'phan_rot_compressed_cut');
        filename = fullfile(DataFolder,['optical_prop_',num2str(wavelength(1,iw)),'_',num2str(angle),'.mat']);
        save(filename,'optical_prop');
        close all;
    end
    % wavelength = [757,800,850]; 
    % ep = [5,4,3];
end

