% mar 8, 2024: modified based on MC_Measure2 to return only the data from
% detector surface
function [ref]=MC_Measure3(unit,src_pos,tissue_prop,tend,tg,volume)
cfg.nphoton=1e8;
cfg.unitinmm=unit; % define the pixel size in terms of mm
cfg.vol = volume;
cfg.vol=uint8(cfg.vol);
cfg.srcpos=src_pos;


cfg.prop=tissue_prop;
cfg.issrcfrom0=0;
cfg.srcdir=[0,0,1];
cfg.maxdetphoton=5000000;
cfg.isreflect=0; % enable reflection at exterior boundary
cfg.isrefint=1;  % enable reflection at interior boundary
cfg.issaveref = 1; % save reflectance on the surface
cfg.gpuid=1;
cfg.autopilot=1;
cfg.tstart=0;
cfg.tend=tend;
cfg.tstep=tend/tg;
[flux]=mcxlab(cfg); 
ref=squeeze(flux.dref(:,:,1,:));
% figure; imagesc(squeeze(ref(:,:,10)));
% figure; imagesc(squeeze(ref(:,:,30)));
% figure; imagesc(squeeze(ref(:,:,150)));
end