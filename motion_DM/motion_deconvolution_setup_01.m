RootDir = '/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/motion_params';
RootDirContents = dir(RootDir);

RootDirContents = RootDirContents(cellfun(@(x) ~isempty(regexpi(x,'BABIES')),{RootDirContents.name}));

for DirIDX = 1:numel(RootDirContents)
    
    ThisDirName = fullfile(RootDir,RootDirContents(DirIDX).name);
    ThisMotionFileName = fullfile(ThisDirName,'rest_raw_reoriented_st_mcf.nii.gz.par');
    ThisLeadLagFileName = fullfile(ThisDirName,'LagMotion_final.1D');
    
    
    ThisMotionData = dlmread(ThisMotionFileName,'\t');
    ThisMotionData = ThisMotionData(1:140,:);
    ThisMotionDiff = [zeros([1,6]);diff(ThisMotionData,1,1)];
    ThisMotionDiff =  bsxfun(@minus,ThisMotionDiff,mean(ThisMotionDiff,1));
    
    %ThisLeadLagMatrix = GetLeadLagMatrix([ThisMotionDiff,ThisMotionDiff.^2],0,2);
    ThisLeadLagMatrix = GetLeadLagMatrix(ThisMotionDiff,0,2);
    
    dlmwrite(ThisLeadLagFileName,ThisLeadLagMatrix,'\t')
end