% Motion prelim analysis
RootDir = '/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/motion_params';
[status,cmdout] = system('ls /Users/catcamacho/Box/SNAP/BABIES/processed/preproc/motion_params/*/rest_raw_reoriented_st_mcf.nii.gz.par')
FileListCell = strsplit(cmdout,'\n');

MotionDataCell = cellfun(@(x) dlmread(x),FileListCell(2:end-1),'UniformOutput',false);
MotionDataCell = cellfun(@(x) x(1:140,:),MotionDataCell,'UniformOutput',false);

MotionDataArray = cat(3,MotionDataCell{:});

MotionMeans = zeros([size(MotionDataArray,1),size(MotionDataArray,2)]);
MotionSE = zeros([size(MotionDataArray,1),size(MotionDataArray,2)]);

for MotionIDX = 1:6
    ThisMotionData = squeeze(MotionDataArray(:,MotionIDX,:));
    ThisMotionMean = mean(ThisMotionData,2);
    ThisMotionSE = std(ThisMotionData,[],2)./sqrt(size(ThisMotionData,2));
    
    MotionMeans(:,MotionIDX) = ThisMotionMean;
    MotionSE(:,MotionIDX) = ThisMotionSE;
    
    subplot(3,2,MotionIDX)
    errorbar(ThisMotionMean,ThisMotionSE)
    axis('tight')
end
