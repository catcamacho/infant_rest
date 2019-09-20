function [TimeSeriesArray] = GetMaskTimeSeries(Vols,Mask)

VolCount = size(Vols,4);
VoxelCount = nnz(Mask(:));

TimeSeriesArray = zeros(VolCount,VoxelCount);

for VolIDX = 1:VolCount
    
    ThisVol = squeeze(Vols(:,:,:,VolIDX));
    TimeSeriesArray(VolIDX,:) = ThisVol(Mask);

end