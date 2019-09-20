function [Result] = MatrixToNIFTI(Data,MaskVolume,StructTemplate,FileName)

OutputStruct = StructTemplate;
OutputVol = zeros([size(MaskVolume),size(Data,1)]);

TempVol = zeros(size(MaskVolume));
for VolIDX = 1:size(Data,1)
    TempVol(MaskVolume) = Data(VolIDX,:);
    OutputVol(:,:,:,VolIDX) = TempVol;
end

OutputStruct.vol = OutputVol;
Result = save_nifti(OutputStruct,FileName);

end