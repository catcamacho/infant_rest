%-----------------------------------------------------------------------
% Job saved on 22-Jul-2016 14:30:06 by cfg_util (rev $Rev: 6460 $)
% spm SPM - SPM12 (6685)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.spatial.smooth.data = {'/share/iang/users/ellwoodloweME/spm/999-BABIES-T2/anat/t1w.nii,1'};
matlabbatch{1}.spm.spatial.smooth.fwhm = [4.5 4.5 4.5];
matlabbatch{1}.spm.spatial.smooth.dtype = 0;
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = 's';
matlabbatch{2}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Smooth: Smoothed Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{2}.spm.spatial.preproc.channel.write = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {'/home/monicae3/matlab/spm12/tpm/TPM.nii,1'};
matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {'/home/monicae3/matlab/spm12/tpm/TPM.nii,2'};
matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {'/home/monicae3/matlab/spm12/tpm/TPM.nii,3'};
matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [1 1];
matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {'/home/monicae3/matlab/spm12/tpm/TPM.nii,4'};
matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {'/home/monicae3/matlab/spm12/tpm/TPM.nii,5'};
matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {'/home/monicae3/matlab/spm12/tpm/TPM.nii,6'};
matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{2}.spm.spatial.preproc.warp.samp = 2;
matlabbatch{2}.spm.spatial.preproc.warp.write = [0 1];
