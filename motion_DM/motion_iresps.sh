#! /bin/bash

RestRootDir='/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/registered_func'
MotionRootDir='/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/motion_params'
DestRootDir='/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/motion_iresps'
DenoiseDir='/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/denoised_func'

DriftPCsFile='/Users/catcamacho/Box/SNAP/BABIES/processed/preproc/mean_func/mean_func_components.txt'


SubjectDirs=($(ls -d ${RestRootDir}/*BABIES))

for ThisSubjectDirName in "${SubjectDirs[@]}"; do
	
	SubjectLabel=$(basename ${ThisSubjectDirName})
	
	ThisRestFile="${RestRootDir}/${SubjectLabel}/preproc_func.nii.gz"
	ThisLeadLagFile="${MotionRootDir}/${SubjectLabel}/LagMotion_final.1D"
	
	ThisMotionDirName="${DestRootDir}/${SubjectLabel}"
 	ThisMotionIRESPFile="${ThisMotionDirName}/motion_weights_final.nii.gz"
 	ThisResidualFile="${DenoiseDir}/${SubjectLabel}/denoised_func_final.nii.gz"
 
 	if [ ! -d "${ThisMotionDirName}" ]; then
 		mkdir "${ThisMotionDirName}"
 		echo "made: ${ThisMotionDirName}!!!"
 	fi	
	rm "${ThisMotionIRESPFile}" 
	rm "${ThisResidualFile}"
	3dDeconvolve \
		-jobs 4 \
		-input "${ThisRestFile}[0..139]" \
		-ortvec "${ThisLeadLagFile}" 'Motion' \
		-ortvec "${DriftPCsFile}" 'Drift' \
		-nofullf_atall \
		-noFDR \
		-bout \
		-bucket "${ThisMotionIRESPFile}" \
		-mask /Users/catcamacho/Box/SNAP/BABIES/templates/T2w_BABIES_template_2mm_mask.nii.gz \
		-polort 0 \
		-errts "${ThisResidualFile}"

done