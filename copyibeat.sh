#! /bin/tcsh

set ibeat_dir = /share/iang/active/BABIES/BABIES_ibeat/subjDir/BABIES_C
set analysis_dir = /share/iang/active/BABIES/BABIES_rest/subjDir
set log = $analysis_dir/ibeatlog.txt

foreach sub ($argv)

set fldr = ${sub}-BABIES-T1

if (-e $ibeat_dir/T1${sub}/T1${sub}-5) then
	#mkdir $analysis_dir/$fldr
	
	cd $ibeat_dir/T1${sub}/T1${sub}-5
	#mri_convert --in_orientation RAS *ravens-gm.img gm.nii
	#mri_convert --in_orientation RAS *ravens-wm.img wm.nii
	mri_convert --in_orientation RAS *T2-reoriented-strip.img skullstripped_anat.nii
	#mri_convert --in_orientation RAS *seg-aal.img aal_segmentation.nii
	
	cp skullstripped_anat.nii $analysis_dir/$fldr/	
endif

if (-e $analysis_dir/$fldr/skullstripped_anat.nii) then
	echo '---------------- iBEAT anats successfully copied for '$sub >> $log
else
	echo 'ERROR: iBEAT directory not found for '$sub >> $log
endif
end
