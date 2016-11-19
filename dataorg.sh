#! /bin/csh

set rawfp = /Volumes/iang/active/BABIES/BABIES-T1
set raw_fmri = functional/rest
set ibeat_dir = /Volumes/iang/active/BABIES/BABIES_ibeat/subjDir
set analysis_dir = ~/Desktop
set log = $analysis_dir/log.txt

foreach sub ($argv)

set fldr = ${sub}-BABIES-T1

# if (-e $rawfp/$fldr/$raw_fmri) then
# 	mkdir $analysis_dir/$fldr
# 	mkdir $analysis_dir/$fldr/rest
# 	cp $rawfp/$fldr/$raw_fmri/rest.den.nii $analysis_dir/$fldr/rest/rest.den.nii
# 	mv $analysis_dir/$fldr/rest/rest.den.nii $analysis_dir/$fldr/rest/rest_raw.nii
# 	
# 	cd $analysis_dir/$fldr/rest
# 	fslsplit rest_raw.nii
# 	gunzip vol*	
# endif
# 
# if (-e $analysis_dir/$fldr/rest/vol0000.nii) then
# 	echo '---------------- Rest successfully copied for '$sub >> $log	
# else
# 	echo 'ERROR: rest.den.nii not found for '$sub >> $log
# endif

if (-e $ibeat_dir/${sub}/${sub}-5) then
	mkdir $analysis_dir/$fldr/anat
	
	cd $ibeat_dir/${sub}/${sub}-5
	mri_convert --in_orientation RAS *ravens-gm.img gm.nii
	mri_convert --in_orientation RAS *ravens-wm.img wm.nii
	mri_convert --in_orientation RAS *T2-reoriented-strip.img skullstripped_anat.nii
	mri_convert --in_orientation RAS *seg-aal.img aal_segmentation.nii
	
	cp *.nii $analysis_dir/$fldr/anat/	
endif

if (-e $analysis_dir/$fldr/anat/skullstripped_anat.nii) then
	echo '---------------- iBEAT anats successfully copied for '$sub >> $log
else
	echo 'ERROR: iBEAT directory not found for '$sub >> $log
endif
end
