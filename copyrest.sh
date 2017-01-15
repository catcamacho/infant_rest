#! /bin/tcsh

# NOTE:
# if using rest.nii mark YELLOW
# if using rest.den.nii mark GREEN

foreach ID (002)

	set fldr = /Volumes/group/iang/active/BABIES/BABIES-T1/${ID}-BABIES-T1
	set analysis_dir = /Volumes/group/iang/active/BABIES/BABIES_rest/subjDir

	cd $fldr

		echo ' + making rest analysis dir for '$ID
		mkdir $analysis_dir/${ID}-BABIES-T1
		mkdir $analysis_dir/${ID}-BABIES-T1/rest
		cp $fldr/functional/rest/rest.den.nii $analysis_dir/${ID}-BABIES-T1/rest/rest_raw.nii
		cd $analysis_dir/${ID}-BABIES-T1/rest
		fslsplit rest_raw.nii
		gunzip vol*
		
		echo ' + checking rest motion for '$ID
		fsl_motion_outliers -i rest_raw.nii -o dvars_outliers -s dvars.txt -p dvars.png --dvars
		fsl_motion_outliers -i rest_raw.nii -o fd_outliers -s fd.txt -p fd.png --fd
		
		
end

end