#! /bin/tcsh

# NOTE:
# if using rest.nii mark YELLOW
# if using rest.den.nii mark GREEN

foreach ID (056 061 062 064x 065 067 072 076x 077x 087 099 100 102 106)

	set fldr = /share/iang/active/BABIES/BABIES_Crossectional/BABIES_Crossectional-T1/${ID}-C-T1
	set analysis_dir = /share/iang/active/BABIES/BABIES_rest/subjDir

	cd $fldr

		echo ' + making rest analysis dir for '$ID
		mkdir $analysis_dir/${ID}-BABIES-T1
		cp $fldr/functional/rest/recon/*.den.nii* $analysis_dir/${ID}-BABIES-T1/
		gunzip $analysis_dir/${ID}-BABIES-T1/*.den.nii.gz
		mv $analysis_dir/${ID}-BABIES-T1/*.den.nii $analysis_dir/${ID}-BABIES-T1/rest_raw.nii
		
end

end