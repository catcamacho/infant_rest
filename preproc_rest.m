function  preproc_rest
fp = '/share/iang/users/ellwoodloweME/spm/';
restfolder = '/rest/';
restVols =  cellstr(spm_select('FPList',[fp subjects{1,i} restfolder],'vol*'));

end

