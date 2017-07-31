
# coding: utf-8

# In[1]:

#import packages
from os import listdir
from nipype.interfaces.io import DataSink, SelectFiles # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow        # pypeline engine
from nipype.interfaces.nipy.preprocess import Trim

from nipype.algorithms.rapidart import ArtifactDetect 
from nipype.interfaces.fsl.preprocess import SliceTimer, MCFLIRT, FLIRT, FAST, SUSAN
from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.interfaces.fsl.model import GLM
from nipype.interfaces.fsl.maths import ApplyMask, TemporalFilter
from nipype.interfaces.freesurfer import Resample, Binarize
from nipype.algorithms.confounds import CompCor
from nipype.interfaces.afni.preprocess import Bandpass
from nipype.interfaces.afni.utils import AFNItoNIFTI

#set output file type for FSL to NIFTI
from nipype.interfaces.fsl.preprocess import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
studyhome = '/Users/catcamacho/Box/BABIES'
#studyhome = '/share/iang/active/BABIES/BABIES_rest'
raw_data = studyhome + '/subjDir'
output_dir = studyhome + '/processed/preproc'
workflow_dir = studyhome + '/workflows'
subjects_list = open(studyhome + '/misc/subjects.txt').read().splitlines()
#subjects_list = ['021-BABIES-T1','033x-BABIES-T1'] #listdir(raw_data)
#subjects_list = ['061-BABIES-T1']

template_brain = studyhome + '/templates/T2w_BABIES_template_2mm.nii'
template_wm = studyhome + '/templates/WM_T2wreg_eroded.nii'

proc_cores = 2 # number of cores of processing for the workflows

vols_to_trim = 4
interleave = False
TR = 2.5 # in seconds
slice_dir = 3 # 1=x, 2=y, 3=z
resampled_voxel_size = (2,2,2)
fwhm = 4 #fwhm for smoothing with SUSAN

highpass_freq = 0.009 #in Hz
lowpass_freq = 0.08 #in Hz

mask_erosion = 1
mask_dilation = 2


# In[2]:

## File handling Nodes

# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subjects_list)


# Data grabber- select fMRI and sMRI
templates = {'struct': raw_data + '/{subject_id}/skullstripped_anat.nii',
            'func': raw_data + '/{subject_id}/rest_raw.nii'}
selectfiles = Node(SelectFiles(templates), name='selectfiles')

# Datasink- where our select outputs will go
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir


# In[3]:

## Nodes for preprocessing

# Reorient to standard space using FSL
reorientfunc = Node(Reorient2Std(), name='reorientfunc')
reorientstruct = Node(Reorient2Std(), name='reorientstruct')

# Reslice- using MRI_convert 
reslice_struct = Node(Resample(voxel_size=resampled_voxel_size), 
                       name='reslice_struct')

# Segment structural scan
#segment = Node(Segment(affine_regularization='none'), name='segment')
segment = Node(FAST(no_bias=True, 
                    segments=True, 
                    number_classes=3), 
               name='segment')

# Trim first 4 volumes using nipype 
trimvols = Node(Trim(begin_index=vols_to_trim), name='trimvols')

#Slice timing correction based on interleaved acquisition using FSL
slicetime_correct = Node(SliceTimer(interleaved=interleave, 
                                    slice_direction=slice_dir,
                                   time_repetition=TR),
                            name='slicetime_correct')

# Motion correction- MEL
motion_correct = Node(MCFLIRT(save_plots=True, 
                              mean_vol=True), 
                      name='motion_correct')

# Registration- using FLIRT
# The BOLD image is 'in_file', the anat is 'reference', the output is 'out_file'
coreg1 = Node(FLIRT(), name='coreg1')
coreg2 = Node(FLIRT(apply_xfm=True), name = 'coreg2')

# make binary mask 
# structural is the 'in_file', output is 'binary_file'
binarize_struct = Node(Binarize(dilate=mask_dilation, 
                                erode=mask_erosion, 
                                min=1), 
                       name='binarize_struct')

# apply the binary mask to the functional data
# functional is 'in_file', binary mask is 'mask_file', output is 'out_file'
mask_func = Node(ApplyMask(), name='mask_func')


# Artifact detection for scrubbing/motion assessment
art = Node(ArtifactDetect(mask_type='file',
                          parameter_source='FSL',
                          norm_threshold=1, #mutually exclusive with rotation and translation thresh
                          zintensity_threshold=2,
                          use_differences=[True, False]),
           name='art')


# In[4]:

# Data QC nodes
def create_coreg_plot(epi,anat):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    coreg_filename='coregistration.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'coregistration to anatomy')
    display.add_edges(anat)
    display.savefig(coreg_filename) 
    display.close()
    coreg_file = os.path.abspath(coreg_filename)
    
    return(coreg_file)

def check_mask_coverage(epi,brainmask):
    import os
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    
    maskcheck_filename='maskcheck.png'
    display = plotting.plot_anat(epi, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'brainmask coverage')
    display.add_contours(brainmask,levels=[.5], colors='r')
    display.savefig(maskcheck_filename)
    display.close()
    maskcheck_file = os.path.abspath(maskcheck_filename)

    return(maskcheck_file)

make_coreg_img = Node(name='make_coreg_img',
                      interface=Function(input_names=['epi','anat'],
                                         output_names=['coreg_file'],
                                         function=create_coreg_plot))

make_checkmask_img = Node(name='make_checkmask_img',
                      interface=Function(input_names=['epi','brainmask'],
                                         output_names=['maskcheck_file'],
                                         function=check_mask_coverage))


# In[ ]:

## Preprocessing Workflow

# workflowname.connect([(node1,node2,[('node1output','node2input')]),
#                    (node2,node3,[('node2output','node3input')])
#                    ])

preprocwf = Workflow(name='preprocwf')
preprocwf.connect([(infosource,selectfiles,[('subject_id','subject_id')]), 
                   (selectfiles,reorientstruct,[('struct','in_file')]),
                   (selectfiles,reorientfunc,[('func','in_file')]),
                   (reorientstruct,reslice_struct,[('out_file','in_file')]),
                   (reslice_struct,coreg1,[('resampled_file','reference')]),
                   (reslice_struct,coreg2,[('resampled_file','reference')]),
                   (reslice_struct,segment,[('resampled_file','in_files')]),
                   (reorientfunc,trimvols,[('out_file','in_file')]),
                   (trimvols,slicetime_correct,[('out_file','in_file')]),
                   (slicetime_correct,motion_correct,[('slice_time_corrected_file','in_file')]),
                   (motion_correct,coreg1,[('out_file','in_file')]),
                   (motion_correct,coreg2,[('out_file','in_file')]),
                   (coreg1, coreg2,[('out_matrix_file', 'in_matrix_file')]),
                   (reslice_struct, binarize_struct, [('resampled_file','in_file')]),
                   (binarize_struct,mask_func,[('binary_file','mask_file')]),
                   (coreg2,mask_func,[('out_file','in_file')]),
                   (mask_func,art,[('out_file','realigned_files')]),
                   (binarize_struct,art,[('binary_file','mask_file')]),
                   (motion_correct,art,[('par_file','realignment_parameters')]),
                   (coreg1,make_coreg_img,[('out_file','epi')]),
                   (reslice_struct,make_coreg_img,[('resampled_file','anat')]),
                   (binarize_struct,make_checkmask_img,[('binary_file','brainmask')]),
                   (coreg1,make_checkmask_img,[('out_file','epi')]),
                   
                   (motion_correct,datasink,[('par_file','motion_params')]),
                   (reslice_struct,datasink,[('resampled_file','resliced_struct')]),
                   (mask_func,datasink,[('out_file','masked_func')]),
                   (segment,datasink,[('tissue_class_files','tissue_class_files')]),
                   (art,datasink, [('plot_files','art_plot_files')]),
                   (art,datasink, [('outlier_files','vols_to_censor')]),
                   (make_checkmask_img,datasink,[('maskcheck_file','maskcheck_image')]),
                   (make_coreg_img,datasink,[('coreg_file','coreg_image')])                   
                  ])
preprocwf.base_dir = workflow_dir
preprocwf.write_graph(graph2use='flat')
preprocwf.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[5]:

# Resting state preprocessing
# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id']),
                     name='infosource')
infosource.iterables = ('subject_id', subjects_list)


# Data grabber- select fMRI and sMRI
templates = {'struct': output_dir + '/resliced_struct/_subject_id_{subject_id}/skullstripped_anat_reoriented_resample.nii',
             'func': output_dir + '/masked_func/_subject_id_{subject_id}/rest_raw_reoriented_trim_st_mcf_flirt_masked.nii',
             'csf': output_dir + '/tissue_class_files/_subject_id_{subject_id}/skullstripped_anat_reoriented_resample_seg_0.nii', 
             'vols_to_censor':output_dir + '/vols_to_censor/_subject_id_{subject_id}/art.rest_raw_reoriented_trim_st_mcf_flirt_masked_outliers.txt', 
             'motion_params':output_dir + '/motion_params/_subject_id_{subject_id}/rest_raw_reoriented_trim_st_mcf.nii.par',
             'wm':template_wm}
selectfiles = Node(SelectFiles(templates), name='selectfiles')


# In[8]:

# Normalization
register_template = Node(FLIRT(reference=template_brain), 
                         name='register_template')
xfmCSF = Node(FLIRT(reference=template_brain,apply_xfm=True), 
              name='xfmCSF')
xfmFUNC = Node(FLIRT(reference=template_brain,apply_xfm=True), 
               name='xfmFUNC')

def combine_masks(mask1,mask2):
    from nipype.interfaces.fsl.utils import Merge
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    vols = []
    vols.append(mask1)
    vols.append(mask2)
    
    return(vols)
    
merge_confs = Node(name='merge_confs', interface=Function(input_names=['mask1','mask2'], 
                                                          output_names=['vols'], 
                                                          function=combine_masks))

compcor = Node(CompCor(merge_method='none'), 
               name='compcor')


# Remove all noise (GLM with noise params)
def create_noise_matrix(vols_to_censor,motion_params,comp_noise):
    from numpy import genfromtxt, zeros,concatenate, savetxt
    from os import path
    
    motion = genfromtxt(motion_params, delimiter='  ', dtype=None, skip_header=0)
    comp_noise = genfromtxt(comp_noise, delimiter='\t', dtype=None, skip_header=1)
    censor_vol_list = genfromtxt(vols_to_censor, delimiter='\t', dtype=None, skip_header=0)
    
    c = len(censor_vol_list)
    d = len(comp_noise)
    if c > 0:
        scrubbing = zeros((d,c),dtype=int)
        for t in range(c):
            scrubbing[censor_vol_list[t]][t] = 1
        noise_matrix = concatenate((motion,comp_noise,scrubbing),axis=1)
    else:
        noise_matrix = concatenate((motion,comp_noise),axis=1)
    
    noise_file = 'noise_matrix.txt'
    savetxt(noise_file, noise_matrix, delimiter='\t')
    noise_filepath = path.abspath(noise_file)
    
    return(noise_filepath)

noise_mat = Node(name='noise_mat', interface=Function(input_names=['vols_to_censor','motion_params','comp_noise'],
                                                      output_names=['noise_filepath'], 
                                                      function=create_noise_matrix))

denoise = Node(GLM(out_res_name='denoised_residuals.nii', 
                   out_data_name='denoised_func.nii'), 
               name='denoise')

# AR filter- We'll need to play with this a bit for newborns- not super necessary right now. 

# band pass filtering- all rates are in Hz (1/TR or samples/second)
def bandpass_filter(in_file, lowpass, highpass, sampling_rate):
    import numpy as np
    import nibabel as nb
    from os import path
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    out_file = 'func_filtered.nii'
    
    img = nb.load(in_file)
    timepoints = img.shape[-1]
    F = np.zeros(timepoints)
    lowidx = np.round(lowpass / sampling_rate * timepoints)
    lowidx = lowidx.astype(int)
    highidx = np.round(highpass / sampling_rate * timepoints)
    highidx = highidx.astype(int)
    F[highidx:lowidx] = 1
    F = ((F + F[::-1]) > 0).astype(int)
    data = img.get_data()
    data[data==0] = np.nan
    filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
    filtered_data[np.isnan(filtered_data)] = 0
    img_out = nb.Nifti1Image(filtered_data, img.get_affine(),
                             img.get_header())
    nb.save(img_out,out_file)
    out_file = path.abspath(out_file)
    return(out_file)

bandpass = Node(name='bandpass', 
                interface=Function(input_names=['in_file','lowpass','highpass','sampling_rate'], 
                                   output_names=['out_file'],
                                   function=bandpass_filter))
bandpass.inputs.lowpass = lowpass_freq
bandpass.inputs.highpass = highpass_freq
bandpass.inputs.sampling_rate = 1/TR

bandpass2 = Node(Bandpass(highpass=highpass_freq,
                          lowpass=lowpass_freq), 
                 name='bandpass')
# Convert afni to nifti format
afni_convert = Node(AFNItoNIFTI(out_file='func_filtered'), 
                    name='afni_convert')

def convertafni(in_file):
    from nipype.interfaces.afni.utils import AFNItoNIFTI
    from os import path
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    cvt = AFNItoNIFTI()
    cvt.inputs.in_file = in_file
    cvt.inputs.out_file = 'func_filtered.nii'
    cvt.run()
    
    out_file = path.abspath('func_filtered.nii')
    return(out_file)

afni_convert2 = Node(name='afni_convert2', 
                     interface=Function(input_names=['in_file'], 
                                        output_names=['out_file'],
                                        function=convertafni))
# Spatial smoothing using FSL
# Brightness threshold should be 0.75 * the contrast between the median brain intensity and the background
def brightthresh(func):
    import nibabel as nib
    from numpy import median, where
    
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    
    func_nifti1 = nib.load(func)
    func_data = func_nifti1.get_data()
    func_data = func_data.astype(float)
    
    brain_values = where(func_data > 0)
    median_thresh = median(brain_values)
    bright_thresh = 0.75 * median_thresh
    
    return(bright_thresh)

brightthresh_filt = Node(name='brightthresh_filt',
                         interface=Function(input_names=['func'], 
                                            output_names=['bright_thresh'], 
                                            function=brightthresh))    
    
smooth_filt = Node(SUSAN(fwhm=fwhm), name='smooth_filt')

brightthresh_orig = Node(name='brightthresh_orig',
                         interface=Function(input_names=['func'], 
                                            output_names=['bright_thresh'], 
                                            function=brightthresh))    
    
smooth_orig = Node(SUSAN(fwhm=fwhm), name='smooth_orig')


# In[10]:

# workflowname.connect([(node1,node2,[('node1output','node2input')]),
#                       (node2,node3,[('node2output','node3input')])
#                     ])

rs_procwf = Workflow(name='rs_procwf')
rs_procwf.connect([(infosource,selectfiles,[('subject_id','subject_id')]),
                   (selectfiles,register_template,[('struct','in_file')]),
                   (selectfiles,xfmFUNC,[('func','in_file')]),
                   (selectfiles,xfmCSF,[('csf','in_file')]),
                   (register_template, xfmFUNC,[('out_matrix_file','in_matrix_file')]),
                   (register_template, xfmCSF,[('out_matrix_file','in_matrix_file')]),
                   (xfmCSF,merge_confs,[('out_file','mask1')]),
                   (selectfiles,merge_confs,[('wm','mask2')]),
                   (merge_confs,compcor,[('vols','mask_files')]),
                   (xfmFUNC,compcor,[('out_file','realigned_file')]),
                   (compcor,noise_mat,[('components_file','comp_noise')]),
                   (selectfiles,noise_mat,[('vols_to_censor','vols_to_censor')]),
                   (selectfiles,noise_mat,[('motion_params','motion_params')]),
                   (noise_mat,denoise,[('noise_filepath','design')]),
                   (xfmFUNC,denoise,[('out_file','in_file')]),
                   (denoise,bandpass2,[('out_data','in_file')]),
                   (bandpass2,afni_convert2,[('out_file','in_file')]),
                   (afni_convert2,brightthresh_filt,[('out_file','func')]),
                   (brightthresh_filt,smooth_filt,[('bright_thresh','brightness_threshold')]),
                   (afni_convert2,smooth_filt,[('out_file','in_file')]), 
                   (denoise,brightthresh_orig,[('out_file','func')]),
                   (brightthresh_orig,smooth_orig,[('bright_thresh','brightness_threshold')]),
                   (denoise,smooth_orig,[('out_data','in_file')]),  
                   
                   (compcor,datasink,[('components_file','components_file')]),
                   (smooth_filt,datasink,[('smoothed_file','smoothed_filt_func')]),
                   (smooth_orig,datasink,[('smoothed_file','smoothed_orig_func')]),
                   (afni_convert2,datasink,[('out_file','bp_filtered_func')]),
                   #(denoise,datasink,[('out_res','denoise_resids')]),
                   (denoise,datasink,[('out_data','denoised_func')])
                   ])

rs_procwf.base_dir = workflow_dir
rs_procwf.write_graph(graph2use='flat')
rs_procwf.run('MultiProc', plugin_args={'n_procs': proc_cores})

