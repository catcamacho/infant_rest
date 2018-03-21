
# coding: utf-8

# # Infant resting state fMRI preprocessing
# This notebook contains preprocessing tailored to infant resting state fMRI collected in 5-8 month olds. 
# 
# The processing steps for the fMRI broadly include:
# * Slice-time correction
# * Rigid realignment
# * Co-registration to the sMRI (T2-weighted structural MRI)
# * Artifact detection:
#     - Motion
#     - Global intensity outliers
# * De-noising to remove:
#     - Component noise associated with white matter and CSF
#     - component noise associated with motion
#     - Censoring/scrubbing of individual volumes detected as artifacts in the previous step
#     - Frame-wise displacement
# * Bandpass filtering
# * Spatial smoothing
# * Registration to infant sample template

# In[ ]:


#import packages
from os import listdir, makedirs
from os.path import isdir
from nipype.interfaces.io import DataSink, SelectFiles # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow, MapNode        # pypeline engine
from nipype.interfaces.nipy.preprocess import Trim

from nipype.algorithms.rapidart import ArtifactDetect
import nipype.interfaces.fsl
from nipype.interfaces.fsl.preprocess import SliceTimer, MCFLIRT, FLIRT, FAST, SUSAN
from nipype.interfaces.fsl.utils import Reorient2Std, MotionOutliers
from nipype.interfaces.fsl.model import GLM
from nipype.interfaces.fsl.maths import ApplyMask, TemporalFilter
from nipype.interfaces.freesurfer import Resample, Binarize
from nipype.algorithms.confounds import CompCor
from nipype.interfaces.afni.preprocess import Bandpass
from nipype.interfaces.afni.utils import AFNItoNIFTI
from pandas import DataFrame, Series

#set output file type for FSL to NIFTI
from nipype.interfaces.fsl.preprocess import FSLCommand
FSLCommand.set_default_output_type('NIFTI')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
studyhome = '/Users/catcamacho/Box/SNAP/BABIES'
#studyhome = '/Volumes/iang/active/BABIES/BABIES_rest'
raw_data = studyhome + '/subjDir'
output_dir = studyhome + '/processed/preproc'
workflow_dir = studyhome + '/workflows'
subjects_list = open(studyhome + '/misc/subjects.txt').read().splitlines()
#subjects_list = ['021-BABIES-T1']

template_brain = studyhome + '/templates/T2w_BABIES_template_2mm.nii'
template_wm = studyhome + '/templates/BABIES_wm_mask_2mm.nii'

proc_cores = 2 # number of cores of processing for the workflows

vols_to_trim = 4
interleave = False
TR = 2.5 # in seconds
slice_dir = 3 # 1=x, 2=y, 3=z
resampled_voxel_size = (2,2,2)
fwhm = 4 #fwhm for smoothing with SUSAN

#changed to match Pendl et al 2017 (HBM)
highpass_freq = 0.01 #in Hz
lowpass_freq = 0.08 #in Hz

mask_erosion = 1
mask_dilation = 1


# In[ ]:


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
substitutions = [('_subject_id_', '')]
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir
datasink.inputs.substitutions = substitutions


# In[ ]:


## Nodes for preprocessing

# Reorient to standard space using FSL
reorientfunc = Node(Reorient2Std(), name='reorientfunc')
reorientstruct = Node(Reorient2Std(), name='reorientstruct')

# Reslice- using MRI_convert 
reslice_struct = Node(Resample(voxel_size=resampled_voxel_size), 
                       name='reslice_struct')

# Segment structural scan
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

# Get frame-wise displacement for each run: in_file; out_file, out_metric_plot, out_metric_values
get_FD = Node(MotionOutliers(metric = 'fd',
                             out_metric_values = 'FD.txt',
                             out_metric_plot = 'motionplot.png',
                             no_motion_correction=False),
                 name='get_FD',)

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
                          norm_threshold=0.2, #mutually exclusive with rotation and translation thresh
                          zintensity_threshold=2,
                          use_differences=[True, False]),
           name='art')


# In[ ]:


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
                   (slicetime_correct,get_FD,[('slice_time_corrected_file','in_file')]),
                   (motion_correct,coreg1,[('out_file','in_file')]),
                   (motion_correct,coreg2,[('out_file','in_file')]),
                   (coreg1, coreg2,[('out_matrix_file', 'in_matrix_file')]),
                   (reslice_struct, binarize_struct, [('resampled_file','in_file')]),
                   (binarize_struct,mask_func,[('binary_file','mask_file')]),
                   (coreg2,mask_func,[('out_file','in_file')]),
                   (mask_func,art,[('out_file','realigned_files')]),
                   (binarize_struct,art,[('binary_file','mask_file')]),
                   (motion_correct,art,[('par_file','realignment_parameters')]),
                   #(coreg1,make_coreg_img,[('out_file','epi')]),
                   #(reslice_struct,make_coreg_img,[('resampled_file','anat')]),
                   #(binarize_struct,make_checkmask_img,[('binary_file','brainmask')]),
                   #(coreg1,make_checkmask_img,[('out_file','epi')]),
                   
                   (get_FD, datasink, [('out_metric_values','FD_out_metric_values')]),
                   (motion_correct,datasink,[('par_file','motion_params')]),
                   (reslice_struct,datasink,[('resampled_file','resliced_struct')]),
                   (mask_func,datasink,[('out_file','masked_func')]),
                   (segment,datasink,[('tissue_class_files','tissue_class_files')]),
                   (art,datasink, [('plot_files','art_plot_files')]),
                   (art,datasink, [('outlier_files','vols_to_censor')])
                   #(make_checkmask_img,datasink,[('maskcheck_file','maskcheck_image')]),
                   #(make_coreg_img,datasink,[('coreg_file','coreg_image')])                   
                  ])
preprocwf.base_dir = workflow_dir
preprocwf.write_graph(graph2use='flat')
#preprocwf.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:


# Resting state preprocessing
# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id']),
                     name='infosource')
infosource.iterables = ('subject_id', subjects_list)


# Data grabber- select fMRI and sMRI
templates = {'struct': output_dir + '/resliced_struct/{subject_id}/skullstripped_anat_reoriented_resample.nii',
             'func': output_dir + '/masked_func/{subject_id}/rest_raw_reoriented_trim_st_mcf_flirt_masked.nii',
             'csf': output_dir + '/tissue_class_files/{subject_id}/skullstripped_anat_reoriented_resample_seg_0.nii', 
             'vols_to_censor':output_dir + '/vols_to_censor/{subject_id}/art.rest_raw_reoriented_trim_st_mcf_flirt_masked_outliers.txt', 
             'motion_params':output_dir + '/FD_out_metric_values/{subject_id}/FD.txt',
             'wm':template_wm}
selectfiles = Node(SelectFiles(templates), name='selectfiles')


# In[ ]:


## Pull motion info for all subjects

motion_df = DataFrame(columns=['meanFD','maxFD','NumCensoredVols'])

if isdir(output_dir + '/motion_summary') ==False:
    makedirs(output_dir + '/motion_summary')
    
motion_df_file = output_dir + '/motion_summary/motionSummary.csv'
motion_df.to_csv(motion_df_file)

def summarize_motion(motion_df_file, motion_file, vols_to_censor):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from os.path import dirname, basename
    from numpy import asarray, mean
    from pandas import DataFrame, Series, read_csv
    
    motion_df = read_csv(motion_df_file, index_col=0)
    
    motion = asarray(open(motion_file).read().splitlines()).astype(float)
    censvols = open(vols_to_censor).read().splitlines()

    fp = dirname(motion_file)
    subject = basename(fp)

    motion_df.loc[subject] = [mean(motion),max(motion),len(censvols)]
    motion_df.to_csv(motion_df_file)

    return()

# Make a list of tissues for component noise removal
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
    
# Remove all noise (GLM with noise params)
def create_noise_matrix(vols_to_censor,motion_params,comp_noise):
    from numpy import genfromtxt, zeros,concatenate, savetxt
    from os import path

    motion = genfromtxt(motion_params, delimiter=' ', dtype=None, skip_header=0)
    comp_noise = genfromtxt(comp_noise, delimiter='\t', dtype=None, skip_header=1)
    censor_vol_list = genfromtxt(vols_to_censor, delimiter='\t', dtype=None, skip_header=0)

    c = len(censor_vol_list)
    d = len(comp_noise)
    if c > 0:
        scrubbing = zeros((d,c),dtype=int)
        for t in range(0,c):
            scrubbing[censor_vol_list[t]][t] = 1    
        noise_matrix = concatenate([motion[:,None],comp_noise,scrubbing],axis=1)
    else:
        noise_matrix = concatenate((motion[:,None],comp_noise),axis=1)

    noise_file = 'noise_matrix.txt'
    savetxt(noise_file, noise_matrix, delimiter='\t')
    noise_filepath = path.abspath(noise_file)
    
    return(noise_filepath)

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


# In[ ]:


# Registration
register_template = Node(FLIRT(reference=template_brain), 
                         name='register_template')
xfmCSF = Node(FLIRT(reference=template_brain,apply_xfm=True), 
              name='xfmCSF')
xfmFUNC = Node(FLIRT(reference=template_brain,apply_xfm=True), 
               name='xfmFUNC')

# Denoising
merge_confs = Node(Function(input_names=['mask1','mask2'],
                            output_names=['vols'], 
                            function=combine_masks), 
                   name='merge_confs')

compcor = Node(CompCor(merge_method='none'), 
               name='compcor')

noise_mat = Node(Function(input_names=['vols_to_censor','motion_params','comp_noise'],
                          output_names=['noise_filepath'], 
                          function=create_noise_matrix), 
                 name='noise_mat')

denoise = Node(GLM(out_res_name='denoised_residuals.nii', 
                   out_data_name='denoised_func.nii'), 
               name='denoise')

# band pass filtering- all rates are in Hz (1/TR or samples/second)
bandpass = Node(Bandpass(highpass=highpass_freq,
                         lowpass=lowpass_freq), 
                name='bandpass')

afni_convert = Node(Function(input_names=['in_file'],
                             output_names=['out_file'],
                             function=convertafni), 
                    name='afni_convert')

# Spatial smoothing 
brightthresh_filt = Node(Function(input_names=['func'], 
                                  output_names=['bright_thresh'], 
                                  function=brightthresh), 
                         name='brightthresh_filt')    
    
smooth_filt = Node(SUSAN(fwhm=fwhm), name='smooth_filt')

motion_summary = Node(Function(input_names=['motion_df_file','motion_file','vols_to_censor'], 
                               output_names=[], 
                               function=summarize_motion), 
                      name='motion_summary')
motion_summary.inputs.motion_df_file = motion_df_file


# In[ ]:


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
                   (selectfiles,noise_mat,[('vols_to_censor','vols_to_censor'),
                                           ('motion_params','motion_params')]),
                   (noise_mat,denoise,[('noise_filepath','design')]),
                   (xfmFUNC,denoise,[('out_file','in_file')]),
                   (denoise,bandpass,[('out_data','in_file')]),
                   (bandpass,afni_convert,[('out_file','in_file')]),
                   (afni_convert,brightthresh_filt,[('out_file','func')]),
                   (brightthresh_filt,smooth_filt,[('bright_thresh','brightness_threshold')]),
                   (afni_convert,smooth_filt,[('out_file','in_file')]),  
                   (selectfiles, motion_summary, [('motion_params','motion_file'),
                                                  ('vols_to_censor','vols_to_censor')]),
                   
                   (register_template, datasink,[('out_file','preproc_struct')]),
                   (smooth_filt,datasink,[('smoothed_file','preproc_func')])
                   ])

rs_procwf.base_dir = workflow_dir
rs_procwf.write_graph(graph2use='flat')
rs_procwf.run('MultiProc', plugin_args={'n_procs': proc_cores})


