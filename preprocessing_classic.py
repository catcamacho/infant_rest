
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
from nipype.interfaces.io import DataSink, SelectFiles, DataGrabber # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow, MapNode        # pypeline engine
from nipype.interfaces.nipy.preprocess import Trim

from nipype.algorithms.rapidart import ArtifactDetect
from nipype.interfaces.fsl.preprocess import SliceTimer, MCFLIRT, FLIRT, FAST, SUSAN
from nipype.interfaces.fsl.utils import Reorient2Std, MotionOutliers, Merge
from nipype.interfaces.fsl.model import GLM
from nipype.interfaces.fsl.maths import ApplyMask, TemporalFilter
from nipype.interfaces.freesurfer import Resample, Binarize, MRIConvert
from nipype.algorithms.confounds import CompCor
from nipype.interfaces.afni.preprocess import Bandpass
from nipype.interfaces.afni.utils import AFNItoNIFTI
from nipype.algorithms.misc import Gunzip
from pandas import DataFrame, Series

#set output file type for FSL to NIFTI
from nipype.interfaces.fsl.preprocess import FSLCommand
FSLCommand.set_default_output_type('NIFTI_GZ')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
setup='Lucy'

if setup == 'Lucy':
    studyhome = '/Volumes/group/iang/active/BABIES/BABIES_rest/'
    raw_data = studyhome + '/subjDir/'
    output_dir = studyhome + '/processed/preproc'
    workflow_dir = studyhome + '/workflows'
    template_brain = studyhome + '/BABIES_rest/templates/T2w_BABIES_template_2mm.nii.gz'
    template_wm = studyhome + '/BABIES_rest/templates/BABIES_wm_mask_2mm.nii.gz'
    template_mask = studyhome + '/BABIES_rest/templates/T2w_BABIES_template_2mm_mask.nii.gz'
    
elif setup =='Cat':
    studyhome = '/Users/catcamacho/Box/SNAP/BABIES'
    raw_data = studyhome + '/raw'
    output_dir = studyhome + '/BABIES_rest/processed/preproc'
    workflow_dir = studyhome + '/BABIES_rest/workflows'
    template_brain = studyhome + '/templates/T2w_BABIES_template_2mm.nii.gz'
    template_wm = studyhome + '/templates/BABIES_wm_mask_2mm.nii.gz'
    template_mask = studyhome + '/templates/T2w_BABIES_template_2mm_mask.nii.gz'

#subjects_list = open(studyhome + '/BABIES_rest/misc/subjects_test.txt').read().splitlines()
subjects_list = ['sham1','sham2']

proc_cores = 2 # number of cores of processing for the workflows
vols_to_trim = 4
interleave = False
TR = 2.5 # in seconds
slice_dir = 3 # 1=x, 2=y, 3=z
resampled_voxel_size = (2,2,2)
fwhm = 4 #fwhm for smoothing with SUSAN

#changed to match Pendl et al 2017 (HBM)
highpass_freq = 0.08 #in Hz
lowpass_freq = 0.1 #in Hz

mask_erosion = 1
mask_dilation = 1


# In[ ]:

## File handling Nodes

# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id']),
                     name='infosource')
infosource.iterables = ('subject_id', subjects_list)

# Datasink- where our select outputs will go
substitutions = [('_subject_id_', '')]
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir
datasink.inputs.substitutions = substitutions

# Data grabber- select fMRI and sMRI
if setup=='Lucy':
    anat_template = {'struct': skullstrip_struct_data + '/{subject_id}/skullstripped_anat.nii.gz'}
    select_anat = Node(SelectFiles(anat_template), name='selectanat')

    func_template = {'func': raw_func_data + '/%s/rest_raw*.nii.gz'}
    select_func = Node(DataGrabber(sort_filelist=True,
                                   template = raw_data + '/%s/rest_raw*.nii.gz',
                                   field_template = func_template,
                                   base_directory=raw_data,
                                   infields=['subject_id'], 
                                   template_args={'func':[['subject_id']]}), 
                       name='select_func')    
    
elif setup=='Cat':
    anat_template = {'struct': raw_data + '/{subject_id}/skullstripped_anat.nii.gz'}
    select_anat = Node(SelectFiles(anat_template), name='selectanat')

    func_template = {'func': raw_data + '/%s/rest_raw*.nii.gz'}
    select_func = Node(DataGrabber(sort_filelist=True,
                                   template = raw_data + '/%s/rest_raw*.nii.gz',
                                   field_template = func_template,
                                   base_directory=raw_data,
                                   infields=['subject_id'], 
                                   template_args={'func':[['subject_id']]}), 
                       name='select_func')



# In[ ]:

## struct processing
# reorient to standard
reorientstruct = Node(Reorient2Std(), name='reorientstruct')

# convert files to nifti
reslice_struct = Node(MRIConvert(out_type='niigz',
                                 conform_size=2,
                                 crop_size=(128, 128, 128),
                                ),
                   name='reslice_struct')
# Segment structural scan
segment = Node(FAST(no_bias=True, 
                    segments=True, 
                    number_classes=3), 
               name='segment')

# register BOLD to anat
coregT2 = Node(FLIRT(out_matrix_file='xform.mat'),
           name='coregT1')

# apply transform to func
applyT2xform = Node(FLIRT(apply_xfm=True),
                    name='applyT2xform')

# register anat to template
reg_temp = Node(FLIRT(reference=template_brain, 
                      out_matrix_file='xform.mat',
                      out_file='preproc_anat.nii.gz'),
                name='reg_temp')

# apply transform to func
applyxform = Node(FLIRT(reference=template_brain,
                        apply_xfm=True, 
                        out_file='preproc_func.nii.gz'),
                  name='applyxform')


# In[ ]:

def combine_fd(fd_list):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from os.path import abspath
    from numpy import asarray, savetxt
    
    motion = open(fd_list[0]).read().splitlines()

    if len(fd_list)>1:
        for file in fd_list[1:]:
            temp = open(file).read().splitlines()
            motion = motion+temp

    motion = asarray(motion).astype(float)
    filename = 'FD_full.txt'
    savetxt(filename,motion)
    combined_fd = abspath(filename)
    return(combined_fd)


def combine_par(par_list):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from os.path import abspath
    from numpy import vstack, savetxt, genfromtxt
    
    motion = genfromtxt(par_list[0], dtype=float)
    if len(par_list)>1:
        for file in par_list[1:]:
            temp = genfromtxt(par_list[0], dtype=float)
            motion=vstack((motion,temp))
    
    filename = 'motion.par'
    savetxt(filename, motion, delimiter=' ')
    combined_par = abspath(filename)
    return(combined_par)


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
    from os.path import abspath
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nilearn import plotting
    from nipype.interfaces.nipy.preprocess import Trim
    
    trim = Trim()
    trim.inputs.in_file = epi
    trim.inputs.end_index = 1
    trim.inputs.out_file = 'epi_vol1.nii.gz'
    trim.run()
    epi_vol = abspath('epi_vol1.nii.gz')
    
    maskcheck_filename='maskcheck.png'
    display = plotting.plot_anat(epi_vol, display_mode='ortho',
                                 draw_cross=False,
                                 title = 'brainmask coverage')
    display.add_contours(brainmask,levels=[.5], colors='r')
    display.savefig(maskcheck_filename)
    display.close()
    maskcheck_file = abspath(maskcheck_filename)

    return(maskcheck_file)

make_coreg_img = Node(Function(input_names=['epi','anat'],
                               output_names=['coreg_file'],
                               function=create_coreg_plot), name='make_coreg_img')

make_checkmask_img = Node(
    Function(input_names=['epi','brainmask'],
             output_names=['maskcheck_file'],
             function=check_mask_coverage), name='make_checkmask_img')
make_checkmask_img.inputs.brainmask = template_mask


# In[ ]:

## Nodes for func preprocessing

# Reorient to standard space using FSL
reorientfunc = MapNode(Reorient2Std(), name='reorientfunc', iterfield=['in_file'])

# Trim first 4 volumes
trimvols = MapNode(Trim(begin_index=vols_to_trim), name='trimvols', iterfield=['in_file'])

# Motion correction
motion_correct = MapNode(MCFLIRT(save_plots=True, mean_vol=True), name='motion_correct', iterfield=['in_file'])

# Get frame-wise displacement for each run: in_file; out_file, out_metric_plot, out_metric_values
get_FD = MapNode(MotionOutliers(metric = 'fd',
                                out_metric_values = 'FD.txt',
                                out_metric_plot = 'motionplot.png',
                                no_motion_correction=False,
                                threshold=0.01),
                    name='get_FD', iterfield=['in_file'])

# Merge rest
merge = Node(Merge(dimension='t'), name='merge')

# Merge motion (6 params)
comb_par = Node(Function(input_names=['par_list'],
                         output_names=['combined_par'],
                         function=combine_par), name='comb_par')

# Merge FD
comb_fd = Node(Function(input_names=['fd_list'],
                         output_names=['combined_fd'],
                         function=combine_fd), name='comb_fd')

# Rigid body realignment
realignment = Node(MCFLIRT(), name='realignment')

#Slice timing correction based on interleaved acquisition using FSL
slicetime_correct = Node(SliceTimer(interleaved=interleave, 
                                    slice_direction=slice_dir,
                                    time_repetition=TR, 
                                    out_file='st_func.nii.gz'),
                         name='slicetime_correct')

# unzip the nifti for ART
gunzip = Node(Gunzip(), name='gunzip')

# Artifact detection for scrubbing/motion assessment
art = Node(ArtifactDetect(mask_type='file',
                          parameter_source='FSL',
                          norm_threshold=0.25, #mutually exclusive with rotation and translation thresh
                          zintensity_threshold=2,
                          use_differences=[True, False], 
                          mask_file=template_mask),
           name='art')


# In[ ]:

## Preprocessing Workflow

# workflowname.connect([(node1,node2,[('node1output','node2input')]),
#                    (node2,node3,[('node2output','node3input')])
#                    ])

preprocwf = Workflow(name='preprocwf')
preprocwf.connect([(infosource,select_func,[('subject_id','subject_id')]), 
                   (select_func,reorientfunc,[('func','in_file')]),
                   (reorientfunc,trimvols,[('out_file','in_file')]),
                   (trimvols,motion_correct,[('out_file','in_file')]),
                   (trimvols,get_FD,[('out_file','in_file')]),
                   (motion_correct,merge,[('out_file','in_files')]),
                   (merge,realignment,[('merged_file','in_file')]),
                   (realignment,slicetime_correct,[('out_file','in_file')]),
                   (get_FD,comb_fd,[('out_metric_values','fd_list')]),
                   (motion_correct,comb_par,[('par_file','par_list')]),
                   (comb_par,art,[('combined_par','realignment_parameters')]),
                   (applyxform,gunzip,[('out_file','in_file')]),
                   (gunzip,art,[('out_file','realigned_files')]),
                   
                   (infosource,select_anat,[('subject_id','subject_id')]),
                   (select_anat,reorientstruct,[('struct','in_file')]),
                   (reorientstruct,reslice_struct,[('out_file','in_file')]),
                   (reslice_struct,coregT2,[('out_file','reference')]),
                   (reslice_struct,applyT2xform,[('out_file','reference')]),
                   (reslice_struct,reg_temp,[('out_file','in_file')]),
                   (reg_temp,applyxform,[('out_matrix_file','in_matrix_file')]),
                   (slicetime_correct,coregT2,[('slice_time_corrected_file','in_file')]),
                   (slicetime_correct,applyT2xform,[('slice_time_corrected_file','in_file')]),
                   (coregT2,applyT2xform,[('out_matrix_file','in_matrix_file')]),
                   (applyT2xform, applyxform,[('out_file','in_file')]),
                   (reg_temp,segment,[('out_file','in_files')]),
                   
                   (coregT2,make_coreg_img,[('out_file','epi')]),
                   (applyxform,make_checkmask_img,[('out_file','epi')]),
                   (reslice_struct,make_coreg_img,[('out_file','anat')]),
                   
                   (applyxform, datasink, [('out_file','preproc_func')]),
                   (reg_temp, datasink,[('out_file','preproc_anat')]),
                   (comb_fd, datasink, [('combined_fd','FD_out_metric_values')]),
                   (comb_par,datasink,[('combined_par','motion_params')]),
                   (segment,datasink,[('tissue_class_files','tissue_class_files')]),
                   (art,datasink, [('plot_files','art_plot_files')]),
                   (art,datasink, [('outlier_files','vols_to_censor')]),
                   (make_checkmask_img,datasink,[('maskcheck_file','maskcheck_image')]),
                   (make_coreg_img,datasink,[('coreg_file','coreg_image')])                   
                  ])
preprocwf.base_dir = workflow_dir
preprocwf.write_graph(graph2use='flat')
preprocwf.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:

# Resting state preprocessing
# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id']),
                     name='infosource')
infosource.iterables = ('subject_id', subjects_list)


# Data grabber- select fMRI and sMRI
templates = {'func': output_dir + '/preproc_func/{subject_id}/preproc_func.nii.gz',
             'csf': output_dir + '/tissue_class_files/{subject_id}/preproc_anat_seg_0.nii.gz', 
             'vols_to_censor':output_dir + '/vols_to_censor/{subject_id}/art.preproc_func_outliers.txt', 
             'motion_params':output_dir + '/FD_out_metric_values/{subject_id}/FD_full.txt',
             'wm':template_wm}
selectfiles = Node(SelectFiles(templates), name='selectfiles')


# In[ ]:

## Pull motion info for all subjects

motion_df_file = output_dir + '/motion_summary/motionSummary.csv'

if isdir(output_dir + '/motion_summary') ==False:
    makedirs(output_dir + '/motion_summary')
    motion_df = DataFrame(columns=['meanFD','maxFD','NumCensoredVols','totalVolumes'])
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

    motion_df.loc[subject] = [mean(motion),max(motion),len(censvols),len(motion)]
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
    cvt.inputs.out_file = 'func_filtered.nii.gz'
    cvt.run()
    
    out_file = path.abspath('func_filtered.nii.gz')
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

denoise = Node(GLM(out_res_name='denoised_residuals.nii.gz', 
                   out_data_name='denoised_func.nii.gz'), 
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
                   (selectfiles,compcor,[('func','realigned_file')]),
                   (selectfiles,merge_confs,[('csf','mask1')]),
                   (selectfiles,merge_confs,[('wm','mask2')]),
                   (merge_confs,compcor,[('vols','mask_files')]),
                   (compcor,noise_mat,[('components_file','comp_noise')]),
                   (selectfiles,noise_mat,[('vols_to_censor','vols_to_censor'),
                                           ('motion_params','motion_params')]),
                   (noise_mat,denoise,[('noise_filepath','design')]),
                   (selectfiles,denoise,[('func','in_file')]),
                   (denoise,bandpass,[('out_data','in_file')]),
                   (bandpass,afni_convert,[('out_file','in_file')]),
                   (afni_convert,brightthresh_filt,[('out_file','func')]),
                   (brightthresh_filt,smooth_filt,[('bright_thresh','brightness_threshold')]),
                   (afni_convert,smooth_filt,[('out_file','in_file')]),  
                   (selectfiles, motion_summary, [('motion_params','motion_file'),
                                                  ('vols_to_censor','vols_to_censor')]),
                   
                   (smooth_filt,datasink,[('smoothed_file','preproc_func')])
                   ])

rs_procwf.base_dir = workflow_dir
rs_procwf.write_graph(graph2use='flat')
rs_procwf.run('MultiProc', plugin_args={'n_procs': proc_cores})


