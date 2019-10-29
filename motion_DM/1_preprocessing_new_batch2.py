
# coding: utf-8

# # Infant resting state fMRI preprocessing
# This notebook contains preprocessing tailored to infant resting state fMRI collected in 5-8 month olds. 
# 
# The processing steps for the fMRI broadly include:
# * Slice-time correction
# * Rigid realignment
# * Co-registration to the sMRI (T2-weighted structural MRI)
# * Co-registration to template
# * De-noising to remove:
#     - Mean timeseries for that voxel
#     - Component noise associated with white matter and CSF- delete the GM and smooth what is left
#     - Component noise associated with background signal - delete brain and smooth what's left
#     - Component noise from the averaged timeseries
#     - motion regressors
#     - Motion derivatives (lagged 6 times)
#     - Squared derivatives (lagged 6 times) as an exploratory
# * Bandpass filtering

# In[1]:


#import packages
from os import listdir, makedirs
from os.path import isdir
from nipype.interfaces.io import DataSink, SelectFiles, DataGrabber # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow, MapNode, JoinNode        # pypeline engine
from nipype.interfaces.nipy.preprocess import Trim
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.fsl import SliceTimer, MCFLIRT, FLIRT, BET
from nipype.interfaces.fsl.utils import Reorient2Std, MotionOutliers
from nipype.interfaces.fsl.maths import ApplyMask, MeanImage
from nipype.interfaces.freesurfer import Resample, Binarize
from nipype.algorithms.confounds import CompCor
from nipype.interfaces.afni.preprocess import Bandpass
from nipype.interfaces.afni.utils import AFNItoNIFTI
from pandas import DataFrame, Series,read_csv

#set output file type for FSL to NIFTI_GZ
from nipype.interfaces.fsl.preprocess import FSLCommand
FSLCommand.set_default_output_type('NIFTI_GZ')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12/toolbox')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
#studyhome = '/Users/catcamacho/Box/SNAP/BABIES'
studyhome = '/home/camachocm2/Analysis/SNAP'
raw_data = studyhome + '/raw'
output_dir = studyhome + '/processed/preproc'
workflow_dir = studyhome + '/workflows'
subjects_info = read_csv(studyhome + '/misc/rest_subjects_info.csv',index_col=None, dtype={'subject_id':str})
subjects_info['subject_id'] = subjects_info['subject_id'].apply(lambda x: x.zfill(4))
subjects_list = subjects_info['subject_id'].tolist()
subjects_list = subjects_list[43:57]

template_brain = studyhome + '/templates/6mo_T2w_template_2mm.nii.gz'
template_mask = studyhome + '/templates/6mo_T2w_template_2mm_mask.nii.gz'
template_gmmask = studyhome + '/templates/6mo_T2w_template_2mm_gm.nii.gz'
template_nongm = studyhome + '/templates/6mo_T2w_template_2mm_nongm.nii.gz'
template_nonbrain = studyhome + '/templates/6mo_T2w_template_2mm_nonbrain.nii.gz'
full_image = studyhome + '/templates/6mo_T2w_template_2mm_fullimage.nii.gz'

vols_to_trim = 4
interleave = False
TR = 2.5 # in seconds
slice_dir = 3 # 1=x, 2=y, 3=z
resampled_voxel_size = (2,2,2)
fwhm = 4 #fwhm for smoothing with SUSAN
anat_type='t2'

#changed to match Pendl et al 2017 (HBM)
highpass_freq = 0.005 #in Hz
lowpass_freq = 0.1 #in Hz


# In[2]:


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





# ## Denoising Workflow
# 
# The nodes and workflow below (denoise_flow) is designed to take the nuissance regressors created in the previous section (create_noise_flow) and perform voxel-specific denoising.  This is accomplished through the following steps:
# 1. Voxel-specific denoising
#     - Create unique design matrix for each 3D voxel
#     - Perform a GLM for that voxel
#     - Project results back into 3D space
# 2. Bandpass filtering [0.001:0.08]
# 3. Concatenate and realign multiple runs

# In[ ]:


means_template={'mean_func': output_dir + '/mean_func/mean_funcs.nii.gz', 
                'mean_func_components': output_dir + '/mean_func/components.txt'}
select_mean_noise = Node(SelectFiles(means_template), name='select_mean_noise')

sub_files_template={'leadlagmotion': output_dir + '/leadlagmotion/{subject_id}/_prep_motion{runnum}/leadlag.txt', 
                    'leadlagderivsmotion': output_dir + '/leadlagderivsmotion/{subject_id}/_prep_motion{runnum}/derivsleadlag.txt', 
                    'leadlagderivs_squaremotion': output_dir + '/leadlagderivs_squaremotion/{subject_id}/_prep_motion{runnum}/derivssqleadlag.txt', 
                    'func': output_dir + '/registered_func/{subject_id}/_xfmFUNC{runnum}/realigned_func.nii.gz', 
                    'wmcsf': output_dir + '/subject_wmcsf_comp_noise/{subject_id}/_comp_wmcsf_noise{runnum}/components.txt', 
                    'session': output_dir + '/subject_session_comp_noise/{subject_id}/_comp_session_noise{runnum}/components.txt'}
select_sub_files=Node(SelectFiles(sub_files_template),name='select_sub_files')
select_sub_files.iterables=('runnum',['0','1'])


# In[77]:


def org_shared_noise(leadlagmotion, leadlagderivsmotion, leadlagderivs_squaremotion, wmcsf, session, mean_func_components):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from numpy import loadtxt, concatenate
    from pandas import DataFrame
    from os.path import abspath
    
    noise_list = []
    for file in [leadlagmotion, leadlagderivsmotion, leadlagderivs_squaremotion]:
        mo = loadtxt(file, dtype=float, comments=None)
        length_of_file = mo.shape[0]
        noise_list.append(mo)
    for file in [wmcsf, session]:
        mo = loadtxt(file,dtype=float, skiprows=1, comments=None)
        length_of_file = mo.shape[0]
        noise_list.append(mo)
    
    mean_func_noise = loadtxt(mean_func_components,skiprows=1, comments=None)
    mean_func_noise_trim = mean_func_noise[:length_of_file,:]
    noise_list.append(mean_func_noise_trim)
    shared_noise_data = concatenate(noise_list,axis=1)
    
    col_names = ['noise_{0}'.format(a) for a in range(0,shared_noise_data.shape[1])] 
    
    shared_noise = DataFrame(shared_noise_data, columns=col_names)
    shared_noise.to_csv('shared_noise.csv')
    shared_noise_file = abspath('shared_noise.csv')
    return(shared_noise_file)

def voxelwise_glm(func,shared_noise_file,mean_func,mask):
    from os.path import abspath
    from glm.glm import GLM
    from glm.families import Gaussian
    from numpy import zeros
    from pandas import read_csv
    from nilearn.masking import apply_mask, unmask
    linear_model=GLM(family=Gaussian())

    # import data into an array that is timepoints (rows) by voxel number (columns)
    shared_noise = read_csv(shared_noise_file, index_col=0)
    func_data = apply_mask(func, mask)
    mean_func_data = apply_mask(mean_func, mask)
    mean_func_data = mean_func_data[:func_data.shape[0],:]
    coefficients = zeros((shared_noise.shape[1]+1,func_data.shape[1]))
    residuals = zeros((func_data.shape))

    # perform voxel-wise GLM
    formula = 'signal ~ mean_signal' 
    for a in range(0,shared_noise.shape[1]-1):
        formula = formula + ' + noise_{0}'.format(a)

    for x in range(0,func_data.shape[1]):
        shared_noise['mean_signal'] = mean_func_data[:,x]
        shared_noise['signal'] = func_data[:,x]
        linear_model.fit(shared_noise, formula=formula)
        resid = shared_noise['signal']-linear_model.predict(shared_noise)
        residuals[:,x] = resid
        coefficients[:,x] = linear_model.coef_


    coeff_image = unmask(coefficients, mask)
    resid_image = unmask(residuals, mask)
    coeff_image.to_filename('weights.nii.gz')
    resid_image.to_filename('resids.nii.gz')
    sample_design_df = shared_noise.to_csv('last_noise_mat.csv')

    weights = abspath('weights.nii.gz')
    sample_design_df = abspath('last_noise_mat.csv')
    denoised_func = abspath('resids.nii.gz')

    return(weights,sample_design_df,denoised_func)

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


# In[ ]:


compile_noise_mat = Node(Function(input_names=['leadlagmotion', 'leadlagderivsmotion', 'leadlagderivs_squaremotion', 
                                               'wmcsf', 'session', 'mean_func_components'],
                                  output_names=['shared_noise_file'],
                                  function=org_shared_noise), 
                         name='compile_noise_mat')

denoise_func = Node(Function(input_names=['func','shared_noise_file','mean_func','mask'], 
                             output_names=['weights','sample_design_df','denoised_func'],
                             function=voxelwise_glm),
                    name='denoise_func')
denoise_func.inputs.mask =template_gmmask

# band pass filtering- all rates are in Hz (1/TR or samples/second)
bandpass = Node(Bandpass(highpass=highpass_freq,
                         lowpass=lowpass_freq), 
                name='bandpass')

afni_convert = Node(Function(input_names=['in_file'],
                             output_names=['out_file'],
                             function=convertafni), 
                    name='afni_convert')

# In[ ]:


denoise_flow = Workflow(name='denoise_flow')
denoise_flow.connect([(infosource, select_sub_files,[('subject_id','subject_id')]),
                      (select_sub_files, denoise_func, [('func','func')]),
                      (select_sub_files, compile_noise_mat, [('leadlagmotion','leadlagmotion'),
                                                             ('leadlagderivsmotion','leadlagderivsmotion'), 
                                                             ('leadlagderivs_squaremotion','leadlagderivs_squaremotion'), 
                                                             ('wmcsf','wmcsf'), 
                                                             ('session','session')]),
                      (compile_noise_mat, denoise_func, [('shared_noise_file','shared_noise_file')]),
                      (denoise_func,bandpass,[('out_data','in_file')]),
                      (bandpass,afni_convert,[('out_file','in_file')]),
                      
                      (select_mean_noise,compile_noise_mat,[('mean_func_components','mean_func_components')]),
                      (select_mean_noise,denoise_func,[('mean_func','mean_func')]),
                      
                      (afni_convert,datasink,[('out_file','fully_processed_func')]),
                      (denoise_func,datasink,[('weights','denoising_weights'),
                                              ('sample_design_df','denoise_sample_design_df'),
                                              ('denoised_func','denoised_func')]),
                      
                     ])
denoise_flow.base_dir = workflow_dir
denoise_flow.write_graph(graph2use='flat')
denoise_flow.run('MultiProc', plugin_args={'n_procs': 1})