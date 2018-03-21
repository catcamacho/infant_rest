
# coding: utf-8

# In[ ]:


#import packages
from nipype.interfaces.io import DataSink, SelectFiles, DataGrabber # Data i/o
from nipype.interfaces.utility import IdentityInterface, Function     # utility
from nipype.pipeline.engine import Node, Workflow, JoinNode,MapNode      # pypeline engine

from nipype.interfaces.fsl.model import Randomise, GLM, Cluster
from nipype.interfaces.freesurfer.model import Binarize
from nipype.interfaces.fsl.utils import ImageMeants, Merge, Split
from nipype.interfaces.fsl.maths import ApplyMask

#set output file type for FSL to NIFTI
from nipype.interfaces.fsl.preprocess import FSLCommand
FSLCommand.set_default_output_type('NIFTI_GZ')

# MATLAB setup - Specify path to current SPM and the MATLAB's default mode
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('~/spm12')
MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")

# Set study variables
studyhome = '/Users/catcamacho/Box/SNAP/BABIES'
#studyhome = '/Volumes/iang/active/BABIES/BABIES_rest'
raw_data = studyhome + '/raw'
preproc_dir = studyhome + '/processed/preproc'
output_dir = studyhome + '/processed/sbc_analysis'
workflow_dir = studyhome + '/workflows'
roi_dir = studyhome + '/ROIs'
group_con = studyhome + '/misc/tcon.con'
group_mat = studyhome + '/misc/design.mat'
proc_cores = 2

#subjects_list = ['021-BABIES-T1']
subjects_list = open(studyhome + '/misc/subjects.txt').read().splitlines()

template_brain = studyhome + '/templates/T2w_BABIES_template_2mm.nii'
gm_template = studyhome + '/templates/BABIES_gm_mask_2mm.nii'

# ROIs for connectivity analysis
Lamyg = roi_dir + '/L_amyg_4mm.nii'
Ramyg = roi_dir + '/R_amyg_4mm.nii'

ROIs = [Lamyg, Ramyg]
rois = ['L_amyg','R_amyg']

min_clust_size = 25


# In[ ]:


## File handling
# Identity node- select subjects
infosource = Node(IdentityInterface(fields=['subject_id','ROIs']),
                     name='infosource')
infosource.iterables = [('subject_id', subjects_list),('ROIs',ROIs)]


# Data grabber- select fMRI and ROIs
templates = {'func': preproc_dir + '/preproc_func/{subject_id}/func_filtered_smooth.nii',
             'motion': preproc_dir + '/FD_out_metric_values/{subject_id}/FD.txt'}
selectfiles = Node(SelectFiles(templates), name='selectfiles')

# Datasink- where our select outputs will go
datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir
substitutions = [('_subject_id_', ''),
                ('_ROIs_..Users..catcamacho..Box..SNAP..BABIES..ROIs..','')]
datasink.inputs.substitutions = substitutions


# In[ ]:


## Seed-based level 1

# Extract ROI timeseries
ROI_timeseries = Node(ImageMeants(), name='ROI_timeseries', iterfield='mask')

# model ROI connectivity
glm = Node(GLM(out_file='betas.nii.gz',out_cope='cope.nii.gz'), name='glm', iterfield='design')


# In[ ]:


sbc_workflow = Workflow(name='sbc_workflow')
sbc_workflow.connect([(infosource,selectfiles,[('subject_id','subject_id')]),
                      (selectfiles,ROI_timeseries,[('func','in_file')]),
                      (infosource,ROI_timeseries,[('ROIs','mask')]),
                      (ROI_timeseries,glm,[('out_file','design')]),
                      (selectfiles,glm,[('func','in_file')]),
                      #(ROI_timeseries, datasink, [('out_file','roi_ts')]),
                      (glm,datasink,[('out_cope','glm_seed_copes')]),
                      (glm,datasink,[('out_file','glm_betas')])
                     ])
sbc_workflow.base_dir = workflow_dir
sbc_workflow.write_graph(graph2use='flat')
sbc_workflow.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:


infosource2 = Node(IdentityInterface(fields=['roi']),
                   name='infosource2')
infosource2.iterables = ('roi',rois)


# Data grabber- select fMRI and ROIs
templates = {'roi': 'glm_betas/%s_4mm.nii*/betas.nii.gz'}

datagrabber = Node(DataGrabber(infields=['roi'], 
                               outfields=['roi'],
                               sort_filelist=True,
                               base_directory=output_dir,
                               template='glm_betas/%s_4mm.nii*/betas.nii.gz',
                               field_template=templates,
                               template_args=dict(roi=[['roi']])),
                   name='datagrabber')


# In[ ]:


def extract_cluster_betas(cluster_index_file, sample_betas, min_clust_size, subject_ids):
    from nipype import config, logging
    config.enable_debug_mode()
    logging.update_logging(config)
    from nibabel import load, save, Nifti1Image
    from pandas import DataFrame, Series
    from numpy import unique, zeros_like
    from nipype.interfaces.fsl.utils import ImageMeants
    from os.path import abspath
    
    subject_ids = sorted(subject_ids)
    sample_data = DataFrame(subject_ids, index=None, columns=['Subject'])
    
    cluster_nifti = load(cluster_index_file[0])
    cluster_data = cluster_nifti.get_data()
    clusters, cluster_sizes = unique(cluster_data, return_counts=True)
    
    final_clusters = clusters[cluster_sizes>=min_clust_size]
    for clust_num in final_clusters[1:]:
        temp = zeros_like(cluster_data)
        temp[cluster_data==clust_num] = 1
        temp_nii = Nifti1Image(temp,cluster_nifti.affine)
        temp_file = 'temp_clust_mask.nii.gz'
        save(temp_nii, temp_file)
        
        eb = ImageMeants()
        eb.inputs.in_file = sample_betas
        eb.inputs.mask = temp_file
        eb.inputs.out_file = 'betas.txt'
        eb.run()
        betas = open('betas.txt').read().splitlines()
        sample_data['clust' + str(clust_num)] = Series(betas, index=sample_data.index)
    
    sample_data.to_csv('extracted_betas.csv')
    extracted_betas_csv = abspath('extracted_betas.csv')
    return(extracted_betas_csv)


# In[ ]:


## Level 2

# merge param estimates across all subjects per seed
merge = Node(Merge(dimension='t'),
             name='merge')

# FSL randomise for higher level analysis
highermodel = Node(Randomise(tfce=True,
                             raw_stats_imgs= True,
                             design_mat=group_mat,
                             mask=gm_template,
                             tcon=group_con),
                   name = 'highermodel')

## Cluster results
# make binary masks of sig clusters
binarize = MapNode(Binarize(min=0.99, max=1.0), 
                   name='binarize', 
                   iterfield=['in_file'])

# mask T-map before clustering
mask_tmaps = MapNode(ApplyMask(), 
                     name='mask_tmaps',
                     iterfield=['in_file','mask_file'])

# clusterize and extract cluster stats/peaks
clusterize = MapNode(Cluster(threshold=5, 
                             out_index_file='outindex.nii.gz', 
                             out_localmax_txt_file='localmax.txt'), 
                     name='clusterize',
                     iterfield=['in_file'])

extract_betas = Node(Function(input_names=['cluster_index_file','sample_betas','min_clust_size','subject_ids'],
                              output_names=['extracted_betas_csv'],
                              function=extract_cluster_betas),
                     name='extract_betas')
extract_betas.inputs.min_clust_size = min_clust_size
extract_betas.inputs.subject_ids = subjects_list



# In[ ]:


group_workflow = Workflow(name='group_workflow')
group_workflow.connect([(infosource2,datagrabber,[('roi','roi')]),
                        (datagrabber,merge,[('roi','in_files')]),
                        (merge,highermodel,[('merged_file','in_file')]),
                        (highermodel, binarize, [('t_corrected_p_files','in_file')]),
                        (binarize, mask_tmaps, [('binary_file','mask_file')]),
                        (highermodel, mask_tmaps, [('tstat_files','in_file')]),
                        (mask_tmaps, clusterize, [('out_file','in_file')]),
                        (merge, extract_betas, [('merged_file','sample_betas')]),
                        (clusterize, extract_betas, [('index_file','cluster_index_file')]),

                        (highermodel,datasink,[('t_corrected_p_files','rand_corrp_files')]),
                        (highermodel,datasink,[('tstat_files','rand_tstat_files')]),
                        (clusterize,datasink,[('index_file','cluster_index_file')]),
                        (clusterize,datasink,[('localmax_txt_file','localmax_txt_file')]),
                        (merge, datasink, [('merged_file','merged_betas')]),
                        (extract_betas, datasink, [('extracted_betas_csv','all_cluster_betas')])
                       ])
group_workflow.base_dir = workflow_dir
group_workflow.write_graph(graph2use='flat')
group_workflow.run('MultiProc', plugin_args={'n_procs': proc_cores})

