
# coding: utf-8

# In[ ]:

#import packages
from os import listdir
from nipype.interfaces.io import DataSink, SelectFiles 
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.fsl.preprocess import MCFLIRT
from nipype.interfaces.fsl import PlotMotionParams

# Set study variables
studyhome = '/Users/lucyking/Desktop/testmotion'
subject_data = studyhome + '/BABIES_Longitudinal-T2' #path to folder containing individual subject folders
output_dir = studyhome + '/check_motion' #path to new folder where motion plot will be exported
workflow_dir = studyhome + '/workflows'
subjects_list = listdir(subject_data)
pe_list = ['0', '1']

proc_cores = 4 # number of cores of processing for the workflows

# In[ ]:

## File handling Nodes

# Identity node- select subjects and pe
infosource = Node(IdentityInterface(fields=['subject_id', 'pe_id']),
                     name="infosource")
infosource.iterables = [('subject_id', subjects_list),
                        ('pe_id', pe_list)]

# Data grabber- select fMRI 
templates = {'func': subject_data + '/{subject_id}/functional/rest/recon/pe{pe_id}/rest_rest_pe{pe_id}.nii.gz'}

selectfiles = Node(SelectFiles(templates), name='selectfiles')

# Datasink- where our select outputs will go
datasink = Node(DataSink(), name ='datasink')
datasink.inputs.base_directory = output_dir
datasink.inputs.container = output_dir


# In[ ]:

#Node to check motion
motion_correct = Node(MCFLIRT(save_plots = True, 
                              mean_vol = True),
                      name = 'motion_correct')

plot_motion = Node(PlotMotionParams(in_source = "fsl", 
                                    plot_type = "displacement"), 
                   name = 'plot_motion')


# In[ ]:

#Define workflow 

# workflowname.connect([(node1,node2,[('node1output','node2input')]),
#                       (node2,node3,[('node2output','node3input')])
#                     ])
checkmotion_wf = Workflow(name = 'checkmotion_wf')

checkmotion_wf.connect([(infosource, selectfiles,[('subject_id','subject_id')]),
                        (infosource, selectfiles,[('pe_id','pe_id')]),
                        (selectfiles, motion_correct, [('func', 'in_file')]),
                        (motion_correct, plot_motion, [('par_file','in_file')]),
                       
                       (plot_motion, datasink, [('out_file', 'motion_plot')])])    


# In[ ]:

#Run workflow
checkmotion_wf.run('MultiProc', plugin_args={'n_procs': proc_cores})


# In[ ]:



