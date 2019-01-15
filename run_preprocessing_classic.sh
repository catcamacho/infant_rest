#! /bin/bash
sdev --cores-4
ml load python/3.6.1

#import software modules
ml load biology
ml load fsl/5.0.10
ml load afni/18.2.04
ml load freesurfer/6.0.0
ml load viz
ml load graphviz


#install packages locally
# pip3 install nipype --user
# pip3 install nipy --user #no module
# pip3 install sklearn --user
# pip3 install nilearn --user #no module
# pip3 install graphviz --user #no module

#import python modules
ml load py-pandas/0.23.0_py36
ml load py-numpy/1.14.3_py36
ml load py-nipype/1.1.3_py36
ml load math py-scikit-learn/0.19.1_py36
ml load py-matplotlib/2.2.2_py36
python3 preprocessing_classic.py
