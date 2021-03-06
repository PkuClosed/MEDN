# MEDN 

This is a python demo for the paper:<br />
**Chuyang Ye, "Estimation of Tissue Microstructure Using a Deep Network Inspired by a Sparse Reconstruction Framework", IPMI 2017.** and its extended version **Chuyang Ye, "Tissue Microstructure Estimation Using a Deep Network Inspired by a Dictionary-Based Framework", MedIA 2017.**

The demo includes both the training and test phase. Therefore, to run it, both the training and test data (which are images in the NIfTI format) should be prepared. The input diffusion signals should be normalized.

There are a few dependencies that need to be installed:<br />
**numpy <br />
nibabel <br />
keras <br />
theano <br />**

Here is how to run the scripts <br />
>python MEDN.py < list of training normalized diffusion images> < list of training brain mask images > < list of training ICVF > < list of training ISO > < list of training OD >   < list of test normalized diffusion images > < list of test brain mask images > < output directory > <br />

and 

>python PMEDN.py < list of training normalized diffusion images> < list of training brain mask images > < list of training ICVF > < list of training ISO > < list of training OD >   < list of test normalized diffusion images > < list of test brain mask images > < output directory > <br />

For example, <br />
>python MEDN.py dwis_training.txt masks_training.txt icvfs_training.txt isos_training.txt ods_training.txt dwis_test.txt masks_test.txt output

For more questions, please contact me via chuyang.ye@bit.edu.cn or pkuclosed@gmail.com
