# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:24:06 2016

@author: cyye
"""

import sys
import os
import nibabel as nib
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.layers.advanced_activations import ThresholdedReLU
from keras.layers import merge, Dense, Input
from keras.constraints import nonneg
import time
#%%
def split_last(x):
    Viso = x[:,-1]
    return Viso.reshape((x.shape[0],1))

def split_last_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = 1
    return tuple(shape)
    
def split_others(x):
    Vother = x[:,:-1]
    return Vother

def split_others_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = shape[-1] - 1
    return tuple(shape)
#%%        
dwinames = None
masknames = None
icvfnames = None
isonames = None
odnames = None
pevnames = None
directory = None
trained = None

dwiname = None
maskname = None
modelname = None
pevname = None
    
nLabels = 3 # ICVF, ISO, and OD
if len(sys.argv) == 9:
    dwinames = sys.argv[1]
    masknames = sys.argv[2]
    icvfnames = sys.argv[3]
    isonames = sys.argv[4]
    odnames = sys.argv[5]
    testdwinames = sys.argv[6]
    testmasknames = sys.argv[7]
    directory = sys.argv[8]
else:
    dwinames = ""
    masknames = ""
    icvfnames = ""
    isonames = ""
    odnames = ""
    pevnames = ""
    directory = ""

if os.path.exists(directory) == False:
    os.mkdir(directory)
start = time.time()
###### Training #######
print "Training Phase"    

#### load images
print "Loading"    

with open(dwinames) as f:
    allDwiNames = f.readlines()
with open(masknames) as f:
    allMaskNames = f.readlines()
with open(icvfnames) as f:
    allICVFNames = f.readlines()
with open(isonames) as f:
    alISONames = f.readlines()
with open(odnames) as f:
    allODNames = f.readlines()
allDwiNames = [x.strip('\n') for x in allDwiNames]
allMaskNames = [x.strip('\n') for x in allMaskNames]
allICVFNames = [x.strip('\n') for x in allICVFNames]
alISONames = [x.strip('\n') for x in alISONames]
allODNames = [x.strip('\n') for x in allODNames]

### setting voxels ###
Np = 10
nVox = 0
for iMask in range(len(allMaskNames)):
    print "Counting Voxels for Subject", iMask
    mask = nib.load(allMaskNames[iMask]).get_data()
    # number of voxels
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k] > 0:
                    nVox = nVox + 1
     
dwi = nib.load(allDwiNames[0]).get_data()                   
dwiTraining = np.zeros([nVox, dwi.shape[3]])
icvfTraining = np.zeros([nVox, 1])
isoTraining = np.zeros([nVox, 1])
kappaTraining = np.zeros([nVox, 1])
odTraining = np.zeros([nVox, 1])


print "Initializing Voxel List"
   
nVox = 0
    
for iMask in range(len(allDwiNames)):
    print "Setting Voxel List for Subject:", iMask
    dwi_nii = nib.load(allDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask = nib.load(allMaskNames[iMask]).get_data()
    icvf = nib.load(allICVFNames[iMask]).get_data()
    iso = nib.load(alISONames[iMask]).get_data()
    od = nib.load(allODNames[iMask]).get_data()
    # number of voxels
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] > 0:
                    dwiTraining[nVox,:] = dwi[i,j,k,:]
                    icvfTraining[nVox,0] = icvf[i,j,k]
                    isoTraining[nVox,0] = iso[i,j,k]
                    odTraining[nVox,0] = od[i,j,k]
                    nVox = nVox + 1
                
#%%
### setting architechture ###                    
print "Setting Architechture"
nDict = 301

tau = 1e-10

inputs = Input(shape=(dwiTraining.shape[1],))
W = Dense(nDict, activation='linear', bias = True)(inputs)

TNS = Sequential()

ReLUThres = 0.01
TNS.add(ThresholdedReLU(theta = ReLUThres, input_shape=(nDict,)))
TNS.add(Dense(nDict, activation='linear', bias = True))
Z = TNS(W)
nLayers = 8
for l in range(nLayers-1):
    Y = merge([Z,W],"sum")
    Z = TNS(Y)
Y = merge([Z,W],"sum")
T = ThresholdedReLU(theta = ReLUThres)(Y)

Viso = Lambda(split_last, output_shape=split_last_output_shape)(T)
Vother = Lambda(split_others, output_shape=split_others_output_shape)(T)

normVother = Lambda(lambda x: (x+tau)/(x+tau).norm(1, axis = 1).reshape((x.shape[0],1)))(Vother)
VicK = Dense(2, W_constraint = nonneg(), activation='linear', bias = True)(normVother)
Kappa = Lambda(split_last, output_shape=split_last_output_shape)(VicK)
Vic = Lambda(split_others, output_shape=split_others_output_shape)(VicK)
OD = Lambda(lambda x: 2.0/np.pi*np.arctan(1.0/(x+tau)))(Kappa)

weight = [1.0,1.0,1.0]

epoch = 10
print "nLayers, ReLUThres, epoch, weight, nDict: ", nLayers, ReLUThres, epoch, weight, nDict

### fitting the model ###                    
print "Fitting"    

clf = Model(input=inputs,output=[Vic,OD,Viso])
clf.compile(optimizer=Adam(lr=0.0001), loss='mse', loss_weights = weight)

hist = clf.fit(dwiTraining, [icvfTraining, odTraining, isoTraining], batch_size=128, nb_epoch=epoch, verbose=1, validation_split=0.1)
print(hist.history)
end = time.time()
print "Training took ", (end-start)

#%%###### Test #######
print "Test Phase"    

start = time.time()
with open(testdwinames) as f:
    allTestDwiNames = f.readlines()
with open(testmasknames) as f:
    allTestMaskNames = f.readlines()

allTestDwiNames = [x.strip('\n') for x in allTestDwiNames]
allTestMaskNames = [x.strip('\n') for x in allTestMaskNames]


for iMask in range(len(allTestDwiNames)):
    print "Processing Subject: ", iMask
    #### load images
    print "Loading"  
    dwi_nii = nib.load(allTestDwiNames[iMask])
    dwi = dwi_nii.get_data()
    mask = nib.load(allTestMaskNames[iMask]).get_data()
    print "Counting Voxels"
    nVox = 0
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] > 0:
                    nVox = nVox + 1
                    
    voxelList = np.zeros([nVox, 3], int)
    dwiTest = np.zeros([nVox, dwi.shape[3]])
    
    print "Setting Voxels"
    nVox = 0
    for i in range(dwi.shape[0]):
        for j in range(dwi.shape[1]):
            for k in range(dwi.shape[2]):
                if mask[i,j,k] > 0:
                    dwiTest[nVox,:] = dwi[i,j,k,:]
                    voxelList[nVox,0] = i
                    voxelList[nVox,1] = j
                    voxelList[nVox,2] = k
                    nVox = nVox + 1
    
    rows = mask.shape[0]
    cols = mask.shape[1]
    slices = mask.shape[2]
    
    icvf = np.zeros([rows,cols,slices])
    od = np.zeros([rows,cols,slices])
    iso = np.zeros([rows,cols,slices])
    
    print "Computing"
    icvfList, odList, isoList = clf.predict(dwiTest)
    
    for nVox in range(voxelList.shape[0]):
        x = voxelList[nVox,0]
        y = voxelList[nVox,1]
        z = voxelList[nVox,2]
        icvf[x,y,z] = icvfList[nVox,0]
        od[x,y,z] = odList[nVox,0]
        iso[x,y,z] = isoList[nVox,0]
            
    hdr = dwi_nii.header
    icvf_nii = nib.Nifti1Image(icvf, dwi_nii.get_affine(), hdr)
    icvf_name = os.path.join(directory,"DN_AMICO_ICVF_sub_" + str(iMask) + ".nii.gz")
    icvf_nii.to_filename(icvf_name)
    
    od_nii = nib.Nifti1Image(od, dwi_nii.get_affine(), hdr)
    od_name = os.path.join(directory,"DN_AMICO_OD_sub_" + str(iMask) + ".nii.gz")
    od_nii.to_filename(od_name)
    
    iso_nii = nib.Nifti1Image(iso, dwi_nii.get_affine(), hdr)
    iso_name = os.path.join(directory,"DN_AMICO_ISO_sub_" + str(iMask) + ".nii.gz")
    iso_nii.to_filename(iso_name)
    
end = time.time()
print "Test took ", (end-start)
    
