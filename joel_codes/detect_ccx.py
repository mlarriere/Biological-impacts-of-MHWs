#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:50:10 2022

@author: jwongmeng

Detection of CPX within the vertical column of 150m

"""

#%% Initialise

import numpy as np
import xarray as xr
import scipy
from scipy.ndimage import label
import os
import datetime
import copy
from joblib import Parallel, delayed

np.seterr(divide='ignore', invalid='ignore')

years = np.arange(1980,2020,1)
nyears = np.size(years)
months = np.arange(0,12,1)
nmonths = np.size(months)
days = np.arange(0,365,1)
ndays = np.size(days)
nz = 29
neta = 434
nxi = 1442
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


#%% USER INPUT

# Name of extreme
ext = 'cmhx'
#ext = 'camx'

# What baseline?
#baseline = 'fixed'
#baseline = 'fixed2019'
baseline = 'moving'

# What percentile for detection?
percentile = 95

# Minimum number of grid cell for extreme
mdep = 50

# Define settings for morphing
kernel_temporal_open = 5
kernel_temporal_close = 0

#%% Preprocessing

if ext == 'cmhx':
    csxlist = ['cmhw','chix']
    gsxlist = ['mhw','hix']
elif ext == 'camx':
    csxlist = ['cmhw','casx']
    gsxlist = ['mhw','asx']

if baseline == 'fixed':
    thvar = 'detrend'
    demod = 'poly2'
    mdir = 'meso'
    
if baseline == 'fixed2019':
    thvar = 'uptrend'
    demod = 'poly2'
    mdir = 'sea'
    
elif baseline == 'moving':
    thvar = 'mb'
    #demod = 'poly2'
    demod = 'poly1'
    mdir = 'sea'


# Generate tvname
tvname = thvar + "_" + demod + "_" + str(percentile)

# Root directories
det_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/detect/' + tvname + '/'
stats_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/stats/' + tvname + '/'
idx_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/intensity_index/' + tvname + '/'

# Input GSX detect directories
det_gsx_dir = [det_dir + gsxlist[0] + "/",
                det_dir + gsxlist[1] + "/"]

# Input detect directories for CSX
det_csx_dir = [det_dir + csxlist[0] + "/" + "md" + str(mdep) + "/",
               det_dir + csxlist[1] + "/" + "md" + str(mdep) + "/"]

for ifol in range(0,2):
    if not os.path.exists(det_csx_dir[ifol]):
        os.makedirs(det_csx_dir[ifol])
        
# Output detect folder for CCX
det_ccx_dir = det_dir + ext + "/" + "md" + str(mdep) + "/"

if not os.path.exists(det_ccx_dir):
    os.makedirs(det_ccx_dir) 
    
# Load coords
time_daily_1979 = np.load('/home/jwongmeng/work/ROMS/scripts/coords/time_daily_1979.npy', allow_pickle=True)
time_daily_1980 = np.load('/home/jwongmeng/work/ROMS/scripts/coords/time_daily_1980.npy', allow_pickle=True)
lat_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/lat_rho.npy') 
lon_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/lon_rho.npy') 
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')[0:nz]
delz = np.load('/home/jwongmeng/work/ROMS/scripts/coords/delz.npy')[0:nz]

#%% Morphing functions

def generate_boolean_smoothing_kernel(kernel_size,dimension_name='temporal'):
    if dimension_name == 'temporal':
        iterations = int(np.floor(kernel_size/2))
        boolean_smoothing_kernel = np.zeros((3,3))
        boolean_smoothing_kernel[:,1]=True                ## turn on temporal connection
    return boolean_smoothing_kernel, iterations

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def do_morphing(unsmoothed_array,smoothing_kernel_open,iterations_open,smoothing_kernel_close,iterations_close):
    open_array = scipy.ndimage.binary_opening(unsmoothed_array,structure=smoothing_kernel_open,iterations=iterations_open)
    #smoothed_array = scipy.ndimage.binary_closing(open_array,structure=smoothing_kernel_close,iterations=iterations_close)
    return open_array

def morphological_operations(kernel_temporal_open,kernel_temporal_close,boolean_all):
    #print('Do morphological operations.')
    #print('Generate smoothing kernels.')
    if kernel_temporal_open > 1:
        bsk_temp_open,iter_temp_open = generate_boolean_smoothing_kernel(kernel_temporal_open,'temporal')
    else:
        bsk_temp_open,iter_temp_open = 0,0
        
    if kernel_temporal_close > 1:
        bsk_temp_close,iter_temp_close = generate_boolean_smoothing_kernel(kernel_temporal_close,'temporal')
    else:
        bsk_temp_close,iter_temp_close = 0,0
        
    #print('Apply smoothing kernels to boolean array.')
    it = np.max([iter_temp_open,iter_temp_close])
    if it > 0:   # this means that some smoothing has to be done!
        #print('Pad boolean array.')
        morph = np.pad(boolean_all,it,pad_with)  # pad to allow for correct morphological function behavior along boundaries
        #print('Smooth the boolean array.')
        if np.max([kernel_temporal_open,kernel_temporal_close]) > 1:
            morph = do_morphing(morph,bsk_temp_open,iter_temp_open,bsk_temp_close,iter_temp_close)
        boolean_morphed = morph[it:-it,it:-it]  # reverse of padding function
    else:
        boolean_morphed = boolean_all
    return boolean_morphed

#%% Function to morph CSXs/CCXs

def morph_ccx(det):
    # Load CCX det
    det = np.reshape(det, (nyears*ndays,nxi), order='C')
    
    # Conduct morphing
    det = morphological_operations(kernel_temporal_open,kernel_temporal_close,det)
    det = np.reshape(det, (nyears,ndays,nxi), order='C')
    
    return det

#%% Detect compounds in the column with at least mdep grid cells of each type

def detect_ccx(ieta):

    # Load CSX det computed in previous section
    det_csx_1 = xr.open_dataset(det_csx_dir[0] + "det_all_morphed" + ".nc").detect[:,:,ieta,:].values
        
    det_csx_2 = xr.open_dataset(det_csx_dir[1] + "det_all_morphed" + ".nc").detect[:,:,ieta,:].values
 
    # Compute CCX det
    det_ccx = np.logical_and(det_csx_1, det_csx_2)
    del det_csx_1, det_csx_2
    
    # Morph for 5 day min duration
    det_ccx = morph_ccx(det_ccx)

    print(str(ieta) + ' complete')
    
    return det_ccx
        
#%% Run scripts in parallel

# Detect CCX 
njobs = 20
print("Detecting CCX")

det = np.full((nyears,ndays,neta,nxi), False, dtype=np.bool_)
results = Parallel(n_jobs=njobs)(delayed(detect_ccx)(ieta) for ieta in range(0,neta))

for ieta in range(0,neta):
    det[:,:,ieta,:] = results[ieta]
    

ds = xr.Dataset(
data_vars = dict(
    detect = (["years","days","eta_rho", "xi_rho"], det),
    ),
coords = dict(
    lon_rho = (["eta_rho","xi_rho"],lon_rho),
    lat_rho = (["eta_rho","xi_rho"],lat_rho)
    ),
attrs = dict(description = "Detected c" + ext)
)

ds.to_netcdf(det_ccx_dir + "det_all_morphed" + ".nc",'w')    
       
del det, ds, results
    
print("Detect for CCX complete for " + ext + ' ' + tvname)
