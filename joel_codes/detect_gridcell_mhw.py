#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:57:23 2021

Detect MHW in ROMS-SO using a percentile threshold, under different baselines

@author: jwongmeng
"""
#%% Initialise

import numpy as np
import xarray as xr
import datetime
import os
import copy as cp
from joblib import Parallel, delayed

years = np.arange(1980,2020,1)
nyears = np.size(years)
months = np.arange(1,13,1)
days = np.arange(0,365,1)
ndays = np.size(days)
nz = 35
neta = 434
nxi = 1442

# Limit domain by eta
latmaxidx = 360
neta = latmaxidx

#%% USER INPUT

# Name of extreme
ext = "mhw"

# What baseline?
#baseline = 'fixed'
#baseline = 'fixed2019'
baseline = 'moving'

# Which climatology method?
#climt = "mean"
climt = "median"

# What percentile for detection?
percentile = 95

# Input data directory
in_dir = "/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/"

#%% Preprocessing

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
    demod = 'poly2'
    #demod = 'poly1'
    mdir = 'sea'
    
# Root input directories
trend_dir = '/nfs/meso/work/jwongmeng/ROMS/analysis/trend/'
th_dir = "/nfs/meso/work/jwongmeng/ROMS/analysis/thresholds/"
clim_dir = "/nfs/meso/work/jwongmeng/ROMS/analysis/climatologies/"

if ext == 'mhw':
    var = 'temp'

# Identifier for percentile threshold
pmod = 'p' + str(percentile)

# Trend input file
if demod == "poly1": 
    trend_deg = 1
    
elif demod == "poly2":
    trend_deg = 2
    
# Polynomial fit
fit_fn = trend_dir + var +  "_polynomial_deg_" + str(trend_deg) + '.nc' 

# Threshold filename
th_fn = th_dir + ext + "/thresh_detrend" + '_' + demod + '_' + pmod + '.nc'

# Climatology filename
if climt == "mean":
    clim_fn = clim_dir + ext + "/clim_detrend_" + demod + "_mean.nc"
elif climt == "median":
    clim_fn = clim_dir + ext + "/clim_detrend_" + demod + "_median.nc"
        
### Output filenames ###

# Generate tvname
tvname = thvar + "_" + demod + "_" + str(percentile)

# Root output directories
det_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/detect/' + tvname + '/'
stats_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/stats/' + tvname + '/'
int_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/intensity/' + tvname + '/' + climt + "/"
idx_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/intensity_index/' + tvname + '/' + climt + "/"

# Detect directory
det_gsx_dir = det_dir + ext + "/"

# Intensity directory
int_gsx_dir = int_dir + ext + "/"

# Intensity Index directory
idx_gsx_dir = idx_dir + ext + "/"

# Create output directories
if not os.path.exists(det_gsx_dir):
    os.makedirs(det_gsx_dir)

if not os.path.exists(int_gsx_dir):
    os.makedirs(int_gsx_dir)
    
if not os.path.exists(idx_gsx_dir):
    os.makedirs(idx_gsx_dir)

# Load coords
time_daily_1979 = np.load('/home/jwongmeng/work/ROMS/scripts/coords/time_daily_1979.npy', allow_pickle=True)
time_daily_1980 = np.load('/home/jwongmeng/work/ROMS/scripts/coords/time_daily_1980.npy', allow_pickle=True)
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy') 

#%% Construct trend coefficient matrix for moving baseline detection

if thvar == 'mb' or thvar == 'uptrend':   
    ds_fit = xr.open_dataset(fit_fn) #dim: (trend_deg: 3month: 12z_rho: 35eta_rho: 434xi_rho: 1442)
    coeffs = ds_fit.coeffs[1:,:,:,:,:].values
    
    if demod == 'poly2':

        psto = np.full((2,ndays,nz,neta,nxi),0.0,dtype=np.float32) #5D
        
        # -- from monthly to daily
        psto[0,0:31,:,:,:] = coeffs[0,0,0:nz,0:neta,:] # Days 1-31 (Jan) → Coeff 0,0
        psto[0,31:59,:,:,:] = coeffs[0,1,0:nz,0:neta,:]
        psto[0,59:90,:,:,:] = coeffs[0,2,0:nz,0:neta,:]
        psto[0,90:120,:,:,:] = coeffs[0,3,0:nz,0:neta,:]
        psto[0,120:151,:,:,:] = coeffs[0,4,0:nz,0:neta,:]
        psto[0,151:181,:,:,:] = coeffs[0,5,0:nz,0:neta,:]
        psto[0,181:212,:,:,:] = coeffs[0,6,0:nz,0:neta,:]
        psto[0,212:243,:,:,:] = coeffs[0,7,0:nz,0:neta,:]
        psto[0,243:273,:,:,:] = coeffs[0,8,0:nz,0:neta,:]
        psto[0,273:304,:,:,:] = coeffs[0,9,0:nz,0:neta,:]
        psto[0,304:334,:,:,:] = coeffs[0,10,0:nz,0:neta,:]
        psto[0,334:365,:,:,:] = coeffs[0,11,0:nz,0:neta,:]
        
        psto[1,0:31,:,:,:] = coeffs[1,0,0:nz,0:neta,:]
        psto[1,31:59,:,:,:] = coeffs[1,1,0:nz,0:neta,:]
        psto[1,59:90,:,:,:] = coeffs[1,2,0:nz,0:neta,:]
        psto[1,90:120,:,:,:] = coeffs[1,3,0:nz,0:neta,:]
        psto[1,120:151,:,:,:] = coeffs[1,4,0:nz,0:neta,:]
        psto[1,151:181,:,:,:] = coeffs[1,5,0:nz,0:neta,:]
        psto[1,181:212,:,:,:] = coeffs[1,6,0:nz,0:neta,:]
        psto[1,212:243,:,:,:] = coeffs[1,7,0:nz,0:neta,:]
        psto[1,243:273,:,:,:] = coeffs[1,8,0:nz,0:neta,:]
        psto[1,273:304,:,:,:] = coeffs[1,9,0:nz,0:neta,:]
        psto[1,304:334,:,:,:] = coeffs[1,10,0:nz,0:neta,:]
        psto[1,334:365,:,:,:] = coeffs[1,11,0:nz,0:neta,:]
        
    
    elif demod == 'poly1':

        psto = np.full((ndays,nz,neta,nxi),0.0,dtype=np.float32) #4D
        
        psto[0:31,:,:,:] = coeffs[0,0,0:nz,0:neta,:]
        psto[31:59,:,:,:] = coeffs[0,1,0:nz,0:neta,:]
        psto[59:90,:,:,:] = coeffs[0,2,0:nz,0:neta,:]
        psto[90:120,:,:,:] = coeffs[0,3,0:nz,0:neta,:]
        psto[120:151,:,:,:] = coeffs[0,4,0:nz,0:neta,:]
        psto[151:181,:,:,:] = coeffs[0,5,0:nz,0:neta,:]
        psto[181:212,:,:,:] = coeffs[0,6,0:nz,0:neta,:]
        psto[212:243,:,:,:] = coeffs[0,7,0:nz,0:neta,:]
        psto[243:273,:,:,:] = coeffs[0,8,0:nz,0:neta,:]
        psto[273:304,:,:,:] = coeffs[0,9,0:nz,0:neta,:]
        psto[304:334,:,:,:] = coeffs[0,10,0:nz,0:neta,:]
        psto[334:365,:,:,:] = coeffs[0,11,0:nz,0:neta,:]
    
    del coeffs        


#%% Function to detect grid cell single extremes

def detect_gsx(ieta):
    
    # Read data file
    fn = in_dir + var + '_eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    dav = xr.open_dataset(fn)[var][1:,0:365,:,:].values #Extracts daily data, skipping the first time step (only 40yr) + consider 365 days per year
    
    # For moving baseline detection, add the trend to the threshold (don't overwrite dav_th, used in every loop)
    if thvar == 'mb': #moving baseline
        if demod == 'poly2':
            # Increase threshold each year
            threshold = da_th[:,:,ieta,:].values + np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) + np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
        elif demod == 'poly1':
            # Increase threshold each year
            threshold = da_th[:,:,ieta,:].values + np.multiply.outer((years - 1979),psto[:,:,ieta,:])
        
    elif thvar == 'uptrend':
        if demod == 'poly2':
            # Decrease threshold each year
            threshold = da_th[:,:,ieta,:].values - np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) - np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])

    elif thvar == 'detrend':
        # Threshold is the same every year (1979)
        threshold = da_th[:,:,ieta,:].values
        
    # Set nans to 0 for so that det = False at nans
    threshold[np.isnan(threshold)] = 0
    dav[np.isnan(dav)] = 0 

    # Check if vals is greater than threshold and store in det (np doesn't allow greater for 4D xr arrays)
    det = np.greater(dav,threshold)

    
    del dav, threshold
    
    # Save Results
    ds_det = xr.Dataset(
        data_vars = dict(
            detect = (["years","days","z_rho","xi_rho"], det),
            ),
        coords = dict(
            z_rho = (["z_rho"],z_rho),
            lon_rho = (["eta_rho","xi_rho"],ds_roms.lon_rho.values),
            lat_rho = (["eta_rho","xi_rho"],ds_roms.lat_rho.values)
            ),
        attrs = dict(description = "Detected " + ext + " (boolean array)")
        )

    ds_det.to_netcdf(det_gsx_dir + "det_" + str(ieta) + ".nc",'w')  
    
    print(str(ieta) + " complete")

    del det, ds_det
    
#%% Run detect_gsx in parallel

# Load threshold values
da_th = xr.open_dataset(th_fn).threshold #extracts threshold variable (predefined threshold values used for extremes detection)

njobs = 30

print("Detect gsx and write detect file")
Parallel(n_jobs=njobs)(delayed(detect_gsx)(ieta) for ieta in range(0,neta)) # detects extremes for each latitude in parallel 

print("detect extremes completed for " + ext + " " + tvname)

#%% Function to calculate intensities and index

def calc_intensity_index(ieta): 
    
    '''Computes two key metrics for extreme event detection: 
        (1) Intensity (difference between the data and threshold)
        (2) Index (normalized anomaly relative to climatology)
    '''
    
    # Read data file
    fn = in_dir + var + '_eta' + str(ieta) + '.nc'
    dav = xr.open_dataset(fn)[var][1:,0:365,:,:].values #Extracts daily data, skipping the first time step (only 40yr) + consider 365 days per year

    # For moving baseline detection, add the trend to the threshold & climatology (don't overwrite dav_th/da_clim, used in every loop)
    if thvar == 'mb':
        if demod == 'poly2':
            threshold = da_th[:,:,ieta,:].values + np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) + np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
            dav_clim = da_clim[:,:,ieta,:].values + np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) + np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
    elif thvar == 'uptrend':
        if demod == 'poly2':
            threshold = da_th[:,:,ieta,:].values - np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) - np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
            dav_clim = da_clim[:,:,ieta,:].values - np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) - np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
    elif thvar == 'detrend':
        threshold = da_th[:,:,ieta,:].values
        dav_clim = da_clim[:,:,ieta,:].values
        
    # Load mask to remove non-extreme values
    mask = xr.open_dataset(det_gsx_dir + "det_" + str(ieta) + ".nc").detect.values #True where an extreme event is detected, False otherwise

    # Calculate intensity and write to netcdf
    dav_int = np.float32(dav - threshold) #Intensity: Difference between observed data (dav) and threshold.
    dav_int[~mask] = np.nan # Set to nan when there is no extreme

    # Write intensity data
    ds_intensity = xr.Dataset(
        data_vars = dict(
            intensity = (["years","days","z_rho","xi_rho"], dav_int),
            ),
        coords = dict(
            z_rho = (["z_rho"],z_rho),
            lon_rho = (["eta_rho","xi_rho"],ds_roms.lon_rho.values),
            lat_rho = (["eta_rho","xi_rho"],ds_roms.lat_rho.values)
            ),
        attrs = dict(description = ext + " Intensity")
        )
    
    ds_intensity.to_netcdf(int_gsx_dir + "intensity_" + str(ieta) + ".nc",'w')
    
    del dav_int, ds_intensity
    
    # Calculate index and write to netcdf
    dav_idx = np.float32(np.divide(dav - dav_clim,(threshold- dav_clim))) #Normalized anomaly= idx= (obs-clim)/(thresh-clim) -- Values closer to 1 indicate a stronger event
    dav_idx[~mask] = np.nan # Set to nan when there is no ext

    # Saves index data
    ds_index = xr.Dataset(
        data_vars = dict(
            index = (["years","days","z_rho","xi_rho"], dav_idx),
            ),
        coords = dict(
            z_rho = (["z_rho"],z_rho),
            lon_rho = (["eta_rho","xi_rho"],ds_roms.lon_rho.values),
            lat_rho = (["eta_rho","xi_rho"],ds_roms.lat_rho.values)
            ),
        attrs = dict(description = ext + " Index")
        )
    
    ds_index.to_netcdf(idx_gsx_dir + "index_" + str(ieta) + ".nc",'w')
    
    print(str(ieta) + " complete")
    
    del dav_idx, ds_index
    
#%% Calculate intensities (above threshold), and index
    
print("Calculating intensity and index")

# Load climatology
da_clim = xr.open_dataset(clim_fn).climatology

# Load threshold values
da_th = xr.open_dataset(th_fn).threshold
    
njobs = 20

Parallel(n_jobs=njobs)(delayed(calc_intensity_index)(ieta) for ieta in range(301,neta))

print("intensities and index completed for " + ext + " " + tvname)



#%% Function to calculate anomalies

def calc_anomaly(ieta):
    
    # Read data file
    fn = in_dir + var + '_eta' + str(ieta) + '.nc'
    dav = xr.open_dataset(fn)[var][1:,0:365,:,:].values #var=temp, skip first year (only 40yrs), select all days while considering 1yr=365 days, keep all spatial dimensions
        
    # For moving baseline detection, add the trend to the threshold & climatology (don't overwrite dav_th/da_clim, used in every loop)
    if thvar == 'mb':
        if demod == 'poly2':
            dav_clim = da_clim[:,:,ieta,:].values + np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) + np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
    elif thvar == 'uptrend':
        if demod == 'poly2':
            dav_clim = da_clim[:,:,ieta,:].values - np.multiply.outer((np.square(years) - np.square(1979)),psto[1,:,:,ieta,:]) - np.multiply.outer((years - 1979),psto[0,:,:,ieta,:])
    elif thvar == 'detrend':
        dav_clim = da_clim[:,:,ieta,:].values
        
    # Load mask to remove non-extreme values
    mask = xr.open_dataset(det_gsx_dir + "det_" + str(ieta) + ".nc").detect[1:].values #extreme detection file (nan when non extreme)

    # Calculate anomaly and write to netcdf
    dav_anm = np.float32(dav - dav_clim)
    dav_anm[~mask] = np.nan # Set to nan when there is no ext
    
    # Save results
    ds_anomaly = xr.Dataset(
        data_vars = dict(
            anomaly = (["years","days","z_rho","xi_rho"], dav_anm),
            ),
        coords = dict(
            z_rho = (["z_rho"],z_rho),
            lon_rho = (["eta_rho","xi_rho"],ds_roms.lon_rho.values),
            lat_rho = (["eta_rho","xi_rho"],ds_roms.lat_rho.values)
            ),
        attrs = dict(description = ext + " anomaly")
        )
    
    ds_anomaly.to_netcdf(int_gsx_dir + "anomaly_" + str(ieta) + ".nc",'w')
    
    del dav_anm, ds_anomaly
    
    print(str(ieta) + " complete")
    
#%% Calculate anomalies (above climatology)
    
print("Calculating anomaly")

# Load climatology
da_clim = xr.open_dataset(clim_fn).climatology
    
njobs = 20

Parallel(n_jobs=njobs)(delayed(calc_anomaly)(ieta) for ieta in range(0,neta))

print("intensities and index completed for " + ext + " " + tvname)
