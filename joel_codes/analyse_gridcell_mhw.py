#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 22:34:17 2024

@author: jwongmeng

Analysis script for CSX

"""

#%% Initialise

import numpy as np
import xarray as xr
from scipy.ndimage import label
import datetime
import os
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
nxi = 1440
month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


#%% USER INPUT

# List of CSX
csxlist = ["cmhw", "chix"]
#csxlist = ["chix"]
ncsx = np.shape(csxlist)[0]

# What baseline?
#baseline = 'fixed'
#baseline = 'fixed2019'
baseline = 'moving'

# Which climatology method?
#climt = "mean"
climt = "median"

# What percentile for detection?
percentile = 95

# Minimum number of grid cell for extreme
mdep = 50

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
    #demod = 'poly2'
    demod = 'poly1'
    mdir = 'sea'
    
# Generate tvname
tvname = thvar + "_" + demod + "_" + str(percentile)

# Root directories
det_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/detect/' + tvname + '/'
stats_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/stats/' + tvname + '/' + climt + "/"
idx_dir = '/nfs/' + mdir + '/work/jwongmeng/ROMS/analysis/intensity_index/' + tvname + '/' + climt + "/"

# List of GSX
gsxlist = []
for iex in range(0,ncsx):
    gsxlist = gsxlist + [csxlist[iex][1:]]
    
det_gsx_dir = []
det_csx_dir = []
idx_gsx_dir = []
sdir_csx = []

for iex in range(0,ncsx):
    
    # Input GSX detect directories
    det_gsx_dir = det_gsx_dir + [det_dir + gsxlist[iex] + "/"]
    
    # Input GSX index directories
    idx_gsx_dir = idx_gsx_dir + [idx_dir + gsxlist[iex] + "/"]    
    
    # Input detect directories for CSX
    det_csx_dir = det_csx_dir + [det_dir + csxlist[iex] + "/" + "md" + str(mdep) + "/"]

    # Output CSX Stats directory
    sdir_csx = sdir_csx + [stats_dir + csxlist[iex] + "/" + "md" + str(mdep) + "/"]

for ifol in range(0,ncsx):
    if not os.path.exists(sdir_csx[ifol]):
        os.makedirs(sdir_csx[ifol])

# Load coords
time_daily_1979 = np.load('/home/jwongmeng/work/ROMS/scripts/coords/time_daily_1979.npy', allow_pickle=True)
time_daily_1980 = np.load('/home/jwongmeng/work/ROMS/scripts/coords/time_daily_1980.npy', allow_pickle=True)
lat_rho_short = np.load('/home/jwongmeng/work/ROMS/scripts/coords/lat_rho_short.npy') 
lon_rho_short = np.load('/home/jwongmeng/work/ROMS/scripts/coords/lon_rho_short.npy') 
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')[0:nz]
delz = np.load('/home/jwongmeng/work/ROMS/scripts/coords/delz.npy')[0:nz]
vol = xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc').volume[0:nz,:,1:-1].values

# Calculate total column volume in specified depth range
cvol = np.nansum(vol, axis=0)


#%% Start loop for CSXs

for icsx in range(ncsx):
    
    csx = csxlist[icsx]
    
    
    #%% no. of days statistics for CSXs (required for CCX CP)
    # Mean extreme days per year for individual extremes within column (medy)
    # Annual volume fraction of extremes (avfe)
    # Monthly volume fraction of extremes (mvfe)
    
    print("Calculating no. of days statistics for CSXs")

    medy = np.zeros((nyears,neta,nxi), dtype=np.float32)
    avfe = np.zeros((nyears,neta,nxi), dtype=np.float32)
    mvfe = np.zeros((nyears,nmonths,neta,nxi), dtype=np.float32)

    # Load CSX det
    da_det_csx = xr.open_dataset(det_csx_dir[icsx] + "det_all_morphed.nc").detect[-40:,:,:,1:-1]
    
    for ieta in range(0,neta):
    
        # Load CSX det
        det_csx = da_det_csx[:,:,ieta,:].values
        
        # Calculate medy
        medy[:,ieta,:] = (np.nansum(det_csx, axis=1))     
        
        # Now begin calculation of avfe
        # Load GSX detect
        det_gsx = xr.open_dataset(det_gsx_dir[icsx] + "/det_" + str(ieta) + ".nc").detect[-40:,:,0:nz,1:-1]
        det_gsx = det_gsx.assign_coords(day=("days",time_daily_1979))
        
        # Calculate extreme volume of each extreme cell
        ivol = np.multiply(det_gsx,vol[:,ieta,:])
        
        # Sum up in columns (results in [years,days,xi_rho])
        icvol = ivol.sum(dim='z_rho',skipna=True)
        
        # For avfe, divide by total column volume and take mean over days
        avfe[:,ieta,:] = np.nanmean(np.divide(icvol, cvol[ieta,:]), axis=1)
        
        # For mvfe, take mean over months and divide by total column
        mvfe[:,:,ieta,:] = np.divide(icvol.groupby("day.month").mean(),cvol[ieta,:])

    # Write to NCDF
    ds_medy = xr.Dataset(
        data_vars = dict(
            medy = (["year","eta_rho","xi_rho"], medy),
            ),
        coords = dict(
            year = (["year"],years),
            lon_rho = (["eta_rho","xi_rho"],lon_rho_short),
            lat_rho = (["eta_rho","xi_rho"],lat_rho_short)
            ),
        attrs = dict(description = "CSX days per year")
        )
    
    ds_medy.to_netcdf(sdir_csx[icsx] + "medy.nc",'w')
    del medy, ds_medy
    
    ds_avfe = xr.Dataset(
        data_vars = dict(
            avfe = (["year","eta_rho","xi_rho"], avfe),
            ),
        coords = dict(
            year = (["year"],years),
            lon_rho = (["eta_rho","xi_rho"],lon_rho_short),
            lat_rho = (["eta_rho","xi_rho"],lat_rho_short)
            ),
        attrs = dict(description = "Annual volume fraction")
        )
        
    ds_avfe.to_netcdf(sdir_csx[icsx] + "avfe.nc",'w')
    del avfe, ds_avfe  
    
    ds_mvfe = xr.Dataset(
        data_vars = dict(
            mvfe = (["year","month","eta_rho","xi_rho"], mvfe),
            ),
        coords = dict(
            year = (["year"],years),
            month = (["month"],months),
            lon_rho = (["eta_rho","xi_rho"],lon_rho_short),
            lat_rho = (["eta_rho","xi_rho"],lat_rho_short)
            ),
        attrs = dict(description = "Monthly volume fraction")
        )
        
    ds_mvfe.to_netcdf(sdir_csx[icsx] + "mvfe.nc",'w')
    del mvfe, ds_mvfe  
    
    print("Completed no. of days statistics for CSX " + csxlist[icsx])
    
    #%% Mean duration of column events (medr)
    
    print("Calculating medr statistics")
    
    medr = np.zeros((nyears,neta,nxi), dtype=np.float32) 
    nfeat = np.zeros((neta,nxi), dtype=int)  

    # Load CSX det
    da_det_csx = xr.open_dataset(det_csx_dir[icsx] + "det_all_morphed.nc").detect[1:,:,:,1:-1]
    
    for ieta in range(0,neta):
        
        # Open CSX det file
        det_csx = da_det_csx[:,:,ieta,:].values
        
        # Reshape to stack years after one another (nyears*ndays, nxi)
        det_csx = np.reshape(det_csx, (nyears*ndays,nxi), order='C')
        
        for ixi in range(0,nxi):
            
            # Define year_arr of zeros to count events in each year for this position
            year_arr = np.zeros((nyears), dtype=np.float32)
            
            # Label in time
            lab1d, nfeat1d = label(det_csx[:,ixi])
            
            for ifeat in range(0,nfeat1d):
                ind = np.where(lab1d == ifeat+1)
                
                # Duration statistics
                first = ind[0][0] # find day of first occurrence of ifeat-th event
                last =  ind[0][-1] # find day of last occurrence of ifeat-th event
                
                # Find years of start and end
                syear = int(np.ceil(first/365)) - 1
                eyear = int(np.ceil(last/365)) - 1
                
                # Add event duration to every year within [syear,eyear], and add event count +1
                for iyear in range(syear,eyear+1):
                    medr[iyear,ieta,ixi] = medr[iyear,ieta,ixi] + (last - first + 1)
                    year_arr[iyear] = year_arr[iyear] + 1.0
                
            medr[:,ieta,ixi] = np.divide(medr[:,ieta,ixi],year_arr)
            nfeat[ieta,ixi] = nfeat1d
            
        print(str(ieta) + ' complete')
    
    print('max features = ' + str(np.nanmax(nfeat)))
                
    # Write to NCDF
    ds_medr = xr.Dataset(
        data_vars = dict(
            medr_full = (["year","eta_rho","xi_rho"], medr),
            nfeat= (["eta_rho","xi_rho"], nfeat),
            ),
        coords = dict(
            year = (["year"],years),
            lon_rho = (["eta_rho","xi_rho"],lon_rho_short),
            lat_rho = (["eta_rho","xi_rho"],lat_rho_short)
            ),
        attrs = dict(description = "Mean duration of column events")
        )
    
    ds_medr.to_netcdf(sdir_csx[icsx] + "medr.nc",'w')
    del medr, ds_medr    
    
    print("Completed medr statistics")    
    