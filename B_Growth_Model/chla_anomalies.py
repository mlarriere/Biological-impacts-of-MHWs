"""
Created on Mon 16 Feb 16:19:45 2025

Anomalies in Chla 

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES========================
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob
import collections

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm

from datetime import date, timedelta
import time
from tqdm.contrib.concurrent import process_map

from joblib import Parallel, delayed

#%% ======================== Server ======================== 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% ======================== Figure settings ========================
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif':['Times'],
    "font.size": 9,           
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,   
    "text.latex.preamble": r"\usepackage{mathptmx}",  # to match your Overleaf font
})
# %% ======================== SETTINGS ========================
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_duration = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/mhw_durations'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_det_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer'
path_combined_thesh= '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/combined_thresholds'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model'
path_growth_inputs_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs/austral_summer'
path_biomass = '/nfs/sea/work/mlarriere/mhw_krill_SO/biomass'
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_biomass_ts = os.path.join(path_surrogates, f'biomass_timeseries')
path_biomass_ts_SO = os.path.join(path_biomass_ts, f'SouthernOcean')
path_biomass_ts_MPAs = os.path.join(path_biomass_ts, f'mpas')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')

# %% ======================== Load data ========================
# ==== Actual Conditions
chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended.nc')) #shape (40, 365, 231, 1442)

# ==== Climatological Drivers
chla_30yrs = chla_surf_SO_allyrs.isel(year=slice(0,30))
# chla_clim_mean = chla_clim.mean(dim=['year'])
# chla_clim_expanded = chla_clim_mean.expand_dims({'years': chla_surf_SO_allyrs.years})

# %% ======================== Chla anomalies as Hobday et al. ========================
import warnings

def moving_window_chla(ieta):
    # ieta=200
    
    ds_chla = chla_30yrs.raw_chla.isel(eta_rho=ieta)  # shape (30, 365, 1442)

    
    # Initisalisation
    _, ndays, nxi = ds_chla.shape
    chla_p90_moving = np.full((ndays, nxi), np.nan, dtype=np.float32) #shape (365, 1442)
    chla_climatology = np.full((ndays, nxi), np.nan, dtype=np.float32) #shape (365, 1442)

    mask_nanvalues = np.where(np.isnan(ds_chla.mean(dim='year').values), False, True)  # shape: (365, 1442)

    # Moving window of 11 days - method from Hobday et al. (2016) 
    for dy in range(0, ndays): 
        if dy <= 4:
            window11d_chla = ds_chla.isel(day=np.concatenate([np.arange(360 + dy, 365, 1), np.arange(0, dy + 6, 1)]))
        elif dy >= 360:
            window11d_chla = ds_chla.isel(day=np.concatenate([np.arange(dy - 5, 365, 1), np.arange(0, dy - 359, 1)]))
        else:
            window11d_chla = ds_chla.isel(day=np.arange(dy - 5, dy + 6, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # suppress all-NaN slice warnings (land points)
            # Calculate Chla climatology and threshold
            chla_p90_moving[dy, :] = np.nanpercentile(window11d_chla, 90, axis=(0, 1))
            chla_climatology[dy, :] = np.nanmedian(window11d_chla, axis=(0, 1))
    
    # Apply mask
    chla_p90_moving = np.where(mask_nanvalues, chla_p90_moving, np.nan) #shape (365, 1442)
    chla_climatology = np.where(mask_nanvalues, chla_climatology, np.nan) #shape (365, 1442)

    return chla_p90_moving, chla_climatology

output_file_clim = os.path.join(path_growth_inputs, f"chla_anomalies/chla_clim.nc")
output_file_90p = os.path.join(path_growth_inputs, f"chla_anomalies/chla_90thperc.nc")

ndays = 365  
nyears_clim = 30  # climatology period
chla_clim_data = chla_30yrs.raw_chla  


if not os.path.exists(output_file_clim) and not os.path.exists(output_file_90p):
    # Run in parallel
    results = process_map(moving_window_chla, range(chla_30yrs.raw_chla.shape[2]), max_workers=20, chunksize=1, desc=f"ieta")

    # Put together
    ds_rel_threshold, ds_clim = zip(*results)

    # Datasets initialization 
    chla_clim = np.full((ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32) #shape (365, 231, 1442)
    chla_90perc =  np.full((ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32)  #shape (365, 231, 1442)

    # Loop over neta and write all eta in same Dataset - aggregation 
    for ieta in range(0, 231):
        chla_clim[:, ieta, :] = ds_clim[ieta]  
        chla_90perc[:, ieta, :] = ds_rel_threshold[ieta]

    # To Dataset
    chla_90perc_ds = xr.Dataset(data_vars = dict(chla_p90 =(["day", "eta_rho", "xi_rho"], chla_90perc)),
                                    coords={'day': chla_surf_SO_allyrs.day,
                                            'lat_rho': chla_surf_SO_allyrs.lat_rho,
                                            'lon_rho': chla_surf_SO_allyrs.lon_rho},
                                    attrs = {'chla_p90': 'Daily climatological relative threshold (90th percentile) - computed using a seasonally varying 11‐day moving window '})

    chla_clim_ds = xr.Dataset(data_vars = dict(chla_clim =(["day", "eta_rho", "xi_rho"], chla_clim)),
                                coords={'day': chla_surf_SO_allyrs.day,
                                        'lat_rho': chla_surf_SO_allyrs.lat_rho,
                                        'lon_rho': chla_surf_SO_allyrs.lon_rho},
                                    attrs = {'climatology': 'Daily chla climatology - median value obtained from a seasonally varying 11‐day moving window - baseline 1980-2009 (30yrs)'})
    chla_clim_ds = chla_clim_ds.rename({'day': 'days'})
    chla_90perc_ds = chla_90perc_ds.rename({'day': 'days'})

    # Save output
    # chla_clim_ds.to_netcdf(output_file_clim, mode='w')  
    # chla_90perc_ds.to_netcdf(output_file_90p, mode='w')  

# else:
#     # Load data
#     chla_90perc_ds = xr.open_dataset(output_file_90p)
    # chla_clim_ds = xr.open_dataset(output_file_clim)

    
# %% ======================== Visualition ========================
import matplotlib.dates as mdates

# --- Select a single point ---
ieta, ixi = 220, 1000 

# --- Prepare data
y_start, y_end = 30, 35
chla_raw_point = chla_surf_SO_allyrs.raw_chla.isel(eta_rho=ieta, xi_rho=ixi, year=slice(y_start, y_end))
chla_raw_flat = chla_raw_point.values.flatten()

# --- Climatology and 90th percentile (repeat x time to cover time period) ---
chla_clim_point = chla_clim_ds.chla_clim.isel(eta_rho=ieta, xi_rho=ixi).values # shape (365,)
chla_p90_point = chla_90perc_ds.chla_p90.isel(eta_rho=ieta, xi_rho=ixi).values # shape (365,)
chla_clim_tiled = np.tile(chla_clim_point, (y_end-y_start))   # shape: (1095,)
chla_p90_tiled = np.tile(chla_p90_point, (y_end-y_start))    # shape: (1095,)

# --- Time axis ---
time_axis = [date(y_start + 1980, 1, 1) + timedelta(days=i) for i in range(365 * (y_end - y_start))]

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(time_axis, chla_raw_flat,  color='green',  lw=1,   alpha=0.8, label='Chla (raw)')
ax.plot(time_axis, chla_clim_tiled, color='black',  lw=1.5, linestyle='--', label='Climatology (median)')
ax.plot(time_axis, chla_p90_tiled,  color='red',    lw=1.5, linestyle='--', label='90th percentile')

# Shade area above 90th percentile (MHB-like events)
ax.fill_between(time_axis, chla_p90_tiled, chla_raw_flat,
                where=(chla_raw_flat > chla_p90_tiled),
                color='green', alpha=0.3, label='Above 90th perc.')

ax.set_ylabel('Chla (mg m$^{-3}$)')
ax.set_title(f'Chla time series — eta={ieta}, xi={ixi} | {y_start+1980} - {y_end+1980}')
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.set_xlim(time_axis[0], time_axis[-1])
ax.set_ylim(bottom=0)
plt.xticks(rotation=45)
ax.legend(frameon=True, loc='upper left')
plt.tight_layout()
plt.show()

# %% ======================== Anomalies ========================
def detect_chla_anom(ieta):
    # test
    # ieta=100

    # -- Load data
    ds = chla_surf_SO_allyrs.raw_chla.isel(eta_rho=ieta)
    climatology = xr.open_dataset(output_file_clim)['chla_clim'].isel(eta_rho=ieta)
    relative_threshold = xr.open_dataset(output_file_90p)['chla_p90'].isel(eta_rho=ieta)
    
    # -- Land mask: True where ocean, False where land (NaN in climatology)
    # land_mask = np.isnan(climatology.values)  # shape (365, 1442) - broadcast over years

    # -- Initisalisation
    nyears, ndays, nxi = ds.shape  # shape (40, 365, 1442)
    chla_anomalies = np.full((nyears, ndays, nxi), np.nan, dtype=bool) #(40, 365, 1442)   
    
    # -- Anomalies detections
    chla_anomalies[:,:,:] = np.greater(ds.values, relative_threshold.values) #Boolean
    # chla_anomalies[:, land_mask] = np.nan  # restore NaN on land points

    # -- Anomalies intensity 
    chla_intensity = ds.values - climatology.values # Anomaly relative to climatology
    
    return chla_anomalies, chla_intensity



output_file_anomalies = os.path.join(path_growth_inputs, f"chla_anomalies/chla_anomalies.nc")
output_file_intensity = os.path.join(path_growth_inputs, f"chla_anomalies/chla_intensity.nc")

if not os.path.exists(output_file_anomalies) and not os.path.exists(output_file_intensity):
    # Run in parallel
    results = process_map(detect_chla_anom, range(chla_30yrs.raw_chla.shape[2]), max_workers=20, chunksize=1, desc=f"ieta")

    # Put together
    chla_anomalies, chla_intensity = zip(*results)

    # Datasets initialization 
    nyears=40
    chla_anomalies_arr = np.full((nyears, ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32) #shape (40, 365, 231, 1442)
    chla_intensity_arr =  np.full((nyears, ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32)  #shape (40, 365, 231, 1442)

    # Loop over neta and write all eta in same Dataset
    for ieta in range(0, 231):
        chla_anomalies_arr[:, :, ieta, :] = chla_anomalies[ieta]  
        chla_intensity_arr[:, :, ieta, :] = chla_intensity[ieta]

    # To Dataset
    chla_anomalies_ds = xr.Dataset(data_vars = dict(chla_anom =(["years", "day", "eta_rho", "xi_rho"], chla_anomalies_arr)),
                                    coords={'day': chla_surf_SO_allyrs.day,
                                            'lat_rho': chla_surf_SO_allyrs.lat_rho,
                                            'lon_rho': chla_surf_SO_allyrs.lon_rho},
                                    attrs = {'chla_anom': 'Chla anomalies, defined as exceeding the 90th percentile (30years baseline - same method as for MHWs).'})

    chla_intensity_ds = xr.Dataset(data_vars = dict(chla_intens =(["years", "day", "eta_rho", "xi_rho"], chla_intensity_arr)),
                                coords={'day': chla_surf_SO_allyrs.day,
                                        'lat_rho': chla_surf_SO_allyrs.lat_rho,
                                        'lon_rho': chla_surf_SO_allyrs.lon_rho},
                                    attrs = {'chla_intens': 'Intensities of the chla anomalies, defined as exceeding the 90th percentile (30years baseline - same method as for MHWs).'})
    chla_anomalies_ds = chla_anomalies_ds.rename({'day': 'days'})
    chla_intensity_ds = chla_intensity_ds.rename({'day': 'days'})

    
    # Save output
    # chla_anomalies_ds.to_netcdf(output_file_anomalies, mode='w')  
    # chla_intensity_ds.to_netcdf(output_file_intensity, mode='w')  

# else:
#     # Load data
#     chla_anomalies_ds = xr.open_dataset(output_file_anomalies)
#     chla_intensity_ds = xr.open_dataset(output_file_intensity)


# %%  ======================== Quality checks ========================
yr_check = slice(0, 2)  # 1980-1981

anom_vals  = chla_anomalies_ds.chla_anom.isel(years=yr_check).values
intens_vals = chla_intensity_ds.chla_intens.isel(years=yr_check).values

print("=" * 60)
print("QUALITY CHECKS — Chla Anomalies & Intensity (1980-1981)")
print("=" * 60)

# --- 1. Basic structure
print("\n[1] Subset shapes")
print(f"  chla_anomalies : {anom_vals.shape}")   # should be (2, 365, 231, 1442)
print(f"  chla_intensity : {intens_vals.shape}")  # should be (2, 365, 231, 1442)

# --- 2. Value range checks
print("\n[2] Value ranges")
print(f"  Anomalies  — unique values      : {np.unique(anom_vals[~np.isnan(anom_vals)])}")  # should be [0. 1.]
print(f"  Anomalies  — % flagged as True  : {np.nanmean(anom_vals)*100:.2f}%")              # expected ~10%
print(f"  Intensity  — min  : {np.nanmin(intens_vals):.4f}")
print(f"  Intensity  — max  : {np.nanmax(intens_vals):.4f}")
print(f"  Intensity  — mean : {np.nanmean(intens_vals):.4f}")

# --- 3. NaN consistency
print("\n[3] NaN consistency (land mask)")
nan_mismatch = np.sum(np.isnan(anom_vals) != np.isnan(intens_vals))
print(f"  NaN mismatch between anomalies and intensity: {nan_mismatch} cells")  # should be 0

# --- 4. Intensity sign consistency
print("\n[4] Intensity sign consistency")
ocean_mask  = ~np.isnan(anom_vals)
anom_true   = anom_vals[ocean_mask].astype(bool)
intens_flat = intens_vals[ocean_mask]
print(f"  Intensity > 0 when anomaly=True  : {np.mean(intens_flat[anom_true] > 0)*100:.1f}%")   # expected ~100%
print(f"  Intensity < 0 when anomaly=False : {np.mean(intens_flat[~anom_true] < 0)*100:.1f}%")  # expected ~100%

# --- 5. Visual checks
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 5a. Daily anomaly frequency
anom_freq_per_day = np.nanmean(anom_vals, axis=(0, 2, 3))  # shape (365,)
axes[0].plot(anom_freq_per_day * 100, color='green', lw=1.5)
axes[0].axhline(10, color='red', linestyle='--', lw=1, label='Expected 10%')
axes[0].set_ylabel('Anomaly frequency (%)')
axes[0].set_xlabel('Day of year')
axes[0].set_title('Daily anomaly frequency (1980-1981)')
axes[0].legend(frameon=False)

# 5b. Annual anomaly frequency
anom_freq_per_year = np.nanmean(anom_vals, axis=(1, 2, 3))  # shape (2,)
axes[1].bar(np.arange(1980, 1982), anom_freq_per_year * 100, color='green', alpha=0.7)
axes[1].axhline(10, color='red', linestyle='--', lw=1, label='Expected 10%')
axes[1].set_ylabel('Anomaly frequency (%)')
axes[1].set_xlabel('Year')
axes[1].set_title('Annual anomaly frequency')
axes[1].legend(frameon=False)

# 5c. Single point time series
ieta, ixi   = 100, 700
chla_raw    = chla_surf_SO_allyrs.raw_chla.isel(eta_rho=ieta, xi_rho=ixi, year=yr_check).values.flatten()
chla_clim_p = xr.open_dataset(output_file_clim)['chla_clim'].isel(eta_rho=ieta, xi_rho=ixi).values
chla_p90_p  = xr.open_dataset(output_file_90p)['chla_p90'].isel(eta_rho=ieta, xi_rho=ixi).values
chla_anom_p = chla_anomalies_ds.chla_anom.isel(eta_rho=ieta, xi_rho=ixi, years=yr_check).values.flatten()

time_axis = [date(1980 + y, 1, 1) + timedelta(days=d) for y in range(2) for d in range(365)]

axes[2].plot(time_axis, chla_raw,                    color='green', lw=1,   alpha=0.8, label='Chla raw')
axes[2].plot(time_axis, np.tile(chla_clim_p, 2),     color='black', lw=1.5, linestyle='--', label='Climatology')
axes[2].plot(time_axis, np.tile(chla_p90_p,  2),     color='red',   lw=1.5, linestyle='--', label='90th perc.')
axes[2].fill_between(time_axis, 0, chla_raw,
                     where=chla_anom_p.astype(bool),
                     color='green', alpha=0.3, label='Anomaly detected')
axes[2].set_ylabel('Chla (mg m$^{-3}$)')
axes[2].set_title(f'Single point — eta={ieta}, xi={ixi} | 1980–1981')
axes[2].legend(frameon=False)

plt.tight_layout()
plt.show()

# %% ======================== Visualisation ========================
from matplotlib.colors import ListedColormap

iday  = 100
iyear = 36

fig = plt.figure(figsize=(13,5))
gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.05, hspace=0.3)

# --- Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Plot 1: Anomalies (binary 0/1)
data1 = chla_anomalies_ds.chla_anom.isel(years=iyear, days=iday)
ax1 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
ax1.set_boundary(circle, transform=ax1.transAxes)  # ax1 not ax
ax1.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax1.coastlines(color='black', linewidth=0.7, zorder=3)

cmap_anom = ListedColormap(['white', 'green'])

im1 = ax1.pcolormesh(data1.lon_rho, data1.lat_rho, data1,
                     cmap=cmap_anom, vmin=0, vmax=1,
                     transform=ccrs.PlateCarree(), zorder=1)
gl = ax1.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
gl.xlabels_top  = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 8, 'rotation': 0}
gl.ylabel_style = {'size': 8, 'rotation': 0}
ax1.set_title('Chla Anomalies', fontsize=16)

# --- Plot 2: Intensity
data2 = chla_intensity_ds.chla_intens.isel(years=iyear, days=iday)
ax2 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
ax2.set_boundary(circle, transform=ax2.transAxes)  # ax2 not ax
ax2.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax2.coastlines(color='black', linewidth=0.7, zorder=3)

im2 = ax2.pcolormesh(data2.lon_rho, data2.lat_rho, data2,  # assign to im2, use data2 not data2.chla_anom
                     cmap='Reds', vmin=0, vmax=1,
                     transform=ccrs.PlateCarree(), zorder=1)

gl = ax2.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
gl.xlabels_top  = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 8, 'rotation': 0}
gl.ylabel_style = {'size': 8, 'rotation': 0}
ax2.set_title('Chla Intensity of anomalies', fontsize=16)

# --- Colorbars
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='white', edgecolor='gray', label='No anomaly'),
                   Patch(facecolor='green', label='Anomaly')]
ax1.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, -0.05),  # centered just below the map
           fontsize=9, frameon=True, ncol=2)

cbar_ax2 = fig.add_axes([0.92, 0.25, 0.01, 0.55])
plt.colorbar(im2, cax=cbar_ax2, orientation='vertical', extend='both').set_label("Chla [mg m$^{-3}$]", fontsize=12)

# --- Overall title
plt.suptitle(f"Chla anomalies — day {iday}, year {iyear + 1980}", fontsize=18, y=1.04, x=0.52)
plt.show()

# %%
