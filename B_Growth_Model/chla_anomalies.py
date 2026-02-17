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

# %% ======================== Chla anomalies ========================
import warnings

def moving_window_chla(ieta):
    # ieta=200
    
    # -- Load data
    ds_chla = chla_30yrs.raw_chla.isel(eta_rho=ieta)  # shape (30, 365, 1442)

    
    # Initisalisation
    _, ndays, nxi = ds_chla.shape
    relative_threshold_p90 = np.full((ndays, nxi), np.nan, dtype=np.float32) #shape (365, 1442)
    relative_threshold_p10 = np.full((ndays, nxi), np.nan, dtype=np.float32) #shape (365, 1442)
    climatology = np.full((ndays, nxi), np.nan, dtype=np.float32) #shape (365, 1442)

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
            relative_threshold_p90[dy, :] = np.nanpercentile(window11d_chla, 90, axis=(0, 1))
            relative_threshold_p10[dy, :] = np.nanpercentile(window11d_chla, 10, axis=(0, 1))
            climatology[dy, :] = np.nanmedian(window11d_chla, axis=(0, 1))
    
    # Apply mask
    relative_threshold_p90 = np.where(mask_nanvalues, relative_threshold_p90, np.nan) #shape (365, 1442)
    relative_threshold_p10 = np.where(mask_nanvalues, relative_threshold_p10, np.nan) #shape (365, 1442)
    climatology = np.where(mask_nanvalues, climatology, np.nan) #shape (365, 1442)

    return relative_threshold_p90, relative_threshold_p10, climatology

output_file_clim = os.path.join(path_growth_inputs, f"chla_anomalies/chla_clim.nc")
output_file_rel = os.path.join(path_growth_inputs, f"chla_anomalies/chla_relative_thresholds.nc")

ndays = 365  
nyears_clim = 30  # climatology period
chla_clim_data = chla_30yrs.raw_chla  


if not os.path.exists(output_file_clim) and not os.path.exists(output_file_rel):
    # Run in parallel
    results = process_map(moving_window_chla, range(chla_30yrs.raw_chla.shape[2]), max_workers=20, chunksize=1, desc=f"ieta")

    # Put together
    rel_threshold_90p, rel_threshold_10p, clim = zip(*results)

    # Datasets initialization 
    chla_clim = np.full((ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32) #shape (365, 231, 1442)
    chla_90perc =  np.full((ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32)  #shape (365, 231, 1442)
    chla_10perc =  np.full((ndays, chla_30yrs.raw_chla.shape[2], chla_30yrs.raw_chla.shape[3]), np.nan, dtype=np.float32)  #shape (365, 231, 1442)

    # Loop over neta and write all eta in same Dataset - aggregation 
    for ieta in range(0, 231):
        chla_clim[:, ieta, :] = clim[ieta]  
        chla_90perc[:, ieta, :] = rel_threshold_90p[ieta]
        chla_10perc[:, ieta, :] = rel_threshold_10p[ieta]

    # To Dataset
    chla_relative_threshold_ds = xr.Dataset(
        data_vars=dict(rel_p90=(['day', 'eta_rho', 'xi_rho'], chla_90perc),
                       rel_p10=(['day', 'eta_rho', 'xi_rho'], chla_10perc)),
        coords={'day': chla_surf_SO_allyrs.day,
                'lat_rho': chla_surf_SO_allyrs.lat_rho,
                'lon_rho': chla_surf_SO_allyrs.lon_rho,},
        attrs={'rel_p90': 'Daily climatological relative threshold (90th percentile) - computed using a seasonally varying 11-day moving window',
               'rel_p10': 'Daily climatological relative threshold (10th percentile) - computed using a seasonally varying 11-day moving window'})
    
    chla_clim_ds = xr.Dataset(
        data_vars = dict(chla_clim =(["day", "eta_rho", "xi_rho"], chla_clim)),
        coords={'day': chla_surf_SO_allyrs.day,
                'lat_rho': chla_surf_SO_allyrs.lat_rho,
                'lon_rho': chla_surf_SO_allyrs.lon_rho},
        attrs = {'climatology': 'Daily chla climatology - median value obtained from a seasonally varying 11‐day moving window - baseline 1980-2009 (30yrs)'})
    chla_clim_ds = chla_clim_ds.rename({'day': 'days'})
    chla_relative_threshold_ds = chla_relative_threshold_ds.rename({'day': 'days'})

    # Save output
    chla_clim_ds.to_netcdf(output_file_clim, mode='w')  
    chla_relative_threshold_ds.to_netcdf(output_file_rel, mode='w')  

else:
    # Load data
    chla_relative_threshold_ds = xr.open_dataset(output_file_rel)
    chla_clim_ds = xr.open_dataset(output_file_clim)

    
# %% ======================== Visualition ========================
import matplotlib.dates as mdates

# --- Select a single point ---
ieta, ixi = 200, 1100 

# --- Prepare data
y_start, y_end = 36, 39
chla_raw_point = chla_surf_SO_allyrs.raw_chla.isel(eta_rho=ieta, xi_rho=ixi, year=slice(y_start, y_end))
chla_raw_flat = chla_raw_point.values.flatten()

# --- Climatology and 90th percentile (repeat x time to cover time period) ---
chla_clim_point = chla_clim_ds.chla_clim.isel(eta_rho=ieta, xi_rho=ixi).values # shape (365,)
chla_p90_point = chla_relative_threshold_ds.rel_p90.isel(eta_rho=ieta, xi_rho=ixi).values # shape (365,)
chla_p10_point = chla_relative_threshold_ds.rel_p10.isel(eta_rho=ieta, xi_rho=ixi).values # shape (365,)
chla_clim_tiled = np.tile(chla_clim_point, (y_end-y_start))
chla_p90_tiled = np.tile(chla_p90_point, (y_end-y_start)) 
chla_p10_tiled = np.tile(chla_p10_point, (y_end-y_start)) 

# --- Time axis ---
time_axis = [date(y_start + 1980, 1, 1) + timedelta(days=i) for i in range(365 * (y_end - y_start))]

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(time_axis, chla_raw_flat,  color='green',  lw=1,   alpha=0.8, label='Chla (raw)')
ax.plot(time_axis, chla_clim_tiled, color='black',  lw=1, linestyle='--', label='Climatology (median)')
ax.plot(time_axis, chla_p90_tiled,  color='red',    lw=1, linestyle='--', label='90th percentile')
ax.plot(time_axis, chla_p10_tiled,  color='orange',    lw=1, linestyle='--', label='10th percentile')

# Shade area above 90th percentile (MHB-like events)
ax.fill_between(time_axis, chla_p90_tiled, chla_raw_flat,
                where=(chla_raw_flat > chla_p90_tiled),
                color='red', alpha=0.3, label='Above 90th perc.')

ax.fill_between(time_axis, chla_p10_tiled, chla_raw_flat,
                where=(chla_raw_flat < chla_p10_tiled),
                color='orange', alpha=0.3, label='Below 10th perc.')

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
    relative_threshold_p90 = xr.open_dataset(output_file_rel)['rel_p90'].isel(eta_rho=ieta)
    relative_threshold_p10 = xr.open_dataset(output_file_rel)['rel_p10'].isel(eta_rho=ieta)

    # -- Combined NaN mask
    nan_mask = np.isnan(ds.values) | np.isnan(climatology.values)  # (40, 365, 1442)

    # -- Positive anomalies: above 90th percentile (bloom)
    chla_anom_pos = np.greater(ds.values, relative_threshold_p90.values).astype(np.float32)
    chla_anom_pos[nan_mask] = np.nan

    # -- Negative anomalies: below 10th percentile (food scarcity)
    chla_anom_neg = np.less(ds.values, relative_threshold_p10.values).astype(np.float32)
    chla_anom_neg[nan_mask] = np.nan

    # -- Intensity (relative to climatology median, sign preserved)
    chla_intensity = ds.values - climatology.values  # positive=bloom, negative=scarcity

    return chla_anom_pos, chla_anom_neg, chla_intensity


output_file_anomalies = os.path.join(path_growth_inputs, f"chla_anomalies/chla_anomalies.nc")
output_file_intensity = os.path.join(path_growth_inputs, f"chla_anomalies/chla_intensity.nc")

if not os.path.exists(output_file_anomalies) and not os.path.exists(output_file_intensity):
    # Run in parallel
    results = process_map(detect_chla_anom, range(chla_30yrs.raw_chla.shape[2]), max_workers=20, chunksize=1, desc=f"ieta")
    chla_anom_pos, chla_anom_neg, chla_intensity = zip(*results)

    # Put together: Stack along eta_rho axis
    chla_anomalies_pos_arr = np.stack(chla_anom_pos, axis=2)   # (40, 365, 231, 1442)
    chla_anomalies_neg_arr = np.stack(chla_anom_neg, axis=2)   # (40, 365, 231, 1442)
    chla_intensity_arr     = np.stack(chla_intensity, axis=2)  # (40, 365, 231, 1442)

    # To Dataset
    chla_anomalies_ds = xr.Dataset(
        data_vars = dict(pos_anom =(["years", "day", "eta_rho", "xi_rho"], chla_anomalies_pos_arr),
                         neg_anom = (["years", "day", "eta_rho", "xi_rho"], chla_anomalies_neg_arr)),
        coords={'day': chla_surf_SO_allyrs.day,
                'lat_rho': chla_surf_SO_allyrs.lat_rho,
                'lon_rho': chla_surf_SO_allyrs.lon_rho},
        attrs = {'pos_anom': 'Positive Chla anomalies, defined as exceeding the 90th percentile (30years baseline).',
                 'neg_anom': 'Nagative Chla anomalies, defined as being below the 10th percentile (30years baseline).'})

    chla_intensity_ds = xr.Dataset(data_vars = dict(chla_intens =(["years", "day", "eta_rho", "xi_rho"], chla_intensity_arr)),
                                coords={'day': chla_surf_SO_allyrs.day,
                                        'lat_rho': chla_surf_SO_allyrs.lat_rho,
                                        'lon_rho': chla_surf_SO_allyrs.lon_rho},
                                    attrs = {'chla_intens': 'Intensities of the chla anomalies, defined as exceeding the 90th percentile (30years baseline - same method as for MHWs).'})
    chla_anomalies_ds = chla_anomalies_ds.rename({'day': 'days'})
    chla_intensity_ds = chla_intensity_ds.rename({'day': 'days'})

    
    # Save output
    chla_anomalies_ds.to_netcdf(output_file_anomalies, mode='w')  
    chla_intensity_ds.to_netcdf(output_file_intensity, mode='w')  

else:
    # Load data
    chla_anomalies_ds = xr.open_dataset(output_file_anomalies)
    chla_intensity_ds = xr.open_dataset(output_file_intensity)


# %%  ======================== Quality checks ========================
yr_check = slice(0, 2)  # 1980-1981 only for speed

pos_anom   = chla_anomalies_ds.pos_anom.isel(years=yr_check).values
neg_anom   = chla_anomalies_ds.neg_anom.isel(years=yr_check).values
intens     = chla_intensity_ds.chla_intens.isel(years=yr_check).values

print("=" * 60)
print("QUALITY CHECKS — Chla Anomalies & Intensity (1980-1981)")
print("=" * 60)

# --- 1. Basic structure
print("\n[1] Shapes")
print(f"  pos_anom : {pos_anom.shape}")   # (2, 365, 231, 1442)
print(f"  neg_anom : {neg_anom.shape}")   # (2, 365, 231, 1442)
print(f"  intens   : {intens.shape}")     # (2, 365, 231, 1442)

# --- 2. Value ranges
print("\n[2] Value ranges")
print(f"  pos_anom unique values : {np.unique(pos_anom[~np.isnan(pos_anom)])}")  # [0. 1.]
print(f"  neg_anom unique values : {np.unique(neg_anom[~np.isnan(neg_anom)])}")  # [0. 1.]
print(f"  pos_anom flagged       : {np.nanmean(pos_anom)*100:.2f}%")             # 17.20%
print(f"  neg_anom flagged       : {np.nanmean(neg_anom)*100:.2f}%")             # 7.83%
print(f"  intensity min/max/mean : {np.nanmin(intens):.3f} / {np.nanmax(intens):.3f} / {np.nanmean(intens):.3f}") #-2.059 / 4.862 / 0.090

# --- 3. NaN consistency across all three arrays
print("\n[3] NaN consistency")
nan_pos    = np.isnan(pos_anom)
nan_neg    = np.isnan(neg_anom)
nan_intens = np.isnan(intens)
print(f"  pos_anom vs neg_anom mismatch : {np.sum(nan_pos != nan_neg)}")      # 0
print(f"  pos_anom vs intens mismatch   : {np.sum(nan_pos != nan_intens)}")   # 0

# --- 4. Mutual exclusivity: a point cannot be both pos and neg anomaly
print("\n[4] Mutual exclusivity (pos AND neg simultaneously)")
both = (pos_anom == 1) & (neg_anom == 1)
print(f"  Points flagged as both pos and neg : {np.sum(both)}")  # 0

# --- 5. Sign consistency with intensity
print("\n[5] Intensity sign consistency")
ocean       = ~nan_pos
pos_flag    = pos_anom[ocean].astype(bool)
neg_flag    = neg_anom[ocean].astype(bool)
intens_flat = intens[ocean]
print(f"  Intensity > 0 when pos_anom=True : {np.mean(intens_flat[pos_flag] > 0)*100:.1f}%")   # 100.0%
print(f"  Intensity < 0 when neg_anom=True : {np.mean(intens_flat[neg_flag] < 0)*100:.1f}%")   # 100.0%


# %% ======================== Visualisation ========================
# %% ======================== Visualisation ========================
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

iday  = 364
iyear = 38

fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.08, hspace=0.3)

# --- Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Plot 1: Both anomalies combined
data_pos = chla_anomalies_ds.pos_anom.isel(years=iyear, days=iday).values
data_neg = chla_anomalies_ds.neg_anom.isel(years=iyear, days=iday).values

# Combine: 0=no anomaly, 1=bloom (pos), 2=scarcity (neg)
combined = np.zeros_like(data_pos)
combined[data_pos == 1] = 1  # bloom
combined[data_neg == 1] = 2  # scarcity
combined[np.isnan(data_pos)] = np.nan

ax1 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
ax1.set_boundary(circle, transform=ax1.transAxes)
ax1.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax1.coastlines(color='black', linewidth=0.7, zorder=3)
im1 = ax1.pcolormesh(chla_anomalies_ds.lon_rho, chla_anomalies_ds.lat_rho, combined,
                     cmap=ListedColormap(['white', 'green', 'orange']), vmin=0, vmax=2,
                     transform=ccrs.PlateCarree(), zorder=1)
gl = ax1.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
gl.xlabels_top = False; gl.ylabels_right = False
gl.xlabel_style = {'size': 8, 'rotation': 0}; gl.ylabel_style = {'size': 8, 'rotation': 0}
ax1.set_title('Chla Anomalies', fontsize=14)
ax1.legend(handles=[Patch(facecolor='white',  edgecolor='gray', label='No anomaly'),
                    Patch(facecolor='green', label=r'Bloom ($>$p90)'),
                    Patch(facecolor='orange', label=r'Scarcity ($<$p10)')],
           loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=True, ncol=3)

# --- Plot 2: Positive anomalies intensity (bloom)
data_intens = chla_intensity_ds.chla_intens.isel(years=iyear, days=iday)
intens_bloom = np.where(data_pos == 1, data_intens.values, np.nan)  # only where bloom

ax2 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())
ax2.set_boundary(circle, transform=ax2.transAxes)
ax2.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax2.coastlines(color='black', linewidth=0.7, zorder=3)
im2 = ax2.pcolormesh(data_intens.lon_rho, data_intens.lat_rho, intens_bloom,
                     cmap='Greens', vmin=0, vmax=1,
                     transform=ccrs.PlateCarree(), zorder=1)
gl = ax2.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 8, 'rotation': 0}
gl.ylabel_style = {'size': 8, 'rotation': 0}
ax2.set_title(r'Bloom Intensity ($>90^{th}$ perc.)', fontsize=14)

cbar_ax2 = fig.add_axes([0.415, 0.05, 0.2, 0.02])
plt.colorbar(im2, cax=cbar_ax2, orientation='horizontal', extend='max').set_label("Chla anomaly [mg m$^{-3}$]", fontsize=10)

# --- Plot 3: Negative anomalies intensity (scarcity)
intens_scarcity = np.where(data_neg == 1, data_intens.values, np.nan)  # only where scarcity

ax3 = fig.add_subplot(gs[2], projection=ccrs.SouthPolarStereo())
ax3.set_boundary(circle, transform=ax3.transAxes)
ax3.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax3.coastlines(color='black', linewidth=0.7, zorder=3)
im3 = ax3.pcolormesh(data_intens.lon_rho, data_intens.lat_rho, intens_scarcity,
                     cmap='Oranges_r', vmin=-1, vmax=0,
                     transform=ccrs.PlateCarree(), zorder=1)
gl = ax3.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 8, 'rotation': 0}
gl.ylabel_style = {'size': 8, 'rotation': 0}
ax3.set_title(r'Scarcity Intensity ($<10^{th}$ perc.)', fontsize=14)
cbar_ax3 = fig.add_axes([0.68, 0.05, 0.2, 0.02])
plt.colorbar(im3, cax=cbar_ax3, orientation='horizontal', extend='min').set_label("Chla anomaly [mg m$^{-3}$]", fontsize=10)

# --- Overall title
plt.suptitle(f"Chla anomalies — day {iday}, year {iyear + 1980}", fontsize=16, y=1.02, x=0.48)
plt.show()

# %%
