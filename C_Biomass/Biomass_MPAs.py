"""
Created on Wed 10 Dec 15:14:30 2025

Biomass and biomass changes inside MPAs

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


from datetime import datetime, timedelta
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


# %% ======================== Defining MPAs ========================
# ---- Load data
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)

# ---- Fix extent 
# South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

# == Settings plot
mpa_dict = {"Ross Sea": (mpas_ds.mask_rs, "#5F0F40"),
            "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#FFBA08"),
            "East Antarctic": (mpas_ds.mask_ea, "#E36414"),
            "Weddell Sea": (mpas_ds.mask_ws, "#4F772D"),
            "Antarctic Peninsula": (mpas_ds.mask_ap, "#0A9396")}


# ---- Plot MPAs
fig = plt.figure(figsize=(5, 8))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)
ax.set_boundary(circle, transform=ax.transAxes)

# Base map
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Plot
lon = mpas_ds.lon_rho
lat = mpas_ds.lat_rho
for name, (mask, color) in mpa_dict.items():
    # Mask lon/lat
    lon_masked = lon.where(mask)
    lat_masked = lat.where(mask)
    ax.scatter(lon_masked.values, lat_masked.values,
               s=2, color=color, transform=ccrs.PlateCarree(), 
               label=name, alpha=0.8, zorder=1)

    # Add name of the MPA
    # Center of the box
    lon_centroid = float(lon_masked.mean().values)+10
    lat_centroid = float(lat_masked.mean().values)

    # Add text box with colored border
    ax.text(lon_centroid, lat_centroid, name, transform=ccrs.PlateCarree(), zorder=5, 
            fontsize=10, fontweight='bold', ha='center', va='center',
            bbox=dict( facecolor="white", edgecolor=color, linewidth=1.5, boxstyle="round,pad=0.3", alpha=0.9))

# Legend and title
ax.set_title("Southern Ocean MPAs", fontsize=12)

plt.tight_layout()
plt.show()

# %% ======================== Areas and volume MPAs ========================
# --- Load data
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km2
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc').isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)) #in km3

# --- Calculate total Southern Ocean area (south of 60°S)
# Select surface layer
area_SO_surf = area_roms['area'].isel(z_t=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# Mask latitudes south of 60°S (lat_rho <= -60)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# --- Calculate area and volume of each MPA
mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

area_mpa = {}
volume_mpa = {}

for abbrv, (name, mask) in mpa_masks.items():
    area_mpa[abbrv] = area_60S_SO.where(mask)
    volume_mpa[abbrv] = volume_60S_SO_100m.where(mask)
    volume_mpa[name] = volume_60S_SO_100m.where(mask)

# %% ======================== Biomass data ========================
path_biomass_ts_SO_regridd = os.path.join(path_biomass_ts_SO, 'biomass_regridded')
path_biomass_ts_SO_interp = os.path.join(path_biomass_ts_SO, 'biomass_interpolated')

# ==== Load data ====
# -- Southern Ocean (from Biomass_calculations.py)
biomass_clim = xr.open_dataset(os.path.join(path_biomass_ts_SO_regridd, "biomass_clim.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_clim_interp = xr.open_dataset(os.path.join(path_biomass_ts_SO_interp, "biomass_clim.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_actual = xr.open_dataset(os.path.join(path_biomass_ts_SO_regridd, "biomass_actual.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_actual_interp = xr.open_dataset(os.path.join(path_biomass_ts_SO_interp, "biomass_actual.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_noMHWs = xr.open_dataset(os.path.join(path_biomass_ts_SO_regridd, "biomass_nomhws.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_noMHWs_interp = xr.open_dataset(os.path.join(path_biomass_ts_SO_interp, "biomass_nomhws.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_clim_trended = xr.open_dataset(os.path.join(path_biomass_ts_SO_regridd, "biomass_clim_trend.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_clim_trended_interp = xr.open_dataset(os.path.join(path_biomass_ts_SO_interp, "biomass_clim_trend.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

biomass_nowarming = xr.open_dataset(os.path.join(path_biomass_ts_SO_regridd, "biomass_nowarming.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
biomass_nowarming_interp = xr.open_dataset(os.path.join(path_biomass_ts_SO_interp, "biomass_nowarming.nc")).isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))

# Put together in dictionnary
biomass_ds = {"clim": biomass_clim, "actual": biomass_actual, "nomhws": biomass_noMHWs, "climtrend": biomass_clim_trended, "nowarming": biomass_nowarming,}
biomass_ds_interp = {"clim": biomass_clim_interp, "actual": biomass_actual_interp, "nomhws": biomass_noMHWs_interp, "climtrend": biomass_clim_trended_interp, "nowarming": biomass_nowarming_interp,}

surrogate_names = {"clim": "Climatology", "actual": "Actual Conditions", "nomhws": "No Marine Heatwaves", "climtrend":"Climatology wih trend", "nowarming": "No Warming"}
mpa_names = list(mpa_dict.keys())
mpa_abbrs = list(mpa_masks.keys())

# %% ======================== Functions ========================
from tqdm.contrib.concurrent import process_map 

# --- Function to mask one surrogate for one MPA
def mask_one_surrogate(args):
    # Extract arguments
    surrog, ds, mpa_name, mpa_abbrv, mpa_mask, interp = args

    # test
    # surrog ='clim'
    # ds = biomass_ds['clim']
    # mpa_name = 'Ross Sea'
    # mpa_abbrv = 'RS'
    # mpa_mask = mpa_masks['RS'][1]
    # interp = False

    # 2cases, regridded only or with interpolation
    if interp:
        file_path = os.path.join(path_biomass_ts_MPAs_interp, f"{surrog}_biomass_{mpa_abbrv}.nc")
    else:
        file_path = os.path.join(path_biomass_ts_MPAs_regridd, f"{surrog}_biomass_{mpa_abbrv}.nc")

    # RUn only if file doesn't already exist
    if not os.path.exists(file_path):
        # Expand mask dimensions
        mask = mpa_mask
        if "years" in ds.dims:
            mask = mask.expand_dims({"years": ds.years.size}, axis=1)
        if "days" in ds.dims:
            mask = mask.expand_dims({"days": ds.days.size}, axis=2)
        
        # Mask biomass to MPAs
        biomass_masked = ds.biomass.where(mask)
        
        # To Dataset
        ds_mpa = xr.Dataset({"biomass": biomass_masked},
                            coords={"years": ds.years if "years" in ds.dims else None,
                                    "days": ds.days,
                                    "lon_rho": (("eta_rho", "xi_rho"), ds.lon_rho.data),
                                    "lat_rho": (("eta_rho", "xi_rho"), ds.lat_rho.data),},)
        
        ds_mpa.attrs.update({"mpa_name": mpa_name,
                             **{k: ds.attrs[k] for k in ["Description", "Units"] if k in ds.attrs}})

        # Save
        ds_mpa.to_netcdf(file_path)
        return f"Saved {mpa_name}"
    
    else:
        return 'MPAs already saved to file.'

# %% ==================================== Mask MPAs - Regridded ====================================
print('\nInitial biomass: Regridded')

path_biomass_ts_MPAs_regridd = os.path.join(path_biomass_ts_MPAs, 'biomass_regridded')
files_regrid = [os.path.join(path_biomass_ts_MPAs_regridd, f"{sur}_biomass_{abbrv}.nc") for sur in surrogate_names.keys() for abbrv in mpa_masks.keys()]

if not all(os.path.exists(f) for f in files_regrid):
    print('Masking biomass to MPAs and writing to file...')
    interp=False
    # --- Loop over MPAs
    for abbrv, (mpa_name, mpa_mask) in mpa_masks.items():
        # Test
        # abbrv='RS'
        # mpa_name='Ross Sea'
        # mpa_mask=mpa_masks['RS'][1]

        print(f"\nProcessing MPA: {mpa_name} ({abbrv})")

        # Prepare arguments for function
        args_list = [(surrog, ds, mpa_name, abbrv, mpa_mask, interp) for surrog, ds in biomass_ds.items()]
        
        # Run in parallel
        results = process_map(mask_one_surrogate, args_list, max_workers=4,  desc=f"{abbrv} | Mask ")

else:
    print('Loading files...')
    biomass_mpas = {}

    for abbrv, (mpa_name, _) in mpa_masks.items():
        biomass_mpas[abbrv] = {}

        for surrog in surrogate_names.keys():
            fname = os.path.join(path_biomass_ts_MPAs_regridd, f"{surrog}_biomass_{abbrv}.nc")
            biomass_mpas[abbrv][surrog] = xr.open_dataset(fname)

    
# %% ==================================== Mask MPAs - Regridded and interpolated ====================================
print('\nInitial biomass: Regridded and Interpolated')

path_biomass_ts_MPAs_interp = os.path.join(path_biomass_ts_MPAs, 'biomass_interpolated')
files_interp = [os.path.join(path_biomass_ts_MPAs_interp, f"{sur}_biomass_{abbrv}.nc") for sur in surrogate_names.keys() for abbrv in mpa_masks.keys()]

if not all(os.path.exists(f) for f in files_interp):
    print('Masking biomass to MPAs and writing to file...')
    interp=True
    # --- Loop over MPAs
    for abbrv, (mpa_name, mpa_mask) in mpa_masks.items():
        # Test
        # abbrv='RS'
        # mpa_name='Ross Sea'
        # mpa_mask=mpa_masks['RS'][1]

        print(f"\nProcesssing MPA: {mpa_name} ({abbrv})")

        # Prepare arguments for function
        args_list = [(surrog, ds, mpa_name, abbrv, mpa_mask, interp) for surrog, ds in biomass_ds_interp.items()]
        
        # Run in parallel
        results = process_map(mask_one_surrogate, args_list, max_workers=4,  desc=f"{abbrv} | Mask ")

else:
    print('Loading files...')
    biomass_mpas_interp = {}

    for abbrv, (mpa_name, _) in mpa_masks.items():
        biomass_mpas_interp[abbrv] = {}

        for surrog in surrogate_names.keys():
            fname = os.path.join(path_biomass_ts_MPAs_interp, f"{surrog}_biomass_{abbrv}.nc")
            biomass_mpas_interp[abbrv][surrog] = xr.open_dataset(fname)

# %% ================================= Plot Biomass concentration =================================    
# Year to plot
years_to_plot = 1989 #1989 2000 2016
year_idx = years_to_plot - 1980

# --- Figure setup
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 4, hspace=0.09, wspace=0.09)
proj = ccrs.SouthPolarStereo()

# --- Circular boundary
theta = np.linspace(0, 2*np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

titles = ["Climatology", "Actual (2016)", "No MHWs", "No Warming",
          "", "Actual − Clim", "No MHWs − Clim", "No Warming − Clim"]

# --- Color Setup
color_data=biomass_mpas_interp['AP']['actual'].biomass.isel(years=year_idx).median('algo').values
vmin_bio, vmax_bio = 5, 40 # np.nanpercentile(color_data, [5, 95])
cmap_bio = 'Reds'

diff_data= color_data - biomass_mpas_interp['AP']['clim'].biomass.median('algo')
max_abs_diff = np.nanmax(np.abs(diff_data))
vmin_diff, vmax_diff = -20, 20 #-max_abs_diff, max_abs_diff
cmap_diff = 'bwr'

# --- Prepare axis
axes = []
for i in range(8):
    ax = fig.add_subplot(gs[i], projection=proj)  

    if i == 4:  # row 2, first column → blank
        ax.axis("off")
        axes.append(ax)
        continue

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 7, 'rotation': 0}
    gl.ylabel_style = {'size': 7, 'rotation': 0}

    ax.set_title(titles[i], fontsize=12)
    axes.append(ax)

# --- Loop over MPAs
pcm_bio  = None
pcm_diff = None

for abbrv, mpa_data in biomass_mpas_interp.items():
    # Prepare data
    ds_clim = mpa_data["clim"]
    ds_actual = mpa_data["actual"]
    ds_clim_trend = mpa_data["climtrend"]
    ds_nowarm = mpa_data["nowarming"]

    # Row 1: Biomass at the end of the season
    bio_clim = ds_clim.biomass.isel(days=-1).median('algo')
    bio_actual = ds_actual.biomass.isel(years=year_idx, days=-1).median('algo')
    bio_clim_trend = ds_clim_trend.biomass.isel(years=year_idx, days=-1).median('algo')
    bio_nowarm = ds_nowarm.biomass.isel(years=year_idx, days=-1).median('algo')

    # Row 2: Difference w.r.t clim at the end of the season
    diff_actual  = bio_actual - bio_clim
    diff_clim_trend  = bio_clim_trend - bio_clim
    diff_nowarm  = bio_nowarm - bio_clim

    # Together
    data_to_plot = [bio_clim, bio_actual, bio_clim_trend, bio_nowarm,
                    None, diff_actual, bio_clim_trend, diff_nowarm]

    for i, data_i in enumerate(data_to_plot):
        if data_i is None:
                    continue
        
        # --- Row 1: Biomass
        if i < 4:
            pcm_bio = axes[i].pcolormesh(ds_clim.lon_rho, ds_clim.lat_rho, data_i, transform=ccrs.PlateCarree(),
                                cmap=cmap_bio, vmin=vmin_bio, vmax=vmax_bio, zorder=1)
            
        # --- Row 2: Difference
        else:
            pcm_diff = axes[i].pcolormesh(ds_clim.lon_rho, ds_clim.lat_rho, data_i, transform=ccrs.PlateCarree(),
                                cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff, zorder=1)

# --- Colorbars
cbar_ax1 = fig.add_axes([0.92, 0.58, 0.01, 0.27]) #(left, bottom, width, height)
plt.colorbar(pcm_bio, cax=cbar_ax1, orientation='vertical', extend='both').set_label("Biomass [mg.m$^{-3}$]", fontsize=12)

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.27]) #(left, bottom, width, height)
plt.colorbar(pcm_diff, cax=cbar_ax2, orientation='vertical', extend='both').set_label("Biomass [mg.m$^{-3}$]", fontsize=12)

fig.suptitle(f"Krill Biomass on 30th April ({years_to_plot}) in the MPAs", fontsize=14, x=0.52)
plt.show()


# %% ================================= Longest MHW per MPA and per threshold =================================
import pandas as pd

# ---- Select threshold ----
thresholds = {"1deg": "det_1deg", "2deg": "det_2deg", "3deg": "det_3deg", "4deg": "det_4deg",}

out_csv = os.path.join(path_combined_thesh, "mpas/longest_MHW_in_MPA_per_threshold.csv")
if not os.path.exists(out_csv):
    # Longest event per MPAs and treshold
    longest_mhw = {}

    for abbrv, (mpa_name, _) in mpa_masks.items():

        print(f"\nProcessing {mpa_name} ({abbrv})")

        biomass_data = biomass_mpas_interp[abbrv]["actual"].biomass_median
        mhw_ds = xr.open_dataset(os.path.join(path_combined_thesh, f"mpas/duration_AND_thresh_{abbrv}.nc"))

        longest_mhw[abbrv] = {"mpa_name": mpa_name}

        for label, thresh_var in thresholds.items():

            print(f"  Threshold: {label}")

            if thresh_var not in mhw_ds:
                print("    → threshold not found")
                longest_mhw[abbrv][label] = None
                continue

            duration_filt = mhw_ds["duration"].where(mhw_ds[thresh_var])

            if not duration_filt.notnull().any():
                print("    → no events")
                longest_mhw[abbrv][label] = None
                continue

            idx = np.unravel_index(np.nanargmax(duration_filt.values), duration_filt.shape)

            year_idx, day_idx, eta_idx, xi_idx = idx
            duration = duration_filt.values[idx]

            lat = biomass_data["lat_rho"].values[eta_idx, xi_idx]
            lon = biomass_data["lon_rho"].values[eta_idx, xi_idx]
            day = biomass_data["days"].values[day_idx]

            longest_mhw[abbrv][label] = {"duration_days": float(duration),
                                        "year": int(1980 + year_idx),
                                        "day_of_year": int(day),
                                        "eta_idx": int(eta_idx),
                                        "xi_idx": int(xi_idx),
                                        "lat": float(lat),
                                        "lon": float(lon),}

            print(f"    {duration:.1f} days | {1980+year_idx} | DOY {day}")


    # Put results to CSV
    rows = []

    for abbrv, mpa_dict in longest_mhw.items():
        mpa_name = mpa_dict["mpa_name"]

        for thresh_label in ["1deg", "2deg", "3deg", "4deg"]:
            event = mpa_dict.get(thresh_label)

            if event is None:
                continue

            rows.append({"mpa_abbrv": abbrv, "mpa_name": mpa_name, "threshold": thresh_label,
                         "duration_days": event["duration_days"],
                         "year": event["year"], "day_of_year": event["day_of_year"],
                         "eta_idx": event["eta_idx"], "xi_idx": event["xi_idx"],
                         "lat": event["lat"], "lon": event["lon"],})

    df_longest = pd.DataFrame(rows)

    # Save
    df_longest.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

else:
    df_longest_mhw = pd.read_csv(out_csv)

# HERE!
# %% ================================= Select MHW event to plot =================================
mpa_choice = 'WS' #'RS' 'AP' 'EA' 'WS' 'SO'
threshold_choice = '3deg'
mhw_event_choice = df_longest_mhw.loc[(df_longest_mhw["mpa_abbrv"] == mpa_choice) & (df_longest_mhw["threshold"] == threshold_choice)].iloc[0]

year_idx=mhw_event_choice["year"]-1980
eta_idx=mhw_event_choice["eta_idx"]
xi_idx=mhw_event_choice["xi_idx"]

# Extract data at that location
biomass_mhw_actual = biomass_mpas_interp[mpa_choice]['actual'].biomass.isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx).median('algo')
biomass_mhw_clim = biomass_mpas_interp[mpa_choice]['clim'].biomass.isel(eta_rho=eta_idx, xi_rho=xi_idx).median('algo')
biomass_mhw_nomhw = biomass_mpas_interp[mpa_choice]['nomhws'].biomass.isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx).median('algo')
biomass_mhw_climtrend = biomass_mpas_interp[mpa_choice]['climtrend'].biomass.isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx).median('algo')
biomass_mhw_nowarming = biomass_mpas_interp[mpa_choice]['nowarming'].biomass.isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx).median('algo')
mhw_timeseries = xr.open_dataset(os.path.join(path_combined_thesh, f'mpas/interpolated/duration_AND_thresh_{mpa_choice}.nc')).isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx)

chla_surf_SO_allyrs= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc')) 
chla_surf_SO_allyrs_event = chla_surf_SO_allyrs.isel(years=year_idx, eta_rho=eta_idx, xi_rho=xi_idx)

# %% ================================= Identify MHW in that cell over the growth season =================================
def label_events(boolean_array):
    labels = np.zeros(len(boolean_array), dtype=int)
    event_id = 0
    in_event = False

    for i, val in enumerate(boolean_array):
        if val and not in_event:
            event_id += 1
            in_event = True
        if not val:
            in_event = False
        labels[i] = event_id if val else 0

    return labels

threshold_events = {}
threshold_map = {'$\\geq$ 90th perc and 1°C': 'det_1deg', '$\\geq$ 90th perc and 2°C': 'det_2deg',
                 '$\\geq$ 90th perc and 3°C': 'det_3deg', '$\\geq$ 90th perc and 4°C': 'det_4deg'}

for label, var in threshold_map.items():
    if var not in mhw_timeseries:
        continue

    bool_days = mhw_timeseries[var].values.astype(bool)

    if not np.any(bool_days):
        continue  # no events at this threshold

    threshold_events[label] = {'days': label_events(bool_days)}

for k, v in threshold_events.items():
    print(k, "→ number of events:", v['days'].max())


# %% ================================= Plot biomass timeserie ================================= 
# Look at the shap of biomass evlution -- see if biomass influence the progretion of biomass increase over the season
okabe_ito = ["#000000", "#009E73", "#0072B2", "#56B4E9", "#F0E442", "#E69F00", "#D55E00", "#CC79A7"]


title_kwargs = {'fontsize': 15} 
label_kwargs = {'fontsize': 12} 
tick_kwargs = {'labelsize': 10} 
suptitle_kwargs = {'fontsize': 18, 'fontweight': 'bold'}
lw = 0.9
# === Threshold colors ===
threshold_colors = {
    '$\geq$ 90th perc and 1°C': '#5A7854',
    '$\geq$ 90th perc and 2°C': '#8780C6',
    '$\geq$ 90th perc and 3°C': '#E07800',
    '$\geq$ 90th perc and 4°C': '#9B2808'
}

# --- Prepare time axis 
days_xaxis = biomass_mhw_actual['days'].values
base_date = datetime(2021, 11, 1)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(len(days_xaxis))]
date_dict = dict(date_list)
tick_positions = np.arange(days_xaxis.min(), days_xaxis.max() + 1, 15)
tick_labels = [date_dict.get(day, '') for day in tick_positions]

# --- Setup figure
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True, constrained_layout=True)

# --- 1. Biomass timeseries 
ax1.plot(days_xaxis, biomass_mhw_clim.values, color=okabe_ito[0],  linewidth=lw, label='Climatology')
ax1.plot(days_xaxis, biomass_mhw_actual.values, color=okabe_ito[1],linewidth=lw, label='Actual')
ax1.plot(days_xaxis, biomass_mhw_nomhw.values, color=okabe_ito[3],  linestyle='--', linewidth=1.3, label='No MHWs')
ax1.plot(days_xaxis, biomass_mhw_climtrend.values, color=okabe_ito[5],  linestyle='--', linewidth=1.3, label='No MHWs (Clim with trend)')
ax1.plot(days_xaxis, biomass_mhw_nowarming.values, color=okabe_ito[-1],  linestyle='--', linewidth=1.3, label='No Warming')
ax1.set_ylabel("Biomass [mg/m³]", **label_kwargs)
ax1.set_title(f"{mhw_event_choice['mpa_name']} in {year_idx+1980}\n" f"Lat {biomass_mhw_actual.lat_rho.values:.2f}, Lon {biomass_mhw_actual.lon_rho.values:.2f}", **title_kwargs)
ax1.tick_params(axis='y', length=2, width=0.5, **tick_kwargs)
ax1.tick_params(axis='x', labelbottom=False)
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left', frameon=True, fontsize=9)


# --- 2. Chla timeseries 
ax2.plot(days_xaxis, chla_surf_SO_allyrs_event.chla.values, color='green',  linewidth=lw)
ax2.set_ylabel("Chla [mg /m³]", **label_kwargs)
ax1.tick_params(axis='y', length=2, width=0.5, **tick_kwargs)
ax1.tick_params(axis='x', labelbottom=False)
ax1.grid(alpha=0.3)

# --- 3. MHW Event Timeline 
ax3.set_ylim(0, len(threshold_events))
ax3.set_yticks([])
ax3.set_ylabel("Detected MHW", rotation=90, labelpad=15, **label_kwargs)

for i, (label, info) in enumerate(threshold_events.items()):
    color = threshold_colors[label]
    active_days = info['days'][:len(days_xaxis)]
    unique_events = np.unique(active_days[active_days > 0])
    for event_id in unique_events:
        idx = np.where(active_days == event_id)[0]
        if len(idx) == 0:
            continue
        x_start = days_xaxis[idx[0]]
        x_end   = days_xaxis[idx[-1]]
        ax3.axvspan(x_start, x_end, color=color, alpha=0.8)

ax3.set_xlabel("Date", **label_kwargs)
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
ax3.tick_params(axis='x', **tick_kwargs, length=2, width=0.5)
ax3.tick_params(axis='y', **tick_kwargs, length=2, width=0.5)

# --- Legend ---
handles = [Patch(facecolor=color, edgecolor='black', lw=0.5) for label, color in threshold_colors.items() if label in threshold_events]
labels = [label for label in threshold_colors.keys() if label in threshold_events]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=len(handles), frameon=True, **label_kwargs)

plt.show()


# %%
