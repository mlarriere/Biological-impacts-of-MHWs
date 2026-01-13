"""
Created on Tue 25 Nov 10:51:30 2025

CEPHALOPOD model outputs for biomass and abundance

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
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

#%% -------------------------------- Server --------------------------------
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% -------------------------------- Figure settings --------------------------------
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
# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
roms_bathymetry = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc').h
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

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

# %% ====================== Biomass data ======================
# --- Load biomass from CEPHALOPOD
# 5 different algorithms, 10 bootstraps per algo
biomass_data = xr.open_dataset('/net/meso/work/aschickele/CEPHALOPOD/output/Marguerite_tot_krill_WOA_2026-01-12 17:27:04.013515/Euphausiacea/tot_krill_biomass.nc') #shape (coords, model, time) = (d1=64800, d2=5*10,  d3=12)


# --- Reformatting
nlat, nlon = 180, 360
lat = np.linspace(-89.5, 89.5, nlat)
lon = np.linspace(-179.5, 179.5, nlon)

# Raw data
arr = biomass_data.y_ens.values  # shape: (12, 50, 64800)

# Reshape northing into (lat, lon)
arr = arr.reshape(12, 50, 180, 360)  # (months, bootstrap, lat, lon)

# Flip latitude (before it was going from -90 to 90, i.e. from south pole to north pole)
arr_global = arr[:, :, ::-1, :]  # flip along lat axis
lat_flipped = lat[::-1]

# Create new Dataset
biomass_cephalopod = xr.Dataset(
    data_vars=dict(total_krill_biomass=(["months", "algo_bootstrap", "lat", "lon"], arr_global)), 
    coords=dict(months=np.arange(1, 13),
                algo_bootstrap=np.arange(1, 51),
                lat=lat_flipped, lon=lon),
    attrs=biomass_data.attrs)

# Add info in attributed
biomass_cephalopod.attrs.update({
    "model_name": "Cephalopod",
    "model_resolution": "1 degree",
    "model_extent": "global",
    "model_inputs": "WOA",
    "units":"mg C m-3",
    "main_predictors": "1. Silicates (correlated with nutrients),\n2. Oxygen (correlated with temperature),\n3. Salinity,\n4. Chla (correlated with PP)",
    "ensemble_members": "5 models out of 6: GLM, MLP, GAM, SWM, RF. Not passing: BRT.",
    "note": "Under assumption that krill spend most of their time in the 0-100m, all observations are integrated (median concentration on depth)."
})

# -- Select only the Southern Ocean (south of 60°S)
lat = np.linspace(89.5, -89.5, 180)  # north → south
lat_mask = lat <= -60
arr_60S = arr[:, :, lat_mask, :]
lat_60S = lat[lat_mask]  # lat_60S: -60.5 → -89.5 (north → south)

arr_60S_flipped = arr_60S[:, :, ::-1, :]
lat_60S_flipped = lat_60S[::-1]  # now first row = south pole

biomass_cephalopod_60S = xr.Dataset(
    data_vars=dict(total_krill_biomass=(["months", "algo_bootstrap", "lat", "lon"], arr_60S_flipped)),
    coords=dict(
        months=np.arange(1, 13),
        algo_bootstrap=np.arange(1, 51),
        lat=lat_60S_flipped,
        lon=np.linspace(-179.5, 179.5, 360)
    )
)

# -- Euphausia superba biomass
# Euphausia = 80% of total krill
biomass_cephalopod_60S_euphausia = xr.Dataset(
    data_vars=dict(euphausia_biomass=(["months", "algo_bootstrap", "lat", "lon"], arr_60S_flipped * 0.8)),
    coords=dict(
        months=np.arange(1, 13),
        algo_bootstrap=np.arange(1, 51),
        lat=lat_60S_flipped,
        lon=np.linspace(-179.5, 179.5, 360)
    )
)

# -- Save 
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')
output_file_biomass_tot_krill = os.path.join(path_cephalopod, "total_krill_biomass_SO.nc")
if not os.path.exists(output_file_biomass_tot_krill):
    biomass_cephalopod_60S.to_netcdf(output_file_biomass_tot_krill, engine="netcdf4")

output_file_biomass_euphausia = os.path.join(path_cephalopod, "euphausia_biomass_SO.nc")
if not os.path.exists(output_file_biomass_euphausia):
    biomass_cephalopod_60S_euphausia.to_netcdf(output_file_biomass_euphausia, engine="netcdf4")

# %% ====================== Visualization ======================
from matplotlib.colors import LinearSegmentedColormap

# ===== Median over bootstraps and models =====
biomass_med_ensemble = biomass_cephalopod_60S_euphausia.euphausia_biomass.median(dim=["algo_bootstrap"], skipna=True)

# ===== Month selection =====
growth_season_plot = True #False

if growth_season_plot:
    months_sel = [10, 11, 0, 1, 2, 3]
    month_labels = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
else:
    months_sel = [10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    month_labels = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

nplots = len(months_sel)

# ===== Coordinates =====
lat = biomass_med_ensemble.lat.values
lon = biomass_med_ensemble.lon.values
lon2d, lat2d = np.meshgrid(lon, lat)

# ===== Custom colormap =====
vmax = np.nanpercentile(biomass_med_ensemble, 95)
norm = mcolors.Normalize(vmin=0, vmax=vmax)

# ===== Figure layout =====
ncols = 3
nrows = int(np.ceil(nplots / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.2 * nrows), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.92, wspace=0.05, hspace=0.08)
axes = np.atleast_1d(axes).flatten()

# ===== Circular boundary =====
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# ===== Plot loop =====
for i, (m, label) in enumerate(zip(months_sel, month_labels)):
    ax = axes[i]

    data = biomass_med_ensemble.isel(months=m).values

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(),
                        cmap='inferno', norm=norm, shading="auto")

    # Gridlines
    if i == 0:
        gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlabel_style = {'size': 7, 'rotation': 0}
        gl.ylabel_style = {'size': 7, 'rotation': 0}
    else:
        ax.gridlines(draw_labels=False, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)

    ax.set_title(label, fontsize=11)

# ===== Remove unused axes =====
for ax in axes[nplots:]:
    ax.remove()

# ===== Shared colorbar & title =====
cbar = fig.colorbar(pcm, ax=axes[:nplots], orientation="vertical", shrink=0.8, pad=0.05, extend="max")
cbar.set_label("Biomass [mg C m$^{-3}$]", fontsize=14)
cbar.ax.tick_params(labelsize=12)

fig.suptitle("Euphausia superba biomass\nMedian over algorithms and bootstraps", fontsize=16, y=1.02, x=0.4)
plt.show()



# %% ====================== Initial Biomass ======================
# Biomass in November = Initial Biomass (growth season)
biomass_nov = biomass_cephalopod_60S_euphausia.euphausia_biomass.isel(months=10)

# # Mean over all bootstraps
# mean_biomass_nov = biomass_cephalopod_60S.euphausia_biomass.mean(dim='bootstrap')
# std_biomass_nov = biomass_nov.std(dim='bootstrap')

# %% ====================== Regridding Biomass and Abundance to ROMS grid ======================
import xesmf as xe
output_file_biomass_regrid= os.path.join(path_cephalopod, "euphausia_biomass_SO_regridded.nc")

if not os.path.exists(output_file_biomass_regrid):
    
    # -- Load dataset with correct grid
    area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')['area'].isel(z_t=0)
    area_SO = area_roms.where(area_roms['lat_rho'] <= -60, drop=True) #shape (231, 1442)

    # -- From monthly to daily dataset
    # Repeat monthly value for each day of the month
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_index = np.concatenate([np.repeat(month, days_in_month[month-1]) for month in range(1, 13)])
    assert day_index.shape[0] == 365
    day_index_xr = xr.DataArray(day_index, dims="days", name="month")
    biomass_daily = biomass_cephalopod_60S_euphausia.sel(months=day_index_xr) #shape: (365, 50, 30, 360)
    
    # Check if the same 
    diff = biomass_daily.isel(days=304) - biomass_daily.isel(days=304+14)
    print(float(diff.euphausia_biomass.max()), float(diff.euphausia_biomass.min()))

    # -- Select only austral summer and early spring
    jan_april_biomass = biomass_daily.sel(days=slice(0, 120))
    jan_april_biomass.coords['days'] = jan_april_biomass.coords['days'] 
    nov_dec_biomass = biomass_daily.sel(days=slice(304, 366))
    nov_dec_biomass.coords['days'] = np.arange(304, 365)
    biomass_daily_austral = xr.concat([nov_dec_biomass, jan_april_biomass], dim="days") #shape: (181, 50, 30, 360)

    # -- Fix longitudes
    # ROMS (24.125, 383.875) - put to (0, 360)
    roms_fixed = area_SO.assign_coords(lon_rho=(area_SO.lon_rho % 360)) #min lon_rho = 0.125 

    # CEPHALOPOD longitude (-180, 180) - put to (0, 360)
    biomass_fixed = biomass_daily_austral.assign_coords(lon=((biomass_daily_austral.lon % 360))).sortby("lon") #min long = 0.5

    # -- Target grids
    in_ds = xr.Dataset(
        {"lon": (("lon",), biomass_fixed.lon.values),
        "lat": (("lat",), biomass_fixed.lat.values),})

    out_ds = xr.Dataset(
        {"lon": (("eta_rho", "xi_rho"), roms_fixed.lon_rho.values),
        "lat": (("eta_rho", "xi_rho"), roms_fixed.lat_rho.values),})

    # -- Regridding 
    # Build xESMF regridder 
    regridder = xe.Regridder(
        in_ds,
        out_ds,
        method="bilinear", #weighted avg of the 4 nearest neighbors
        periodic=True, # if global grid put periodic=True, otherwise the edges of the grid won’t line up with each other
        extrap_method="nearest_s2d"  # fill edges with nearest valid neighbor (otherwise, invalid data -> since do not have 4 neighbors)
    ) 
    
    # Perform regridding
    biomass_regridded = regridder(biomass_fixed) #shape (181, 50, 231, 1442)

    # -- Add coordinates (lat, lon) from ROMS
    biomass_regridded = biomass_regridded.assign_coords(
        lon_rho=(("eta_rho", "xi_rho"), area_SO.lon_rho.values),
        lat_rho=(("eta_rho", "xi_rho"), area_SO.lat_rho.values))


    # -- Save to file 
    biomass_regridded.to_netcdf(output_file_biomass_regrid, engine="netcdf4")

    # ====================== Plot regridded product ======================
    show_differences=True

    if show_differences:
        # ----------------- Median over bootstrap and models -----------------
        biomass_before_med = biomass_fixed.euphausia_biomass.isel(days=0).median(dim="algo_bootstrap", skipna=True)
        biomass_after_med = biomass_regridded.euphausia_biomass.isel(days=0).median(dim="algo_bootstrap", skipna=True)

        # ----------------- Figure -----------------
        fig, axes = plt.subplots(1, 2, figsize=(9, 7), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.06, top=0.92, wspace=0.05)
        axes = np.atleast_1d(axes).flatten()

        # ----------------- Circular boundary -----------------
        theta = np.linspace(0, 2*np.pi, 200)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * 0.5 + 0.5)

        # ----------------- Colorscale -----------------
        vmax = np.nanpercentile(biomass_before_med, 95)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        # norm = mcolors.Normalize(*np.nanpercentile(biomass_before_med, [5, 95]))

        # ----------------- PLOT LOOP -----------------
        for i, ax in enumerate(axes):
            ax.set_extent([0, 360, -90, -60], crs=ccrs.PlateCarree())
            ax.set_boundary(circle, transform=ax.transAxes)
            ax.set_anchor('C')

            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

            # Gridlines
            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}

            if i == 0:
                gl.xlabels_top = False
                gl.ylabels_right = False
                data = biomass_before_med
                lon = biomass_fixed.lon
                lat = biomass_fixed.lat
                title = "BEFORE regridding (1°)"
            else:
                gl.xlabels_top = False
                gl.ylabels_left = False
                gl.ylabels_right = False
                data = biomass_after_med
                lon = biomass_regridded.lon_rho
                lat = biomass_regridded.lat_rho
                title = "AFTER regridding (ROMS 0.25°)"

            pcm = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                                cmap='inferno', norm=norm, shading='auto', zorder=1)
            ax.set_title(title, fontsize=11)

        # ----------------- Colorbar -----------------
        cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', shrink=0.6, pad=0.05, extend='max')
        cbar.set_label("Biomass [mg C m$^{-3}$]", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # ----------------- Title -----------------
        fig.suptitle("Euphausia superba biomass\nMedian over algorithms and bootstraps", fontsize=16, y=0.9, x=0.4)
        plt.show()

else:
    biomass_regridded = xr.open_dataset(output_file_biomass_regrid)



# %% ====================== Spread of models ======================
# --- Prepare data 
# biomass_spatial_mean = biomass_regridded.euphausia_biomass.mean(dim=("eta_rho", "xi_rho"))  # shape (181, 50)

# Total Biomass
layer_thickness = 100 # meters
total_biomass_mg = (biomass_regridded.euphausia_biomass * roms_fixed).sum(dim=("eta_rho","xi_rho")) * layer_thickness

# Convert to metric tons
total_biomass_tons = total_biomass_mg / 1e9

days_in_month = [30, 31, 31, 28, 30, 31]  # Nov, Dec, Jan, Feb, Mar, Apr
month_labels = ["Nov","Dec","Jan","Feb","Mar","Apr"]
month_edges = np.cumsum([0]+days_in_month)

# Assign month index to each day
month_idx = np.zeros(total_biomass_tons.days.size, dtype=int)
for i in range(len(days_in_month)):
    start = month_edges[i]
    end = month_edges[i+1]
    month_idx[start:end] = i

# --- Compute medians
n_models = 5
boot_per_model = 10
model_colors = ['#F94144', '#F8961E', '#90BE6D', '#9E1A6B', '#277DA1']

month_violin_data = []        # all bootstraps per month (for violin)
month_model_medians = []      # list of median per model per month

for m in range(len(days_in_month)):
    days_sel = np.where(month_idx == m)[0]
    all_bootstraps = []
    medians = []
    for model in range(n_models):
        start = model * boot_per_model
        end = (model+1) * boot_per_model
        data = total_biomass_tons.isel(days=days_sel, algo_bootstrap=slice(start, end)).values.flatten()
        all_bootstraps.extend(data)
        medians.append(np.median(data))
    month_violin_data.append(all_bootstraps)
    month_model_medians.append(medians)

# --- Violin plot 
fig, ax = plt.subplots(figsize=(8,5))
vp = ax.violinplot(month_violin_data, showmeans=False, showmedians=True, showextrema=False)
for pc in vp['bodies']:
    pc.set_facecolor('lightgray')
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
if 'cmedians' in vp:
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(2)

model_names = ['GLM', 'MLP', 'GAM', 'SVM', 'RF']
for i, medians in enumerate(month_model_medians):
    for j, median in enumerate(medians):
        ax.scatter(i+1, median, color=model_colors[j], s=10, zorder=3, label=model_names[j] if i==0 else "")

# Add legend for the ensemble median
from matplotlib.lines import Line2D
median_handle = Line2D([0], [0], color='black', lw=2, label='Ensemble')

handles, labels = ax.get_legend_handles_labels()
handles.append(median_handle)
labels.append('Ensemble')

# X-axis labels
ax.set_xticks(range(1, len(month_labels)+1))
ax.set_xticklabels(month_labels)

ax.tick_params(axis='both', which='major', labelsize=12)

# Labels and title
ax.set_ylabel("Total Biomass [ton C]", fontsize=12)
ax.set_xlabel("Months", fontsize=12)
ax.set_title("Euphausia superba total biomass in the Southern Ocean\nSpread of models and bootstraps per month", fontsize=14)

# Legend
ax.legend(handles=handles, loc='upper center', title="Medians", fontsize=11, ncol=3, title_fontsize=12)

plt.tight_layout()
plt.show()


# %%
