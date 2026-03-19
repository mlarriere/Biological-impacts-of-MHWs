"""
Created on Wedn 18 March 09:02:36 2026

What do krill grow under MHWs in the Southern Ocean

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

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta
import time
from tqdm.contrib.concurrent import process_map

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

from B_Growth_Model.Atkinson2006_model import growth_Atkinson2006  # import growth function

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

path_temp = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'

path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'
path_duration = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/mhw_durations'
path_det = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth'
path_det_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer'
path_combined_thesh= '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/det_depth/austral_summer/combined_thresholds'
path_chla = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/z_TOT_CHL/'
path_growth_inputs = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs'
path_growth = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model'
path_growth_inputs_summer = '/nfs/sea/work/mlarriere/mhw_krill_SO/growth_model/inputs/austral_summer'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold
percentile = 90 

# Handling time
from datetime import datetime, timedelta
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0, 121)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)



# %% ======================== Load data ========================
# ---- MPAs
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)
mpa_dict = {"Ross Sea": (mpas_ds.mask_rs, "#c77c27"),
            "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#e05c8a"),
            "East Antarctic": (mpas_ds.mask_ea, "#C00225"),
            "Weddell Sea": (mpas_ds.mask_ws, "#5f0f40"),
            "Antarctic Peninsula": (mpas_ds.mask_ap, "#867308")}

mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}

# ---- MHWs
mhw_events_surface = xr.open_dataset(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc')) #shape (39, 181, 231, 1442)
# mhw_events_surface_full = xr.open_dataset(os.path.join(path_det, 'duration_AND_thresh_5mFULL.nc')) #shape (39, 181, 231, 1442)
# detection_surf = xr.open_dataset(os.path.join(path_det, 'det_5m.nc'))

# # Compare: how many MHW days per grid cell in seasonal vs full year
# mhw_days_seasonal = (mhw_events_surface['duration'] >= 5).sum(dim=['years', 'days'])
# mhw_days_full     = (mhw_events_surface_full['duration'] >= 5).sum(dim=['years', 'days'])
# print('Seasonal - grid cells with 0 MHW days:', (mhw_days_seasonal == 0).sum().item())
# print('Full     - grid cells with 0 MHW days:', (mhw_days_full == 0).sum().item())
# # Fraction of grid cells with no MHWs in season but yes in full year
# no_season = mhw_days_seasonal == 0
# yes_full  = mhw_days_full > 0
# print('Grid cells with MHWs only outside growth season:', (no_season & yes_full).sum().item())

# ---- Growth
krill_growth = xr.open_dataset(os.path.join(path_growth, 'growth_Atkison2006_seasonal.nc')) #shape (39, 231, 1442, 181)

# %% ======================== Growth under MHWs of different intensities ========================
growth_mhw_file= os.path.join(path_growth, "growth_Atkison2006_seasonal_mhws.nc")

if not os.path.exists(growth_mhw_file):

    # --- Base mask: 90th percentile and 5days duration (Hobday)
    duration_mask = mhw_events_surface['duration'] > 0  # (39, 181, 231, 1442)

    # --- Detection mask
    det = {i: mhw_events_surface[f'det_{i}deg'] == 1 for i in [1, 2, 3, 4]}

    # --- 4 mutually exclusive intensity bins ---
    intensity_masks = {'growth_90perc': duration_mask,
                       'growth_1deg': duration_mask & det[1],
                       'growth_2deg': duration_mask & det[2],
                       'growth_3deg': duration_mask & det[3],
                       'growth_4deg': duration_mask & det[4],
                       }
    
    # --- Align mask coordinates
    def align_mask(mask):
        mask = mask.swap_dims({'days': 'days_of_yr'})
        mask = mask.drop_vars('days')                        # drop 0-180 integer index
        mask = mask.rename({'days_of_yr': 'days'})           # rename to match growth 'days' (304-119)
        mask = mask.assign_coords(years=mask.years + 1980)   # 0-38 → 1980-2018
        return mask
    
    # --- Apply masks to growth ---
    def compute_growth_mask(args):
        growth_name, mask = args

        mask_aligned = align_mask(mask)
        growth_masked = krill_growth['growth'].where(mask_aligned)
        print(f'Done with {growth_name} | All NaN: {np.isnan(growth_masked).all().item()}')
        return growth_name, growth_masked
    
    args = list(intensity_masks.items())
    results = process_map(compute_growth_mask, args, max_workers=5, chunksize=1, desc='Growth under MHWs')
    growth_mhw_dict = dict(results)

    #  Sanity check: each intensity bin should be a subset of the previous
    masks_aligned = {k: align_mask(m) for k, m in intensity_masks.items()}
    for a, b in [('growth_90perc', 'growth_1deg'),
                ('growth_1deg',   'growth_2deg'),
                ('growth_2deg',   'growth_3deg'),
                ('growth_3deg',   'growth_4deg')]:
        # Anywhere b is True, a must also be True
        leak = (masks_aligned[b] & ~masks_aligned[a]).sum().item()
        print(f'{b} outside {a}: {leak} cells (should be 0)')
    
    
    # --- To Dataset
    growth_mhw_combined = xr.Dataset(
        data_vars=growth_mhw_dict,
        coords=krill_growth.coords,
        attrs={"description": "Krill growth during MHWs (Nov 1 – Apr 30).",
               "depth": "5m.",
               "growth_90perc": "SST >= 90th perc and duration >= 5 days.",
               "growth_ideg": "SST >= 90th perc and +i°C (absolute threshold) and duration >= 5 days."
               })

    # Save output
    growth_mhw_combined.to_netcdf(growth_mhw_file, mode='w')
    
else:
    # Load data
    growth_mhw_combined = xr.open_dataset(growth_mhw_file)

# %% ================ Climatological growth ================
growth_clim_file = os.path.join(path_growth, 'growth_Atkison2006_seasonal_clim.nc')
if not os.path.exists(growth_clim_file):
    # -- Climatological drivers
    chla_ds = xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears_detrended_seasonal.nc'))
    chla_clim= chla_ds.chla.isel(years=slice(0, 30)).mean(dim=('years', 'days'))

    temp_ds = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears_seasonal.nc'))
    temp_clim= temp_ds.avg_temp.isel(years=slice(0, 30)).mean(dim=('years', 'days'))
 
    # -- Climatological growth (mature krill, 35mm)
    growth_clim = growth_Atkinson2006(chla_clim.values, temp_clim.values, length=35.0)
    
    # To Dataset
    growth_clim_ds = xr.Dataset({'growth': (['eta_rho', 'xi_rho'], growth_clim)},
                                coords={'lon_rho': chla_clim.lon_rho,
                                        'lat_rho': chla_clim.lat_rho,},
                                attrs={'description': 'Climatological krill growth rate (Atkinson 2006), mature krill of 35mm (initial length), 1980-2009.'})
    
    # Save to file
    growth_clim_ds.to_netcdf(growth_clim_file)
else:
    growth_clim_ds = xr.open_dataset(growth_clim_file)

# %% ================ Comparison with climatological growth ================
growth_anom_file = os.path.join(path_growth, 'growth_Atkison2006_anomaly_mhw.nc')
if not os.path.exists(growth_anom_file):
    growth_anomaly = xr.Dataset(
        {var: growth_mhw_combined[var] - growth_clim_ds['growth'] 
        for var in growth_mhw_combined.data_vars},
        attrs={'description': 'Krill growth anomaly during MHWs relative to climatology (median growth under MHWs minus climatological growth).'}
    )

    # Save to file
    growth_anomaly.to_netcdf(growth_anom_file)
else: 
    growth_anomaly=xr.open_dataset(growth_anom_file)


# -- Take the mean growth anomalies over time
def compute_mean_growth(var_name):
    return var_name, growth_anomaly[var_name].mean(dim=['years', 'days'], skipna=True)

results = process_map(compute_mean_growth, list(growth_anomaly.data_vars), max_workers=5, chunksize=1, desc='Median growth anomaly under MHWs')
growth_anomaly_mhw_anom = xr.Dataset(dict(results))

# %% ======================== Plot ========================
from skimage import measure
plot = 'report'  # slides report

if plot == 'report':
    fig_width = 6.3228348611
    fig_height = fig_width * 1.8  # taller for 3 subplots
else:
    fig_width = 7
    fig_height = 7 * 1.8

# Font size settings
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
subtitle_kwargs  = {'fontsize': 15} if plot == 'slides' else {'fontsize': 9}
label_kwargs     = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}
tick_kwargs      = {'labelsize': 13} if plot == 'slides' else {'labelsize': 9}

# Color settings
from matplotlib.colors import LinearSegmentedColormap
cmap_growth = LinearSegmentedColormap.from_list(
    "blue_white_green",
    ["#08519c", "#4292c6", "#9ecae1", "#f7f7f7", "#a1d99b", "#41ab5d", "#006d2c"]
)
# cmap_growth = 'coolwarm'
vmin, vmax = -0.2, 0.2
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Data and titles
data_to_plot = [
                # growth_anomaly_mhw_anom.growth_90perc, 
                growth_anomaly_mhw_anom.growth_1deg, 
                growth_anomaly_mhw_anom.growth_3deg]

titles = [
        #   r'MHWs $\ge$ 90th percentile', 
          r'MHWs $\ge$ 1$^\circ$C and 90th percentile', 
          r'MHWs $\ge$ 3$^\circ$C and 90th percentile']

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})

lw = 1   if plot == 'slides' else 0.5
lw_grid= 0.7 if plot == 'slides' else 0.3
gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}

for ax, data, title in zip(axes, data_to_plot, titles):

    # Circular boundary
    ax.set_boundary(circle, transform=ax.transAxes)

    # Features
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    pcm = ax.pcolormesh(
        data.lon_rho, data.lat_rho, data,
        cmap=cmap_growth, norm=norm,
        transform=ccrs.PlateCarree(), zorder=2, rasterized=True
    )

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5,
                      linestyle='--', linewidth=lw_grid, zorder=20)
    gl.xlabels_top  = False
    gl.ylabels_right = False
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Subtitle
    ax.set_title(title, **subtitle_kwargs)

    # MPA boundaries
    lon = mpas_ds.lon_rho
    lat = mpas_ds.lat_rho
    for name, (mask, color) in mpa_dict.items():
        mask_2d = mask.values if hasattr(mask, "values") else mask
        lon_np  = lon.values
        lat_np  = lat.values

        contours = measure.find_contours(mask_2d.astype(float), 0.5)
        for contour in contours:
            eta_idx = contour[:, 0].astype(int)
            xi_idx  = contour[:, 1].astype(int)
            ax.plot(lon_np[eta_idx, xi_idx], lat_np[eta_idx, xi_idx],
                    color=color, linewidth=1,
                    transform=ccrs.PlateCarree(), zorder=2)

        # Label MPA name, only on 1st subplot
        # label_offsets = {"Ross Sea": (-40, -8),
        #                  "South Orkney Islands southern shelf": (15, 5),
        #                  "East Antarctic": ( 20, -5), 
        #                  "Weddell Sea": (-15,  8), 
        #                  "Antarctic Peninsula": (-20, -8),}

        # # Label MPA name, only on 1st subplot
        # if ax == axes[0]:
        #     lon_centroid = lon_np[mask_2d].mean() 
        #     lat_centroid = lat_np[mask_2d].mean()
        #     lon_off, lat_off = label_offsets.get(name, (10, 5))
        #     ax.text(lon_centroid + lon_off, lat_centroid + lat_off, name,
        #             transform=ccrs.PlateCarree(), zorder=5,
        #             fontsize=8, fontweight='bold', ha='center', va='center',
        #             bbox=dict(facecolor="white", edgecolor=color, 
        #                       linewidth=1.5,
        #                       boxstyle="round,pad=0.3", alpha=0.9)
            # )
            

# Shared colorbar
cbar = fig.colorbar(pcm, ax=axes,
                    orientation="vertical", extend='both',
                    fraction=0.03, pad=0.05, shrink=0.6)
cbar.set_label("Growth anomaly [mm/d]", **label_kwargs)


from matplotlib.patches import Patch
no_mhw_patch = Patch(facecolor='lightgrey', edgecolor='gray', linewidth=0.5, label='No MHWs detected')
fig.legend(handles=[no_mhw_patch], loc='lower center',
           bbox_to_anchor=(0.2, 0.09), frameon=True, **label_kwargs)

# plt.suptitle("Mean krill growth anomaly during MHWs relative to climatology", **maintitle_kwargs, y=0.91)

if plot == 'report':
    plt.show()
    # plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/growth_anom_mhws.pdf'), dpi=200, format='pdf', bbox_inches='tight')
else:
    plt.show()
# %%
