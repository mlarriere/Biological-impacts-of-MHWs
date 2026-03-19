"""
Created on Wedn 18 March 10:46:03 2026

Answering the question:
"How is krill biomass in the southern ocean changing during MHWs?"

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

# %% ======================== Load Biomass data ========================
surrogate_names = {"clim": "Climatology", "actual": "Actual Conditions", "clim_trend":"Climatology wih trend", "nowarming": "No Warming"}
files_interp = [os.path.join(path_biomass_ts_SO, f"biomass_{sur}.nc") for sur in surrogate_names.keys()]
biomass_mpas_interp = {}

for surrog in surrogate_names.keys():
    fname = os.path.join(path_biomass_ts_SO, f"biomass_{surrog}.nc")
    biomass_mpas_interp[surrog] = xr.open_dataset(fname)

# %% ======================== Change in Biomass ========================
# Step1. Seasonal gains for the different surrogates
seasonal_gain_interp = {}
for surrog in surrogate_names.keys():
    seasonal_gain_interp[surrog] = biomass_mpas_interp[surrog].isel(days=-1) - biomass_mpas_interp[surrog].isel(days=0) #shape (39, 5, 231, 1440)

# Step2. Percentage of change relative to climatology per grid cell
p_change_interp_cell = {}
for surrog in [s for s in surrogate_names if s != "clim"]:
    p_change_interp_cell[surrog] = (seasonal_gain_interp[surrog] - seasonal_gain_interp['clim'])/seasonal_gain_interp['clim'] * 100  #shape (39, 5)

# Take the median over the bootstraps
p_actual_change_median_cell = p_change_interp_cell['actual'].median(dim=['bootstraps'])

# %% ======================== Temperature data ========================
ds_mhw = xr.open_dataset(os.path.join(path_combined_thesh, f"duration_AND_thresh_5mSEASON.nc"))
main_path='/nfs/sea/work/mlarriere/mhw_krill_SO'

temp_surf_file = os.path.join(main_path, 'temp_surf_seasonal.nc')
temp_surf_clim_file = os.path.join(path_clim, "threshold_90perc_surf_seasonal.nc")

if not os.path.exists(temp_surf_file) and not os.path.exists(temp_surf_clim_file):
    temp_surf= xr.open_dataset(os.path.join(main_path, f"temp_surf.nc"))
    temp_surf_clim =xr.open_dataset(os.path.join(path_clim, "threshold_90perc_surf.nc"))

    # -- Extent south of 60°S
    south_mask = temp_surf_clim['lat_rho'] <= -60
    temp_surf_clim = temp_surf_clim.where(south_mask, drop=True) #shape (365, 231, 1442)
    south_mask = temp_surf['lat_rho'] <= -60
    temp_surf = temp_surf.where(south_mask, drop=True) #shape (40, 365, 231, 1442)

    # -- To seasonal datasets
    def extract_one_season_pair(args):
        ds_y, ds_y1, y = args
        try:
            days_nov_dec = ds_y.sel(days=slice(304, 364))
            days_jan_apr = ds_y1.sel(days=slice(0, 119))

            # Concatenate days and days_of_yr for new season dimension
            combined_days = np.concatenate([
                days_nov_dec['days'].values,
                days_jan_apr['days'].values
            ])
            combined_doy = np.concatenate([
                days_nov_dec['days_of_yr'].values,
                days_jan_apr['days_of_yr'].values
            ])

            season = xr.concat([days_nov_dec, days_jan_apr],
                            dim=xr.DataArray(combined_days, dims="days", name="days"))
            
            season = season.assign_coords(days_of_yr=("days", combined_doy))

            season = season.expand_dims(season_year=[y])
            return season

        except Exception as e:
            print(f"Skipping year {y}: {e}")
            return None

    def define_season_all_years_parallel(ds, max_workers=6):
        from tqdm.contrib.concurrent import process_map

        all_years = ds['years'].values
        all_years = [int(y) for y in all_years if (y + 1) in all_years]

        # Pre-slice only needed years
        ds_by_year = {int(y): ds.sel(years=y) for y in all_years + [all_years[-1] + 1]}

        args = [(ds_by_year[y], ds_by_year[y + 1], y) for y in all_years]

        season_list = process_map(extract_one_season_pair, args, max_workers=max_workers, chunksize=1)

        season_list = [s for s in season_list if s is not None]
        if not season_list:
            raise ValueError("No valid seasons found.")

        return xr.concat(season_list, dim="season_year", combine_attrs="override")

    # Seasonal temperature timeseries
    temp_surf = temp_surf.assign_coords(days_of_yr=('days', temp_surf['days'].values))
    temp_surf_seasonal = define_season_all_years_parallel(temp_surf, max_workers=10)
    temp_surf_seasonal = temp_surf_seasonal.rename({'season_year': 'season_year_temp'})
    temp_surf_seasonal = temp_surf_seasonal.drop_vars('years')
    temp_surf_seasonal = temp_surf_seasonal.rename({'season_year_temp': 'years'})

    # Seasonal climatology (no year dim, just extract Nov-Apr days in order)
    seasonal_doy = ds_mhw['days_of_yr'].values
    temp_surf_clim_seasonal = temp_surf_clim['relative_threshold'].sel(days=seasonal_doy)

    # Save to file
    temp_surf_seasonal.to_netcdf(temp_surf_file)
    temp_surf_clim_seasonal.to_netcdf(temp_surf_clim_file)
else:
    temp_surf_seasonal=xr.open_dataset(temp_surf_file)
    temp_surf_clim_seasonal = xr.open_dataset(temp_surf_clim_file)

# %% ======================== MHW metrics ========================
# -- Temperature anomaly
temp_anom_file = os.path.join(main_path, 'temp_anomalies_seasonal.nc')
if not os.path.exists(temp_anom_file):
    temp_anomalies = temp_surf_seasonal.temp_surf - temp_surf_clim_seasonal.relative_threshold
    temp_anomalies_reset = temp_anomalies.assign_coords(days=np.arange(181))
    temp_anomalies_reset.to_netcdf(temp_anom_file)
else:
    temp_anomalies_reset=xr.open_dataset(temp_anom_file)
# print(f"Min anomaly: {float(temp_anomalies.min().values):.2f} °C") # -6.04 °C
# print(f"Max anomaly: {float(temp_anomalies.max().values):.2f} °C") #5.67 °C

# -- Cumulative intensity
cum_intensity_file = os.path.join(main_path, 'fixed_baseline30yrs/mhw_cumulative_intensity.nc')
if not os.path.exists(cum_intensity_file):
        
    # Defined the 5 different cases of MHWs: 90th perc, 90th perc +i°C
    years_coord = temp_surf_seasonal['years'] # 1980-2018
    det_90th = (ds_mhw['duration'] > 0).assign_coords(years=years_coord).astype(float)
    det_1deg = ((ds_mhw['duration'] > 0) & (ds_mhw['det_1deg'] == 1)).assign_coords(years=years_coord).astype(float)
    det_2deg = ((ds_mhw['duration'] > 0) & (ds_mhw['det_2deg'] == 1)).assign_coords(years=years_coord).astype(float)
    det_3deg = ((ds_mhw['duration'] > 0) & (ds_mhw['det_3deg'] == 1)).assign_coords(years=years_coord).astype(float)
    det_4deg = ((ds_mhw['duration'] > 0) & (ds_mhw['det_4deg'] == 1)).assign_coords(years=years_coord).astype(float)
    det_cases = {'90th': det_90th.assign_coords(days=np.arange(181)),
                 '1deg': det_1deg.assign_coords(days=np.arange(181)),
                 '2deg': det_2deg.assign_coords(days=np.arange(181)),
                 '3deg': det_3deg.assign_coords(days=np.arange(181)),
                 '4deg': det_4deg.assign_coords(days=np.arange(181)),}

    # Compute cumulative intensity 
    def compute_CI(case_name):
        return case_name, (temp_anomalies_reset * det_cases[case_name]).sum(dim='days')
    results = process_map(compute_CI, list(det_cases.keys()), max_workers=5, chunksize=1, desc='Computing CI')
    CI = {name: da for name, da in results}

    # To Dataset
    CI_ds = xr.Dataset(CI)

    # Save
    CI_ds.to_netcdf(cum_intensity_file)

else: 
    CI_ds= xr.open_dataset(cum_intensity_file)

# %% ======================== MPAs data ========================
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

# %% ======================== Time series for CI and change in biomass ========================
years_coord = np.arange(1980, 2019)

# -- Mean over areas
# Southern Ocean
p_change_SO = np.nanmean(p_actual_change_median_cell.biomass.values.reshape(39, -1), axis=1)
CI_SO = {key: np.nanmean(CI_ds[key].values.reshape(39, -1), axis=1)
         for key in CI_ds}

# MPAs
p_change_mpas = {}
mpa_ci = {}

for abbrv, (name, mask) in mpa_masks.items():
    mask_bool = mask.values.astype(bool)   # (231, 1442)

    p_change_mpas[abbrv] = {'name': name,
        'ts': np.array([np.nanmean(p_actual_change_median_cell.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).biomass.values[y][mask_bool]) for y in range(39)])
    }
    mpa_ci[abbrv] = {
        'name': name,
        'ts': {key: np.array([np.nanmean(CI_ds[key].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values[y][mask_bool]) for y in range(39)])
               for key in CI_ds}
    }

# %% ======================== Plot Time series ========================
ci_key = '3deg'

fig, ax = plt.subplots(figsize=(10, 4))

# -- Plot MPAs
mpa_colors = {abbrv: mpa_dict[name][1] for abbrv, (name, _) in mpa_masks.items()}
abbrvs = list(mpa_ci.keys())
labels = [mpa_ci[a]['name'] for a in abbrvs]
colors = [mpa_colors[a] for a in abbrvs]
stack_data = [mpa_ci[a]['ts'][ci_key] for a in abbrvs]

# -- Find 3 years with highest total stacked CI (sum across all regions)
total_stack = np.array(stack_data).sum(axis=0)  # shape: (n_years,)
top3_idx = np.argsort(total_stack)[-3:]
top3_years = np.array(years_coord)[top3_idx]

# -- Grey shading for top 3 years
for yr in top3_years:
    ax.axvspan(yr - 0.5, yr + 0.5, color='grey', alpha=0.25, zorder=0)

# Stacked area plot for MPAs
ax.stackplot(years_coord, stack_data,
             labels=labels, colors=colors, alpha=0.85)

# -- Plot Whole SO
ax.plot(years_coord, CI_SO[ci_key],
        color='black', lw=2.2, ls='--', label='Southern Ocean')

ax.set_xlabel('Years', fontsize=10)
ax.set_xlim(1980, 2018)
ax.set_ylabel('Cumulative Intensity (°C · days)', fontsize=10)

# -- Add top3 years to xticks with grey angled labels
all_ticks = sorted(set([1980, 1990, 2000, 2010, 2018]) | set(int(y) for y in top3_years))
ax.set_xticks(all_ticks)

fig.canvas.draw()
for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
    if int(tick) in top3_years:
        label.set_color('dimgrey')
        label.set_fontweight('bold')
        label.set_rotation(35)
        label.set_ha('right')
    else:
        label.set_rotation(0)
ax.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.6)
ax.spines[['top', 'right', 'bottom', 'left']].set_visible(True)
ax.tick_params(labelsize=8)

# Reverse legend so stacking order matches visual order
handles, leg_labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], leg_labels[::-1],
          fontsize=8, framealpha=0.85, loc='upper left',
          title='Regions\n', title_fontsize=9)

ax.text(0.01, 1.06, rf'MHW $\ge$ 90th perc and {ci_key[0]}°C',
        transform=ax.transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/cum_itensity_{ci_key[0]}deg.pdf'), dpi=200, format='pdf', bbox_inches='tight')


# %% ======================== Maps of Seasonal Biomass gain change ========================
from skimage import measure
top3_years_sorted = sorted(int(y) for y in top3_years)
years_to_plot = {yr: list(years_coord).index(yr) for yr in top3_years_sorted}
plot='report'

# Colormap
cmap_bio = LinearSegmentedColormap.from_list('purple_white_teal',
               ["#AEA8DE", "white", "#94D2BD"])
norm_bio = mcolors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)

# Setting
lw = 1.0 if plot == 'slides' else 0.5
lw_grid = 0.7 if plot == 'slides' else 0.3
gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
subtitle_kwargs  = {'fontsize': 12, 'fontweight': 'bold'}

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# --- Figure ---
fig, axes = plt.subplots(1, 3, figsize=(10, 4),
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})

for col, (year_label, year_idx) in enumerate(years_to_plot.items()):
    ax   = axes[col]
    data = p_actual_change_median_cell.isel(years=year_idx)

    # Circular boundary
    ax.set_boundary(circle, transform=ax.transAxes)

    # Features
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Plot
    pcm = ax.pcolormesh(data.lon_rho, data.lat_rho, data.biomass,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap_bio, norm=norm_bio,
                        rasterized=True, zorder=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5,
                      linestyle='--', linewidth=lw_grid, zorder=20)
    gl.xlabels_top  = False
    gl.ylabels_right = False
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # MPA boundaries
    lon = mpas_ds.lon_rho
    lat = mpas_ds.lat_rho
    name_to_abbrv = {name: abbrv for abbrv, (name, _) in mpa_masks.items()}

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

        # Add text box next to MPA with spatial mean of percentage change
        abbrv = name_to_abbrv[name]
        mpa_val = p_change_mpas[abbrv]['ts'][year_idx]
        fill_color = '#94D2BD' if mpa_val > 0 else '#AEA8DE'

        # Centroid of MPA
        eta_mean = int(np.mean(np.where(mask_2d > 0.5)[0]))
        xi_mean  = int(np.mean(np.where(mask_2d > 0.5)[1]))
        lon_c = lon_np[eta_mean, xi_mean]
        lat_c = lat_np[eta_mean, xi_mean]

        # Place text box slightly offset from centroid
        lon_off = lon_c + 10
        lat_off = lat_c + 5

        # Annotation with arrow line from MPA centroid to box
        ax.annotate(f'{mpa_val:+.1f}\%',
                    xy=(lon_c, lat_c),                  # arrow tip: MPA centroid
                    xytext=(lon_off, lat_off),           # box position
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    fontsize=6, fontweight='bold', color='black',
                    ha='center', va='center', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=fill_color,
                              edgecolor='black', linewidth=1.0, alpha=0.5),
                    arrowprops=dict(arrowstyle='-', color=color,
                                   linewidth=1.2, zorder=9))
        
    ax.set_title(str(year_label), **subtitle_kwargs)
    if col == 0:
        ax.text(0.01, 1.3, rf'MHW $\ge$ 90th perc and {ci_key[0]}°C',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))

# Shared colorbar
cbar = fig.colorbar(pcm, ax=axes,
                    orientation='vertical', extend='both',
                    fraction=0.03, pad=0.05, shrink=0.6)
cbar.set_label('Biomass change [\%]', fontsize=9)
cbar.set_ticks(np.arange(-50, 51, 10))
cbar.ax.tick_params(labelsize=7)


# MPA legend
# from matplotlib.lines import Line2D
# mpa_handles = [Line2D([0], [0], color=color, lw=2, label=name)
#                for name, (_, color) in mpa_dict.items()]
# fig.legend(handles=mpa_handles, loc='lower center', ncol=5,
#            fontsize=7, framealpha=0.85, bbox_to_anchor=(0.45, -0.04))

# fig.suptitle('Seasonal Biomass Gain Change w.r.t Climatology',
#              fontsize=13, fontweight='bold')

plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/biomass_change_3years_{ci_key[0]}deg.pdf'), dpi=200, format='pdf', bbox_inches='tight')


# %% 
# # Colormap for CI
# from matplotlib.colors import LogNorm

# # Mask zeros so they show as background (lightgrey)
# data = np.where(CI['1deg'].values[year_idx] > 0, CI['1deg'].values[year_idx], np.nan)

# years_to_plot = {1989: 9, 2000: 20, 2016: 36}

# fig, axes = plt.subplots(1, 3, figsize=(10, 4),
#                          subplot_kw={'projection': ccrs.SouthPolarStereo()})

# cmap_ci = plt.cm.YlOrRd.copy()
# cmap_ci.set_bad('lightgrey')   # masked zeros → same as background
# norm_ci = mcolors.Normalize(vmin=0, vmax=5)

# for col, (year_label, year_idx) in enumerate(years_to_plot.items()):
#     ax = axes[col]

#     # Mask zeros
#     data = CI['1deg'].values[year_idx].copy().astype(float)
#     data[data <= 0] = np.nan

#     pcm = ax.pcolormesh(lon2d_n, lat2d, data,
#                         transform=ccrs.PlateCarree(),
#                         cmap=cmap_ci, norm=norm_ci,
#                         rasterized=True, zorder=1)
    
#     ax.set_boundary(circle, transform=ax.transAxes)
#     ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
#     ax.coastlines(color='black', linewidth=lw, zorder=5)
#     ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
#     ax.set_facecolor('lightgrey')

#     pcm = ax.pcolormesh(lon2d_n, lat2d, data,
#                         transform=ccrs.PlateCarree(),
#                         cmap=cmap_ci, norm=norm_ci,
#                         rasterized=True, zorder=1)

#     # MPA boundaries
#     for name, (mask, color) in mpa_dict.items():
#         mask_2d  = mask.values if hasattr(mask, 'values') else mask
#         contours = measure.find_contours(mask_2d.astype(float), 0.5)
#         for contour in contours:
#             ei = np.clip(contour[:, 0].astype(int), 0, lon_np.shape[0] - 1)
#             xi = np.clip(contour[:, 1].astype(int), 0, lon_np.shape[1] - 1)
#             lc = lon_np_n[ei, xi]
#             la = lat_np[ei, xi]
#             brk = np.where(np.abs(np.diff(lc)) > 180)[0] + 1
#             for ls, las in zip(np.split(lc, brk), np.split(la, brk)):
#                 if len(ls) > 1:
#                     ax.plot(ls, las, color=color, linewidth=lw,
#                             transform=ccrs.PlateCarree(), zorder=6)

#     gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5,
#                       linestyle='--', linewidth=lw_grid, zorder=20)
#     gl.xlabels_top   = False
#     gl.ylabels_right = False
#     gl.xlabel_style  = gridlabel_kwargs
#     gl.ylabel_style  = gridlabel_kwargs
#     gl.xformatter    = LongitudeFormatter()
#     gl.yformatter    = LatitudeFormatter()
#     gl.ylocator      = mticker.FixedLocator([-80, -70, -60])
#     gl.xlocator      = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])

#     ax.set_title(str(year_label), fontsize=12, fontweight='bold')

# # Shared colorbar
# cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', extend='max',
#                     fraction=0.03, pad=0.05, shrink=0.6)
# cbar.set_label('Cumulative MHW Intensity (°C·days)', fontsize=9)
# cbar.ax.tick_params(labelsize=7)

# # MPA legend
# mpa_handles = [Line2D([0], [0], color=color, lw=2, label=name)
#                for name, (_, color) in mpa_dict.items()]
# fig.legend(handles=mpa_handles, loc='lower center', ncol=5,
#            fontsize=7, framealpha=0.85, bbox_to_anchor=(0.45, -0.04))

# fig.suptitle('Cumulative MHW Intensity — 90th percentile and 1°C (1980–2018)',
#              fontsize=12, fontweight='bold')


# plt.show()
# # %% Scatter plot
# fig, ax = plt.subplots(figsize=(6, 5))

# x = mpa_ci['AP']['ts']['1deg']
# y = p_change_mpas['AP']['ts']

# norm_yr = mcolors.Normalize(vmin=1980, vmax=2018)

# sc = ax.scatter(x, y, c=years_coord, cmap='coolwarm', norm=norm_yr,
#                 s=60, zorder=3, edgecolors='white', linewidths=0.4)

# ax.axhline(0, color='gray', lw=0.6, ls=':', alpha=0.6)
# ax.set_xlabel('Cumulative MHW Intensity [°C]', fontsize=10)
# ax.set_ylabel('Biomass change [\%]', fontsize=10)
# ax.set_title('Southern Ocean — MHW Intensity vs Krill Biomass Change',
#              fontsize=10, fontweight='bold')

# cbar = fig.colorbar(sc, ax=ax, pad=0.02)
# cbar.set_label('Year', fontsize=9)
# cbar.ax.tick_params(labelsize=8)

# plt.tight_layout()
# plt.show()
# # %%

# %%
