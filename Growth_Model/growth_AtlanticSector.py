"""
Created on Tue 03 June 16:04:36 2025

How a krill grow during 1 season under MHWs in Atlantic Sector

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import os
import xarray as xr
import numpy as np
import gc
import psutil #retracing memory
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.gridspec as gridspec

import time
from tqdm.contrib.concurrent import process_map

from joblib import Parallel, delayed

#%% Server 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
gc.collect()
print(f"Memory used: {psutil.virtual_memory().percent}%")

# %% Figure settings
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.serif':['Times'],
    "font.size": 10,           
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,   
    "text.latex.preamble": r"\usepackage{mathptmx}",  # to match your Overleaf font
})

# %% -------------------------------- SETTINGS --------------------------------
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

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
# MHW durations
mhw_duration_5m = xr.open_dataset(os.path.join(path_duration, "mhw_duration_5m.nc")).mhw_durations #dataset - shape (40, 365, 434, 1442)
print(mhw_duration_5m.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)
det_combined_ds = xr.open_dataset(os.path.join(path_combined_thesh, 'det_depth5m.nc')) #boolean shape (40, 181, 434, 1442)
print(det_combined_ds.det_1deg.isel(eta_rho=224, xi_rho=583, years=38, days=slice(0,30)).values)

# -- Write or load data
combined_file = os.path.join(os.path.join(path_combined_thesh, 'duration_AND_thresh_5mSEASON.nc'))

if not os.path.exists(combined_file):

    # === Select only austral summer and early spring
    jan_april = mhw_duration_5m.sel(days=slice(0, 120)) # 1 Jan to 30 April (Day 0-119) - last idx excluded
    jan_april.coords['days'] = jan_april.coords['days'] #keep info on day
    jan_april.coords['years'] = 1980+ jan_april.coords['years'] #keep info on day
    nov_dec = mhw_duration_5m.sel(days=slice(304, 365)) # 1 Nov to 31 Dec (Day 304–364) - last idx excluded
    nov_dec.coords['days'] = np.arange(304, 365) #keep info on day
    nov_dec.coords['years'] = 1980+ nov_dec.coords['years'] #keep info on day
    mhw_duration_austral_summer = xr.concat([nov_dec, jan_april], dim="days") #181days

    # === Select 60°S south extent
    south_mask = mhw_duration_austral_summer['lat_rho'] <= -60
    mhw_duration_5m_NEW_60S_south = mhw_duration_austral_summer.where(south_mask, drop=True) #shape (40, 181, 231, 1442)
    det_combined_ds_60S_south = det_combined_ds.where(south_mask, drop=True) #shape (40, 181, 231, 1442)

    # === Associate each mhw duration with the event threshold 
    ds_mhw_duration= xr.Dataset(
        data_vars=dict(
            duration = (["years", "days", "eta_rho" ,"xi_rho"], mhw_duration_5m_NEW_60S_south.data), #shape (40, 181, 434, 1442)
            det_1deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_1deg'].data),
            det_2deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_2deg'].data),
            det_3deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_3deg'].data),
            det_4deg = (["years", "days", "eta_rho" ,"xi_rho"], det_combined_ds_60S_south['det_4deg'].data)
            ),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], mhw_duration_5m_NEW_60S_south.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], mhw_duration_5m_NEW_60S_south.lat_rho.values), #(434, 1442)
            days_of_yr=(['days'], mhw_duration_5m_NEW_60S_south.coords['days'].values), # Keeping information on day 
            years=(['years'], mhw_duration_5m_NEW_60S_south.coords['years'].values), # Keeping information on day 
            ),
        attrs = {
                "depth": "5m",
                "duration":"Duration redefined as following the rules of Hobday et al. (2016), based on relative threshold (90thperc) - based on the condition that a mhw is when T°C > absolute AND relative thresholds",
                "det_ideg": "Detected events where SST > (absolute threshold (i°C) AND 90th percentile) , boolean array"
                }                
            )

    # Write to file
    ds_mhw_duration.to_netcdf(combined_file)

else: 
    # Load data
    ds_mhw_duration = xr.open_dataset(combined_file)

#%% ======================== Select extent (Atlantic Sector) ========================
# According to Atkinson 2009: 70 % of the entire circumpolar population is concentrated within the Southwest Atlantic (0–90W)
growth_seasons = xr.open_dataset(os.path.join(path_growth, "growth_Atkison2006_seasonal.nc"))
growth_seasons = growth_seasons.rename_vars({'days': 'days_of_yr'})

def subset_spatial_domain(ds, lat_range=(-80, -60), lon_range=(270, 360)): #, (0, 30)
    lat_min, lat_max = lat_range
    lon_range1, lon_range2 = lon_range

    lat_mask = (ds['lat_rho'] >= lat_min) & (ds['lat_rho'] <= lat_max)
    lon_mask = ((ds['lon_rho'] >= lon_range1) & (ds['lon_rho'] <= lon_range2)) #| ((ds['lon_rho'] >= lon_range2[0]) & (ds['lon_rho'] <= lon_range2[1]))

    return ds.where(lat_mask & lon_mask, drop=True)

# -- Write or load data
growth_file = os.path.join(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc'))
duration_file = os.path.join(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc'))

if not os.path.exists(growth_file):
    growth_study_area = subset_spatial_domain(growth_seasons) #shape (years, eta_rho, xi_rho, days): (39, 360, 385, 181) - 2019 excluded (not full season)
    growth_study_area.to_netcdf(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc')) # Write to file

if not os.path.exists(duration_file):
    mhw_duration_study_area = subset_spatial_domain(ds_mhw_duration) #shape (years, days, eta_rho, xi_rho) :(40, 181, 231, 360)
    mhw_duration_study_area.to_netcdf(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc')) # Write to file

else: 
    # Load data
    growth_study_area = xr.open_dataset(os.path.join(path_growth, 'atlantic_sector/growth_study_area.nc'))
    mhw_duration_study_area = xr.open_dataset(os.path.join(path_growth, 'atlantic_sector/mhw_duration_study_area.nc'))

# %% ======================== Find year with maximum 4°C MHWs events ========================
# Define thresholds
duration_thresh = 30  # at least 30days
intensity_mask_4deg = mhw_duration_study_area['det_4deg'].astype(bool) # Intensity mask - only 4 deg MHWs

# Valid events (duration > 30days and 4°C events)
valid_events_4deg = mhw_duration_study_area['duration'].where((mhw_duration_study_area['duration'] > duration_thresh) & intensity_mask_4deg, drop=True)
valid_mask_4deg = ~np.isnan(valid_events_4deg.values)

# Find when and where
valid_indices_4deg = np.array(np.nonzero(valid_mask_4deg)).T
years_4deg = valid_events_4deg['years'].values
years_list_4deg = []
for idx in valid_indices_4deg:
    y_idx, _, _, _ = idx
    years_list_4deg.append(years_4deg[y_idx])

import collections
year_counts_4deg = collections.Counter(years_list_4deg)
year_max_events_4deg, max_events = year_counts_4deg.most_common(1)[0]
print(f"Year with most 4°C MHW events: {year_max_events_4deg} ({max_events} events)")

# year_max_events_4deg=2017


#%% ======================== Plot Study Area for 1 year ========================
# Year of interest
selected_year = year_max_events_4deg
year_index = selected_year - 1980

# Threshold info
threshold_vars = ['det_1deg', 'det_2deg', 'det_3deg', 'det_4deg']
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']

# Create figure and subplots
from matplotlib.projections import PolarAxes
fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width *2
fig = plt.figure(figsize=(fig_width, fig_width/2))#fig_width, fig_height
gs = gridspec.GridSpec(1, 2, width_ratios=[1,1], wspace=0.75)  # very small space
ax0 = fig.add_subplot(gs[0], projection=ccrs.SouthPolarStereo())
ax1 = fig.add_subplot(gs[1], projection=ccrs.SouthPolarStereo())

# === Subplot 1: Growth ===
growth_data = growth_study_area.growth.isel(years=year_index).mean(dim='days')
growth_plot = growth_data.plot.pcolormesh(ax=ax0, transform=ccrs.PlateCarree(),
                                          x='lon_rho', y='lat_rho', 
                                          cmap='PuOr_r', add_colorbar=False,
                                          norm=mcolors.TwoSlopeNorm(vmin=np.nanmin(growth_study_area.growth), 
                                                                    vcenter=0, 
                                                                    vmax=np.nanmax(growth_study_area.growth)),
                                          rasterized=True)
ax0.set_title(f"Mean growth")# \n Growing season {selected_year}-{selected_year+1}")

# Colorbar for growth
cbar1 = fig.colorbar(
    growth_plot,
    ax=ax0,
    orientation='vertical',
    shrink=0.8,
    pad=0.05,
    aspect=20,
    fraction=0.05,
)
cbar1.ax.set_position(cbar1.ax.get_position())  # optional tweak
cbar1.set_label("Growth [mm/d]")
# cbar1.ax.tick_params(labelsize=11)

# === Subplot 2: Detected Events ===
# Base: where no MHW was detected at any threshold
no_event_mask = (
    (mhw_duration_study_area['det_1deg'].isel(years=year_index).mean(dim='days') == 0) &
    (mhw_duration_study_area['det_2deg'].isel(years=year_index).mean(dim='days') == 0) &
    (mhw_duration_study_area['det_3deg'].isel(years=year_index).mean(dim='days') == 0) &
    (mhw_duration_study_area['det_4deg'].isel(years=year_index).mean(dim='days') == 0)
).fillna(True)

# Plot no detected MHW in white
ax1.contourf(mhw_duration_study_area.lon_rho, mhw_duration_study_area.lat_rho, no_event_mask, 
             levels=[0.5, 1], colors=['white'], 
             transform=ccrs.PlateCarree(), zorder=1)

# Plot detected MHW events for each threshold
for var, color in zip(threshold_vars, threshold_colors):
    event_mask = mhw_duration_study_area[var].isel(years=year_index).mean(dim='days').fillna(0)
    binary_mask = (event_mask >= 0.166).astype(int)  # 1 if event ≥ 30 days else 0
    ax1.contourf(mhw_duration_study_area.lon_rho, mhw_duration_study_area.lat_rho, binary_mask, 
                 levels=[0.5, 1], colors=[color], 
                 transform=ccrs.PlateCarree(), 
                 alpha=0.8, zorder=2)
ax1.set_title(f"Frequent MHW events")# \n Growing season {selected_year}-{selected_year+1}")

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor='white', edgecolor='black', label='No MHW event', linewidth=0.5)]
legend_handles += [Patch(facecolor=c, edgecolor='black', label=l, linewidth=0.5) for c, l in zip(threshold_colors, threshold_labels)]
ax1.legend(handles=legend_handles, 
           loc='upper center', bbox_to_anchor=(0.8, 0.52),
           fontsize=8, ncol=1, frameon=True)


# Maps Features
theta = np.linspace(np.pi / 2,np.pi, 100)  # from 0° to -90° clockwise - Quarter-circle sector boundary
# theta = np.linspace(0 , 2*np.pi, 100)  # from 0° to 369° clockwise
center, radius = [0.5, 0.5], 0.5 # centered at 0.5,0.5
arc = np.vstack([np.cos(theta), np.sin(theta)]).T
verts = np.concatenate([[center], arc * radius + center, [center]])
circle = mpath.Path(verts)

for ax in [ax0, ax1]:
    ax.set_boundary(circle, transform=ax.transAxes)

    # Sectors delimitation
    for i in [-90, 0, 120]:
        ax.plot([i, i], [-90, -60], transform=ccrs.PlateCarree(), color='#495057', linestyle='--', linewidth=1)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=0.7)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Map extent and features
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    # ax.set_extent([0, -90, -90, -60], crs=ccrs.PlateCarree())
    ax.coastlines(color='black', linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')


fig.suptitle(f"Growth and MHW events from 1st November {selected_year} until 30th April {selected_year+1}", y=1.05)
fig.text(0.5, 0.1, f"Note that on panel2, frequent corresponds to events lasting $\geq${round(181*0.166)} days during the growing season", ha='center')

# Final layout
plt.tight_layout(rect=[0, 0, 0.5, 0.95]) #[right, left, bottom, top]
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/atlantic_sector{selected_year}.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% Select only year of interest 
def defining_season(ds, starting_year):

    # Extract day in season (Nov–Apr)
    days_nov_dec = ds.sel(days=slice(304, 364), years=starting_year)
    days_jan_apr = ds.sel(days=slice(0, 119), years=starting_year + 1)

    # Concatenate with original day-of-year coordinate preserved
    ds_season = xr.concat([days_nov_dec, days_jan_apr], dim=xr.DataArray(np.concatenate([days_nov_dec['days'].values, days_jan_apr['days'].values]), dims="days", name="days"))

    return ds_season

# Calling function with starting year being the year with the maximum number of events
growth_1season = defining_season(growth_study_area, year_max_events_4deg)
mhw_duration_1season = defining_season(mhw_duration_study_area, year_max_events_4deg)

# %% Define grwoth season for all years
def define_season_all_years(ds):
    # ds = growth_redimensioned
    all_years = ds['years'].values
    season_list = []
    all_years_set = set(all_years)
    
    for y in all_years: #about 40s per year
        print(y)
        if (y + 1) in all_years_set:  
        # Verify if next year exist - otherwise incomplete season
            try: 
                days_nov_dec = ds.sel(days=slice(304, 364), years=y)
                days_jan_apr = ds.sel(days=slice(0, 119), years=y + 1)

                # Combine days
                season = xr.concat(
                    [days_nov_dec, days_jan_apr],
                    dim=xr.DataArray(np.concatenate([days_nov_dec['days'].values, 
                                                     days_jan_apr['days'].values]),
                                                     dims="days", 
                                                     name="days"))

                season = season.expand_dims(season_year=[y])  # tag with season start year
                season_list.append(season)

            except Exception as e: # Skip incomplete seasons
                print(f"Skipping year {y} due to error: {e}")
                continue

    if not season_list:
        raise ValueError("No valid seasons found.")

    # Concatenate
    return xr.concat(season_list, dim="season_year")

# %% INPUTS growth calculation
# ==== Temperature [°C] -- Weighted averaged temperature of the first 100m - Austral summer - 60S
temp_avg_100m = xr.open_dataset(os.path.join(path_growth_inputs, 'temp_avg100m_allyears.nc')) #shape (40, 181, 231, 1442)
temp_avg_100m = temp_avg_100m.rename({'year': 'years'})
temp_avg_100m_study_area = subset_spatial_domain(temp_avg_100m) #select spatial extent
temp_avg_100m_1season = defining_season(temp_avg_100m_study_area, year_max_events_4deg) #select temporal extent for year with the maximum number of events
temp_avg_100m_1season_SO = defining_season(temp_avg_100m, year_max_events_4deg) #shape (181, 231, 1442)
temp_avg_100m_study_area_allyrs = define_season_all_years(temp_avg_100m_study_area) 
temp_avg_100m_study_area_allyrs_mean = temp_avg_100m_study_area_allyrs.mean(dim='season_year') #shape (181, 231, 360)

# ==== Chla [mh Chla/m3] -- Weighted averaged chla of the first 100m - Austral summer - 60S
chla_surf= xr.open_dataset(os.path.join(path_growth_inputs, 'chla_surf_allyears.nc')) 
chla_surf = chla_surf.rename({'year': 'years'})
chla_surf_study_area = subset_spatial_domain(chla_surf) #select spatial extent
chla_surf_1season = defining_season(chla_surf_study_area, year_max_events_4deg) #select temporal extent for year with the maximum number of events
chla_surf_1season_SO = defining_season(chla_surf, year_max_events_4deg) #shape (181, 231, 1442)
chla_surf_study_area_allyrs = define_season_all_years(chla_surf_study_area) 
chla_surf_study_area_allyrs_mean = chla_surf_study_area_allyrs.mean(dim='season_year')  #shape (181, 231, 360)

#%% === TEST === 
temp_mean_ts = temp_avg_100m_1season['avg_temp'].mean(dim=['eta_rho', 'xi_rho'])
chla_mean_ts = chla_surf_1season['raw_chla'].mean(dim=['eta_rho', 'xi_rho'])
days = temp_avg_100m_1season['avg_temp'].days.values
days_xaxis = np.where(days < 304, days+ 365, days).astype(int)
base_year = 2021  #non-leap year 
doy_list = list(range(304, 364)) + list(range(0+365, 121+365)) #181
date_list = [(doy, (datetime(base_year, 1, 1) + timedelta(days=doy - 1)).strftime('%b %d')) for doy in doy_list]
date_dict = dict(date_list)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot temperature on ax1 (left y-axis)
ax1.plot(days_xaxis, temp_mean_ts, color='#F3722C', label='Mean Temperature (100m)')
ax1.set_xlabel("Date", fontsize=14)
ax1.set_ylabel("Temperature (°C)", color='#F3722C', fontsize=14)
ax1.tick_params(axis='y', labelcolor='#F3722C', labelsize=14)  # bigger y-axis ticks

# Create a second y-axis for chlorophyll
ax2 = ax1.twinx()
ax2.plot(days_xaxis, chla_mean_ts, color='green', label='Mean Surface Chla')
ax2.set_ylabel("Chlorophyll (mg/m³)", color='green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='green', labelsize=14)  # bigger y-axis ticks

# Define labels to keep for the shared x-axis
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)

ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, fontsize=14)  # bigger x-axis tick labels

# Title and grid
plt.title(f"Spatial mean time series of Temperature and Chlorophyll \n Austral Summer {year_max_events_4deg}", fontsize=18)
ax1.grid(True)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/drivers_study_area_{year_max_events_4deg}.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %% ---- Valid Chla data
valid_fraction = (~np.isnan(chla_surf_1season['raw_chla'])).mean(dim=['eta_rho', 'xi_rho'])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(days_xaxis, valid_fraction, color='blue', label='Valid Chla Fraction')
ax.set_title("Fraction of Valid (Non-NaN) Chlorophyll Values Over Time", fontsize=16)
ax.set_ylabel("Fraction of Valid Grid Cells", fontsize=14)
ax.set_xlabel("Date", fontsize=14)
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# ----- CHLA all cells in the region
fig_all, ax_all = plt.subplots(figsize=(10, 6))
ntime, nlat, nlon = chla_surf_1season['raw_chla'].shape
for i in range(nlat):
    for j in range(nlon):
        chla_ts = chla_surf_1season['raw_chla'][:, i, j].values
        ax_all.plot(days_xaxis, chla_ts, color='green', alpha=0.1, linewidth=0.5)
ax_all.plot(days_xaxis, chla_mean_ts, color='darkgreen', linewidth=2, label='Mean Surface Chla')
ax_all.set_xticks(tick_positions)
ax_all.set_xticklabels(tick_labels, rotation=45, fontsize=12)
ax_all.set_xlabel("Date", fontsize=14)
ax_all.set_ylabel("Chlorophyll (mg/m³)", fontsize=14)
ax_all.grid(True)
ax_all.set_title(f"All Chlorophyll Time Series + Mean\nAustral Summer {year_max_events_4deg}", fontsize=16)
ax_all.legend(fontsize=12, loc='upper left')
plt.show()

# %% Calculating length for extent
# Funtion to simulate daily growth based on CHLA, temperature, and initial length
import sys
sys.path.append(working_dir+'Growth_Model') 
import growth_model
print(dir(growth_model))
from growth_model import growth_Atkison2006, length_Atkison2006  # import growth function

simulated_length_full_SO= length_Atkison2006(chla=chla_surf_1season_SO.raw_chla, temp=temp_avg_100m_1season_SO.avg_temp, initial_length= 35, intermoult_period=10)
mean_length_area_SO = simulated_length_full_SO.mean(dim=["eta_rho", "xi_rho"])

simulated_length_study_area_1980_2019= length_Atkison2006(chla=chla_surf_study_area_allyrs_mean.raw_chla, temp=temp_avg_100m_study_area_allyrs_mean.avg_temp, initial_length= 35, intermoult_period=10)
mean_length_study_area_1980_2019 = simulated_length_study_area_1980_2019.mean(dim=["eta_rho", "xi_rho"])

simulated_length_study_area = length_Atkison2006(chla=chla_surf_1season.raw_chla, temp=temp_avg_100m_1season.avg_temp, initial_length= 35, intermoult_period=10)
mean_length_area_study_area = simulated_length_study_area.mean(dim=["eta_rho", "xi_rho"])
std_length_area_study_area = simulated_length_study_area.std(dim=["eta_rho", "xi_rho"])


# %% ======================== Find all MHWs events ========================
abs_thresholds = [1, 2, 3, 4]
valid_events_dict = {}  # To store data per threshold

for thresh in abs_thresholds:
    print(f'------ {thresh} ------')
    # Select intensity of MHWs
    intensity_mask = mhw_duration_study_area[f'det_{thresh}deg'].astype(bool)
    # Valid events
    valid_events = mhw_duration_study_area['duration'].where(((mhw_duration_study_area['duration'] > duration_thresh) & intensity_mask), drop=True)
    valid_mask = ~np.isnan(valid_events.values)

    # Find when and where
    valid_indices = np.array(np.nonzero(valid_mask)).T
    years_dim, days_dim, eta_dim, xi_dim = valid_events.shape
    years = valid_events['years'].values
    days = valid_events['days_of_yr'].values
    lat_rho = valid_events['lat_rho'].values
    lon_rho = valid_events['lon_rho'].values

    years_list = []
    days_list = []
    lats_list = []
    lons_list = []
    duration_list = []

    for idx in valid_indices:
        y_idx, d_idx, e_idx, x_idx = idx
        years_list.append(years[y_idx])
        days_list.append(days[d_idx])
        lats_list.append(lat_rho[e_idx, x_idx])
        lons_list.append(lon_rho[e_idx, x_idx])
        duration_list.append(valid_events.values[y_idx, d_idx, e_idx, x_idx])

    # Store results
    valid_events_dict[thresh] = {
        'years_array': np.array(years_list),
        'days_array': np.array(days_list),
        'lats_array': np.array(lats_list),
        'lons_array': np.array(lons_list),
        'duration_array': np.array(duration_list),
    }

    print(f"Threshold {thresh}°C: Found {len(years_list)} valid events")


# %% Growth during MHWs
length_all_thresh = []

for thresh in abs_thresholds:
    print(f" ----------- {thresh}°C -----------")

    # Extract data arrays from valid_events_dict for selected absolute threshold
    years_array = valid_events_dict[thresh]['years_array']
    lats_array = valid_events_dict[thresh]['lats_array']
    lons_array = valid_events_dict[thresh]['lons_array']

    # Select events only from year with max 4deg events
    mask = (years_array == year_max_events_4deg)

    # Find lat, lon of these events
    lat_extreme_locations = lats_array[mask]
    lon_extreme_locations = lons_array[mask]

    eta_xi_indices = []
    for lat_val, lon_val in zip(lat_extreme_locations, lon_extreme_locations):
        # Grid points matching lat/lon with small buffer
        match_mask = (np.isclose(simulated_length_study_area.lat_rho, lat_val, atol=1e-4) &
                      np.isclose(simulated_length_study_area.lon_rho, lon_val, atol=1e-4))
        indices = np.argwhere(match_mask)
        if indices.size > 0:
            eta_xi_indices.append(indices[0])  # first match: [eta_idx, xi_idx]
        else:
            eta_xi_indices.append([np.nan, np.nan])

    # Extract length time series for these events
    length_series_list = []
    for eta_idx, xi_idx in eta_xi_indices:
        if np.isnan(eta_idx) or np.isnan(xi_idx):
            continue  # skip if no valid index found
        ts = simulated_length_study_area[:, int(eta_idx), int(xi_idx)]
        length_series_list.append(ts.values)

    # Calculating averaged time series (time, n_cells)
    length_series_array = np.stack(length_series_list, axis=1)  # axis=0: time, axis=1: cells
    average_length_ts = np.nanmean(length_series_array, axis=1)  # mean over cells, result shape: (time,)

    length_all_thresh.append(average_length_ts)


# %% PLotting length over 1 season
from collections import OrderedDict

# Define colors and labels
threshold_colors = ['#5A7854', '#8780C6', '#E07800', '#9B2808']
threshold_labels = ['1°C and 90th perc', '2°C and 90th perc', '3°C and 90th perc', '4°C and 90th perc']

# Deal with day to have continuous x-axis
days_xaxis = np.where(simulated_length_study_area.days.values < 304, simulated_length_study_area.days.values + 365, simulated_length_study_area.days.values).astype(int)
from datetime import datetime, timedelta
base_date = datetime(2021, 11, 1)  # Nov 1 (season start)
date_list = [(i, (base_date + timedelta(days=i)).strftime('%b %d')) for i in range(181)]
date_dict = dict(date_list)

days_xaxis = np.arange(181)  # Just 0 to 180 (1 per day)


fig_width = 6.3228348611  # inches = \textwidth
fig_height = fig_width *2

fig, ax_len = plt.subplots(figsize=(fig_width, fig_width/1.9))#fig_width, fig_height

# --- Bottom subplot: Length time series ---
for i, avg_length_ts in enumerate(length_all_thresh):
    ax_len.plot(days_xaxis, avg_length_ts, label=threshold_labels[i], color=threshold_colors[i], linewidth=2)
mean_values_1980_2019 = mean_length_study_area_1980_2019.values
mean_values = mean_length_area_study_area.values
std_values = std_length_area_study_area.values
ax_len.plot(days_xaxis, mean_values, label=f"Length$_{{{year_max_events_4deg}-{year_max_events_4deg+1}}}$", color="black", linestyle='-')
ax_len.plot(days_xaxis, mean_values_1980_2019, label="Length$_{1980-2019}$", color="grey", linestyle='--')
ax_len.fill_between(days_xaxis,
                mean_values - std_values,
                mean_values + std_values,
                color="gray", alpha=0.2, label=f"±1 $\sigma$ $_{{{year_max_events_4deg}-{year_max_events_4deg+1}}}$")

# Define labels to keep
wanted_labels = {"Nov 01", "Dec 01", "Jan 01", "Feb 01", "Mar 01", "Apr 01", "Apr 30"}
tick_positions = []
tick_labels = []
for day, label in date_dict.items():
    if label in wanted_labels:
        tick_positions.append(day)
        tick_labels.append(label)
ax_len.set_xticks(tick_positions)
ax_len.set_xticklabels(tick_labels, rotation=45)
ax_len.set_xlabel("Date")
ax_len.set_ylabel("Length (mm)")
# ax_len.set_ylim(10, 50)
ax_len.grid(True, alpha=0.3)

ax_len.tick_params(axis='both')
ax_len.legend(loc='upper left')

fig.suptitle(f"Length of krill in the Atlantic Sector — {year_max_events_4deg}-{year_max_events_4deg+1}")


plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/case_study_impactMHWs/length_mhw_{year_max_events_4deg}.pdf'), dpi =150, format='pdf', bbox_inches='tight')

# %%
