"""
Created on Mon 17 Feb 09:32:05 2025

Defining climatology according to the methodology of Hobday et al. (2016)

@author: Marguerite Larriere (mlarriere)
"""

# %% --------------------------------PACKAGES------------------------------------
import sys
import os
import xarray as xr
import numpy as np
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from joblib import Parallel, delayed

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

# %% Settings
# Set working directory
working_dir = "/home/mlarriere/Projects/biological_impacts_MHWs/Biological-impacts-of-MHWs/"
os.chdir(working_dir)
print("Working directory set to:", os.getcwd())

# Directories
ds_roms = xr.open_dataset('/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/SO_d025_avg_daily_1979.nc')
z_rho = np.load('/home/jwongmeng/work/ROMS/scripts/coords/z_rho.npy')

path_mhw = '/nfs/meso/work/jwongmeng/ROMS/model_runs/hindcast_2/output/avg/corrected/eta_chunk/' # drift and bias corrected temperature files
joel_path ='/home/jwongmeng/work/ROMS/scripts/mhw_krill/' #codes joel
output_path_clim = '/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/'

# Sizes and dimensions
years = range(1980, 2020)
nyears = np.size(years)
months = range(1, 13)
days = range(0, 365)
ndays = np.size(days)
nz = 35  # depths levels
neta = 434 # lat
nxi = 1442  # lon

var = 'temp' #variable of interest
file_var = 'temp_DC_BC_'


# -- Define Climatology - baseline type
# baseline = 'fixed1980' 
baseline = 'fixed30yrs' 

if baseline=='fixed1980':
    description = "Detected events" + f'T°C > T°C 1980' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline1980/'
if baseline=='fixed30yrs':
    description = "Detected events" + f'T°C > climatology (1980-2010)' + " (boolean array)" #description for the xarray
    output_path = '/nfs/sea/work/mlarriere/mhw_krill_SO/fixed_baseline30yrs/'

# -- Define Thresholds
absolute_thresholds = [1, 2, 3, 4] # Fixed absolute threshold


# %%# %% ---- Climatology for each eta
def calculate_climSST(ieta, baseline):

    print(f"Processing eta {ieta}...")
    start_time = time.time()

    # ieta =200
    # Read data
    fn = path_mhw + file_var + 'eta' + str(ieta) + '.nc' #dim: (year: 41, day: 365, z_rho: 35, xi_rho: 1442)
    ds_original = xr.open_dataset(fn)[var][1:31,0:365,:,:] #Extracts daily data : only 30yr + consider 365 days per year. shape:(30, 365, 35, 1442)
    # ds_original_surf = ds_original.isel(z_rho=0) #select only surf for faster computing
    ds_original_100m = ds_original.isel(z_rho=slice(0,14)) #first 100m depth - 14 levels
    # print(np.unique(ds_original.lat_rho.values))
    # ds_original.values[np.isnan(ds_original.values)] = 0 # Set to 0 so that det = False at nans
    
    # Initialization arrays 
    nyears, ndays, nz, nxi = ds_original_100m.shape
    relative_threshold = np.full((ndays, nz, nxi),np.nan,dtype=np.float32)
    climatology30yrs = np.full((ndays, nz, nxi),np.nan,dtype=np.float32)

    # Deal with NANs values
    mask_nanvalues = np.where(np.isnan(ds_original_100m.mean(dim='year').values), False, True)  # 16sec Mask: True where valid, False where NaN. shape:(365, 14, 1442) same as if it take avg over years - nan =land

    if baseline == 'fixed1980':
        climatology_sst = ds_original[0, :, :, :]  # Only 1980
    elif baseline == 'fixed30yrs': # ~4min
        # Moving window of 11days - method from Hobday et al. (2016)
        for dy in range(0,ndays): #days index going from 0 to 364
            if dy<=4:
                window11d_sst = ds_original_100m.isel(day=np.concatenate([np.arange(360+dy,365,1), np.arange(0, dy+6,1)]))
            elif dy>=360:
                window11d_sst = ds_original_100m.isel(day=np.concatenate([np.arange(dy-5, 365, 1), np.arange(0,dy-359,1)]))
            else:
                window11d_sst = ds_original_100m.isel(day=np.arange(dy-5, dy+6, 1)) #shape: (30, 11, 14, 1442)

            # Calculate SST climatology and threshold
            relative_threshold[dy, :, :] = np.percentile(window11d_sst, 90, axis=(0,1)) # 90th percentile in 11days time window over 30yrs. shape: (365, 14, 1442)
            climatology30yrs[dy, :, :]  = np.nanmedian(window11d_sst, axis=(0,1)) # median of 30yrs in 11days time window over 30yrs- ignoring NaNs . shape:(365, 14, 1442)
            # climatology30yrs[dy,:, :]  = np.nanmean(window11d_sst, axis=(0,1)) # mean -- compute the median along the specified axis, while ignoring NaNs)
    
    # relative_threshold[:, 0, 1000]
    # window11d_sst.isel(xi_rho=1000, z_rho=0)

    # Apply mask
    relative_threshold = np.where(mask_nanvalues, relative_threshold, np.nan) #shape (365, 14, 1442)
    climatology30yrs = np.where(mask_nanvalues, climatology30yrs, np.nan) #shape (365, 14, 1442)

    # Write to dataset
    # dict_vars = {}
    # dict_vars['relative_threshold'] = (["years", "day","nz", "xi_rho"], relative_threshold)
    # dict_vars['climatology'] = (["day","nz", "xi_rho"], climatology30yrs)

    ds_rel_threshold = xr.Dataset(
        data_vars= dict(relative_threshold = (["day", "z_rho", "xi_rho"], relative_threshold)),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            ),
        attrs = {'relative_threshold': 'Daily climatological relative threshold (90th percentile) - computed using a seasonally varying 11‐day moving window '
            }
        ) 

    ds_clim = xr.Dataset(
        data_vars= dict(climSST =(["day", "z_rho", "xi_rho"], climatology30yrs)),
        coords=dict(
            lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
            lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
            ),
        attrs = {'climatology': 'Daily climatology SST - median value obtained from a seasonally varying 11‐day moving window - baseline 1980-2009 (30yrs)'}
        ) 

    
    # Save output
    output_file_clim = os.path.join(output_path_clim, f"clim_{ieta}.nc")
    if not os.path.exists(output_file_clim):
        ds_clim.to_netcdf(output_file_clim, mode='w')  

    output_file_thresh = os.path.join(output_path_clim, f"thresh_90perc_{ieta}.nc")
    if not os.path.exists(output_file_thresh):
        ds_rel_threshold.to_netcdf(output_file_thresh, mode='w')  
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing time for eta {ieta}: {elapsed_time:.2f} seconds")

    return climatology30yrs, relative_threshold

# Calling function
results = Parallel(n_jobs=30)(delayed(calculate_climSST)(ieta, baseline) for ieta in range(0, neta)) # Computing time per eta ~15-20s,  in total ~4min
# ---> error to fix: <ipython-input-3-bbb4c30461f8>:36: RuntimeWarning: All-NaN slice encountered


# %% Merging all eta in same dataset
ds_clim, ds_rel_threshold = zip(*results)

# Datasets initialization 
nz = ds_rel_threshold[0].shape[1]
clim_sst_surf = np.full((ndays, nz, neta, nxi), np.nan, dtype=np.float32) #dim (365, 14, 434, 1442)
relative_threshold_surf =  np.full((ndays, nz, neta, nxi), np.nan, dtype=np.float32) #dim (365, 14, 434, 1442)

# Loop over neta and write all eta in same Dataset - aggregation 
for ieta in range(0, neta):
    clim_sst_surf[:, :, ieta, :] = ds_clim[ieta]  
    relative_threshold_surf[:, :, ieta, :] = ds_rel_threshold[ieta]

# Reformating
ds_clim_sst_surf = xr.Dataset(
    data_vars=dict(clim_temp = (["days", "z_rho", "eta_rho", "xi_rho"], clim_sst_surf)),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Daily climatology SST - median value obtained from a seasonally varying 11‐day moving window - baseline 1980-2009 (30yrs)'),
        ) 

ds_rel_threshold_surf = xr.Dataset(
    data_vars=dict(relative_threshold = (["days", "z_rho", "eta_rho", "xi_rho"], relative_threshold_surf)),
    coords=dict(
        lon_rho=(["eta_rho", "xi_rho"], ds_roms.lon_rho.values), #(434, 1442)
        lat_rho=(["eta_rho", "xi_rho"], ds_roms.lat_rho.values), #(434, 1442)
        ),
    attrs=dict(description='Daily climatological relative threshold (90th percentile) - computed using a seasonally varying 11‐day moving window '),
        ) 

# Save outputs
# output_file_clim_sst = os.path.join(output_path_clim, f"climSST_surf.nc")
output_file_clim_sst = os.path.join(output_path_clim, f"clim_temp_100m.nc")
if not os.path.exists(output_file_clim_sst):
    ds_clim_sst_surf.to_netcdf(output_file_clim_sst, mode='w') 

# output_file_rel_threshold = os.path.join(output_path_clim, f"threshold_90perc_surf.nc")
output_file_rel_threshold = os.path.join(output_path_clim, f"threshold_90perc_100m.nc")
if not os.path.exists(output_file_rel_threshold):
    ds_rel_threshold_surf.to_netcdf(output_file_rel_threshold, mode='w') 

#%% ---------------------------------------------------------- PLOTS (slide) ----------------------------------------------------------
# Settings and data
choice_eta = 200
choice_xi = 1000

# Surface temperature for 1 location (eta, xi)
temp_surf = xr.open_dataset(path_mhw + file_var + 'eta' + str(choice_eta) + '.nc')[var][1:31, 0:365, 0, :]  # 30yrs - 365days per year
selected_temp_surf = temp_surf.sel(xi_rho=choice_xi)

# Climatology computed above
clim_sst_100m = xr.open_dataset(output_path_clim +  'clim_temp_100m.nc' )['clim_temp']
selected_clim_sst_100m = xr.open_dataset(output_path_clim +  'clim_'+ str(choice_eta)+ '.nc')['climSST']
lon = clim_sst_100m.sel(eta_rho=choice_eta, xi_rho=choice_xi).lon_rho.item()
lat = clim_sst_100m.sel(eta_rho=choice_eta, xi_rho=choice_xi).lat_rho.item()
value = clim_sst_100m.sel(eta_rho=choice_eta, xi_rho=choice_xi, days=230, z_rho=0).item()

#%% --- Linear SST time serie for 30yrs
temp_toplot = selected_temp_surf.values # shape: (30, 365)
days = selected_temp_surf['day'].values

cmap = cm.OrRd
norm = mcolors.Normalize(vmin=0, vmax=30)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

fig, ax = plt.subplots(figsize=(10, 5))#, constrained_layout= True) 
for i in range(30):
    ax.plot(days, temp_toplot[i], color= cmap(norm(i)))

ax.set_xlabel(r'Day of the Year')
ax.set_ylabel(r'SST ($^{\circ}\textnormal{C}$)')
# ax.set_title(rf'Sea Surface Temperature (from 1980 to 2009) \nLocation: ({round(lat)}°S, {round(lon)}°E)')
ax.set_title(rf'\noindent Sea Surface Temperature (1980--2009), Location: ({lat:.0f}°S, {lon:.0f}°E)')#.format(abs(lat), lon))
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.99)
cbar.set_label(r'Year', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/time_serie_SST_30yrs.eps'), format='eps') #image vectorielle

#%% --- Circular time serie
clim_to_plot = selected_clim_sst_100m.sel(xi_rho=choice_xi, z_rho=0)
thresh_to_plot = xr.open_dataset(output_path_clim +  'thresh_90perc_'+ str(choice_eta)+ '.nc')['relative_threshold'].sel(xi_rho=choice_xi, z_rho=0)
temp_toplot = selected_temp_surf.values # shape: (30, 365)

theta = 2 * np.pi * (np.arange(0, 365) / 365)  # Convert days to angles (0 to 2π)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
cmap = cm.OrRd
norm = mcolors.Normalize(vmin=0, vmax=30)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)

for i in range(30):
    temp = temp_toplot[i]  # (365,)
    ax.plot(theta, temp, color=cmap(norm(i)), linewidth=1.5, alpha=0.8)

ax.plot(theta, clim_to_plot.values, color= '#3A6EA5', linewidth=3, linestyle='--', label= 'Climatology (median)')
ax.plot(theta, thresh_to_plot.values, color= 'black', linewidth=3, label= 'Rel. Thresh (90th perc)')

ax.set_ylim(np.nanmin(selected_temp_surf), np.nanmax(selected_temp_surf))
month_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(month_angles)
ax.set_xticklabels(month_labels, fontsize=10)

ax.set_ylabel(r'SST ($^{\circ}\textnormal{C}$)', fontsize=15, labelpad=20)
ax.set_title(f'Sea Surface Temperature (1980 -- 2009)\nLocation: ({round(lat)}°S, {round(lon)}°E)', va='bottom', fontsize=14)
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, shrink=0.9)
cbar.set_label(r'Year', fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.legend(loc='lower right', bbox_to_anchor =(1.1, -0.15))
plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/time_serie_SST_30yrs_circular.eps'), format='eps') #image vectorielle


# %% --- Moving window explanation
# Day to center the window
center_day = 75 #320
window_start = center_day - 5
window_end = center_day + 5

# Fixed median and threshold
clim_to_plot = selected_clim_sst_100m.isel(xi_rho=choice_xi, day=slice(window_start, window_end+1), z_rho=0)
thresh_to_plot = xr.open_dataset(output_path_clim +  'thresh_90perc_'+ str(choice_eta)+ '.nc')['relative_threshold'].sel(xi_rho=choice_xi, day=slice(window_start, window_end+1), z_rho=0)

clim_value_center = selected_clim_sst_100m.sel(xi_rho=choice_xi).isel(day=center_day, z_rho=0).values
thresh_value_center = xr.open_dataset(output_path_clim +  'thresh_90perc_'+ str(choice_eta)+ '.nc')['relative_threshold'].isel(xi_rho=choice_xi, day=center_day, z_rho=0).values

# --- Plot ---
plot = 'slides' # report slides
# --- Layout Setup ---
if plot == 'report':
    fig_width = 6.3228348611  # half textwidth in inches
    fig_height = fig_width / 2
    fig = plt.figure(figsize=(fig_width, fig_height))
else:  # slides
    fig_width = 10
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))

ax = fig.add_subplot(1, 1, 1)

# --- Font size settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}
ticklabelsize = 13 if plot == 'slides' else 9

# Plot SST
for i in range(30):
    ax.plot(selected_temp_surf.isel(year=i)['day'], selected_temp_surf.isel(year=i), color= cmap(norm(i)))

# Shaded window (±5 days around center day)
ax.axvspan(window_start, window_end, color='#98C4D7', alpha=0.3, label='11day window')

# Horizontal lines representing "median" and "90th percentile threshold"
# ax.plot(np.arange(window_start, window_end+1), clim_to_plot, color='#3A6EA5', linestyle='--', linewidth = 1)
# ax.plot(np.arange(window_start, window_end+1), thresh_to_plot, color='black', linestyle='-', linewidth = 1)
if plot=='report':
    label1='Climatology'
    label2='Rel. Thresh'
else:
    label1='Climatology (median)'
    label2='Rel. Thresh (90th perc)'
ax.scatter(center_day, clim_value_center, color='#3A6EA5', edgecolor='white', zorder=5, s=50, label=label1)
ax.scatter(center_day, thresh_value_center, color='black', edgecolor='white', zorder=5, s=50, label=label2)

ax.set_xlabel(r'Day of the Year', **label_kwargs)
ax.set_ylabel(r'Temperature [$^\circ$C]', **label_kwargs)
ax.set_title('Illustration of 11days moving window calculation',  **maintitle_kwargs)#\nLocation: ({round(lat)}°S, {round(lon)}°E)')
ax.set_xlim(center_day - 20, center_day +20)
ax.set_ylim(-1,5)

# Set x-ticks at desired locations and ensure center_day is included
tick_positions = [tick for tick in ax.get_xticks() if center_day - 20 <= tick <= center_day + 20]
tick_labels = [f"{int(tick)}" for tick in tick_positions]

# Highlight the three specific days: (i-5), (i), (i+5)
highlight_ticks = [center_day - 5, center_day, center_day + 5]
for tick in highlight_ticks:
    if tick not in tick_positions:
        tick_positions.append(tick)

# Set the x-ticks and labels
ax.set_xticks(tick_positions)  # Set x-ticks explicitly

# Set custom labels for the highlighted ticks: "i-5", "i", "i+5"
tick_labels = []
for tick in tick_positions:
    if tick == center_day - 5:
        tick_labels.append("i -5")
    elif tick == center_day:
        tick_labels.append("i")
    elif tick == center_day + 5:
        tick_labels.append("i+5")
    else:
        tick_labels.append(str(int(tick)))

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=ticklabelsize)

# Highlight the specific ticks in red
for i, tick in enumerate(tick_positions):
    if tick in highlight_ticks:
        ax.get_xticklabels()[i].set_color('red')  # Change the label color to red

# Plot month lines and labels
month_days = {
    'Jan': 1, 'Feb': 32, 'March': 60, 'April': 91, 'May': 120, 'Jun': 151, 
    'Jul': 182, 'Aug': 213, 'Sept': 244, 'Oct': 274, 'Nov': 305, 'Dec': 335
}
month_days_filtered = {month: day for month, day in month_days.items() if center_day - 20 <= day <= center_day + 20}
for month, day in month_days_filtered.items():
    ax.axvline(day, color='#014F86', linestyle='--', alpha=0.9, linewidth=2 if plot == 'slides' else 1)
    ax.text(day-0.4, ax.get_ylim()[1]-1.05, month, rotation=90, verticalalignment='bottom', horizontalalignment='center', color='#014F86',   **legend_kwargs)

if plot=='report':
    box_legend = (0.117, 0.03)
    ncol=1
else:
    box_legend = (0.1, -0.25)
    ncol=3

ax.legend(loc='lower left', bbox_to_anchor=box_legend, ncol=ncol,  **legend_kwargs)

# --- Output handling ---
if plot == 'report':
    plt.tight_layout()
    outdir = os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/climatology and 90th percentile')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"11days_window_illustration_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/climatology and 90th percentile/11days_window_illustration_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %% Illustrtion 
# --- Plot ---
plot = 'report' # report slides
# --- Layout Setup ---
if plot == 'report':
    fig_width = 6.3228348611  # half textwidth in inches
    fig_height = fig_width / 2
    fig = plt.figure(figsize=(fig_width, fig_height))
else:  # slides
    fig_width = 10
    fig_height = 5
    fig = plt.figure(figsize=(fig_width, fig_height))

ax = fig.add_subplot(1, 1, 1)

# --- Font size settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 12} if plot == 'slides' else {}
ticklabelsize = 13 if plot == 'slides' else 9

# --- Data Setup ---
np.random.seed(42)
days = np.arange(1, 70)
climatology = 2 + np.sin(3 * np.pi * (days - 1) / 365)
threshold = climatology + 0.75

# Start with climatology + noise
temperature = climatology + np.random.normal(0, 0.3, size=len(days))

# --- Heat Events ---
temperature[10:13] = threshold[10:13] + np.random.uniform(0.1, 0.4, size=3) # Short heat spike: 3 days, above threshold
temperature[40:47] = threshold[40:47] + np.random.uniform(0.1, 0.5, size=7) # MHW: 7 days ≥ threshold with varying values

# Climatology and threshold
lw = 1 if plot == 'slides' else 1
ax.plot(days, climatology, color='#3A6EA5', label='Climatology', linewidth=lw)
ax.plot(days, threshold, color='black', linestyle='--', label='Relative Threshold', linewidth=lw)

# Temperature time series
ax.plot(days, temperature, color='#DC2F02', linewidth=lw, label='Daily Temperature')


# --- Formatting ---
ax.set_xlabel('Day of Year', **label_kwargs)
ax.set_ylabel('Temperature [°C]', **label_kwargs)
ax.set_title('Schematic to define a Marine HeatWave (MHW)', **label_kwargs)
ax.set_xlim(0, 70)
ax.set_ylim(1.5, 4.5)
# ax.tick_params(labelsize=ticklabelsize)
ax.tick_params(axis='both', direction='in', labelsize=ticklabelsize)
ax.set_xticklabels([])  # Remove x-axis tick labels
ax.set_yticklabels([])  # Remove y-axis tick labels
ax.grid(True, linestyle=':', alpha=0.4)
if plot == 'report':
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=3,
              frameon=True,
              columnspacing=1.5,
              handletextpad=0.5,
              **legend_kwargs)
else:
    ax.legend(loc='upper left', frameon=True, **legend_kwargs)


# --- Output handling ---
if plot == 'report':
    plt.tight_layout()
    outdir = os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/climatology and 90th percentile')
    os.makedirs(outdir, exist_ok=True)
    outfile = f"MHW_schematic_{plot}.pdf"
    plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    # plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Marine_HeatWaves/figures_outputs/climatology and 90th percentile/MHW_schematic_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()




# %% --- SST above threshold plot
ds_yr_example = xr.open_dataset(path_mhw + file_var + 'eta' + str(choice_eta) + '.nc')[var].isel(year=21, z_rho=0, xi_rho=choice_xi)
thresh_to_plot = xr.open_dataset(output_path_clim +  'thresh_90perc_'+ str(choice_eta)+ '.nc')['relative_threshold'].isel(xi_rho=choice_xi, z_rho=0)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ds_yr_example['day'], ds_yr_example, color= 'darkred', label='SST')
ax.plot(thresh_to_plot['day'], thresh_to_plot, color= 'black', label='90th perc')

# Condition: temp>90th perc
condition = ds_yr_example.values >= thresh_to_plot
sst_masked = np.where(condition, ds_yr_example.values, np.nan)
threshold_masked = np.where(condition, thresh_to_plot, np.nan)

ax.fill_between(ds_yr_example.day.values, sst_masked, threshold_masked,
                color='#FF9F1C', alpha=0.7) 

ax.set_xlabel('Day of the Year')
ax.set_ylabel('SST (°C)')
ax.set_title(f'Example of SST and relative threshold\nLocation: ({round(ds_yr_example.lat_rho.values.item())}°S, {round(ds_yr_example.lon_rho.values.item())}°E) in {ds_yr_example.year.values}')
# ax.set_ylim(0, 4.2)
# ax.set_xlim(250,350)
ax.legend(loc='upper right', fontsize=10, bbox_to_anchor = (1, 1))
plt.tight_layout()
plt.show()

# plt.savefig(os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/climatology and 90th percentile/example_SST_90thperc.pdf'), format='pdf', bbox_inches='tight') #image vectorielle

# %% --- Climatology map
fn = output_path_clim +  'climSST_surf.nc' #dim: (days: 365, z_rho: 35, eta_rho: 434, xi_rho: 1442)
clim_sst_surf = xr.open_dataset(fn)['clim_sst']

thresh_to_plot = xr.open_dataset(output_path_clim +  'threshold_90perc_surf.nc')['relative_threshold']

day_to_plot= 100
year_to_plot = 1990

temp_DC_BC_surface = xr.open_dataset( os.path.join(path_mhw, f"temp_DC_BC_surface.nc"))[var][1:, 0:365, :, :]  # 30yrs - 365days per year
temp_DC_BC_surface_1990 = temp_DC_BC_surface.sel(year=year_to_plot)

lon = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi).lon_rho.item()
lat = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi).lat_rho.item()

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-90, central_longitude=0))
ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

# Circular map boundary
theta = np.linspace(0, 2 * np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# -------- Plot Clim --------
# pcolormesh = clim_sst_surf.sel(days=day_to_plot).plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(),
#     x="lon_rho", y="lat_rho",
#     add_colorbar=False, 
#     vmin=-5, vmax=5,
#     cmap='coolwarm'
# )

# -------- Plot rel threshold --------
# pcolormesh = thresh_to_plot.sel(days=day_to_plot).plot.pcolormesh(
#     ax=ax, transform=ccrs.PlateCarree(),
#     x="lon_rho", y="lat_rho",
#     add_colorbar=False, 
#     vmin=-5, vmax=5,
#     cmap='coolwarm'
# )

# -------- Plot SST (ROMS output) --------
pcolormesh = temp_DC_BC_surface_1990.isel(day=day_to_plot).plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(),
    x="lon_rho", y="lat_rho",
    add_colorbar=False, 
    vmin=-5, vmax=5,
    cmap='coolwarm'
)

# -------- Plot scatter point on top --------
# value = clim_sst_surf.sel(eta_rho=choice_eta, xi_rho=choice_xi, days=day_to_plot).item()
# value = thresh_to_plot.sel(eta_rho=choice_eta, xi_rho=choice_xi, days=day_to_plot).item()
value = temp_DC_BC_surface_1990.sel(eta_rho=choice_eta, xi_rho=choice_xi, day=day_to_plot).item()

sc = ax.scatter(lon, lat, c=[value], cmap='coolwarm', vmin=-5, vmax=5,
                transform=ccrs.PlateCarree(), s=100, edgecolor='black', zorder=3, label='Selected Cell')

# -------- Add colorbar --------
cbar = plt.colorbar(pcolormesh, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
cbar.set_label('°C', fontsize=13)
cbar.ax.tick_params(labelsize=12)

# -------- Add map features --------
ax.coastlines(color='black', linewidth=1.5, zorder=1)
ax.add_feature(cfeature.LAND, zorder=2, facecolor='lightgray')
ax.set_facecolor('lightgrey')

# -------- Title --------
# ax.set_title(f"SST median climatolgy \nLocation: ({round(lat)}°S, {round(lon)}°E), $day_{{{day_to_plot}}}$", fontsize=16, pad=30)
ax.set_title(f"Sea Surface Temperature, day$_{{{day_to_plot}}}$, y$_{{{year_to_plot}}}$", fontsize=16, pad=30)
# ax.set_title(f"ROMS output SST in {temp_DC_BC_surface_1990.year.values.item()} \nLocation: ({round(lat)}°S, {round(lon)}°E), $day_{{{day_to_plot}}}$", fontsize=16, pad=30)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/climatology and 90th percentile/example_SST_map.pdf'), format='pdf', bbox_inches='tight') #image vectorielle


# %% Vertical profile temperature for selected location on graph before
temp_DC_BC_100m= xr.open_dataset(os.path.join(path_mhw, f"temp_DC_BC_eta200.nc"))[var][1:, 0:365, :, :]  # 30yrs - 365days per year
temp_profile_1990 = temp_DC_BC_100m.isel(year=year_to_plot-1980, xi_rho=choice_xi, day=day_to_plot)

# Plot
plt.figure(figsize=(4, 6))
plt.hlines(-100, xmin= temp_profile_1990.min()-0.1, xmax=temp_profile_1990.max()+0.1, color='black', linewidth=2)

plt.plot(temp_profile_1990, temp_DC_BC_100m['z_rho'], marker='o', markersize=3, color='firebrick')
plt.xlabel('Temperature (°C)')
plt.ylabel('Depth (m)')
plt.xlim(temp_profile_1990.min()-0.1, temp_profile_1990.max()+0.1)
plt.ylim(-505, 1)
# plt.title(f'Temperature Profile\n1990 - Day {day_to_plot}, Loc ({choice_eta}, {choice_xi})')
# plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(os.getcwd(), 'Marine_HeatWaves/figures_outputs/climatology and 90th percentile/example_temp_profile.pdf'), format='pdf', bbox_inches='tight') #image vectorielle

# %%
import pandas as pd

# --- Load temperature data
temp_daily = xr.open_dataset(path_mhw + file_var + 'eta' + str(200) + '.nc')[var][1:, 0:365, 0, 1000]  # shape: (40, 365)

# Flatten temp to 1D: (14600,)
temp_flat = temp_daily.values.reshape(-1)

# --- Load threshold: shape (365,) → tile to match temp
rel_thresh = xr.open_dataset('/nfs/sea/work/mlarriere/mhw_krill_SO/clim30yrs/threshold_90perc_surf.nc')['relative_threshold'][:, 200, 1000].values
thresh_tiled = np.tile(rel_thresh, 40)  # (365,) → (14600,)

# --- Build datetime index
start_year = temp_daily['year'].values[0]
dates = pd.date_range(start=f"{start_year}-01-01", periods=14600, freq='D')

# --- Create mask
mask_above_thresh = temp_flat > thresh_tiled

# --- Plot
plt.figure(figsize=(13, 4))
plt.plot(dates, temp_flat, color='black', linewidth=0.5, label='Daily Temp')

# Highlight temps > threshold in red
plt.scatter(dates[mask_above_thresh], temp_flat[mask_above_thresh], color='red', s=1, label='Above 90th Percentile')

# Threshold line
plt.plot(dates, thresh_tiled, color='purple', linestyle='-', linewidth=0.8, label='90th Percentile Threshold')

# Title and axis labels
plt.title("Daily Temperature Time Series (Surface) at (eta=200, xi=1000)")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")

# Show only 2014–2019
plt.xlim(pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31"))

# Legend and grid
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
