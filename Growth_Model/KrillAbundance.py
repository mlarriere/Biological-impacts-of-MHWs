"""
Created on Tue 22 July 09:00:36 2025

Investigation of KRILLBASE dataset

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
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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


# %% ============================================ Load data ============================================
# path_AtlanECO_grid = '/nfs/sea/public/ftp/AtlantECO/GRID'
# file_Euphausia_grid = 'AtlantECO-GRID-v1_microbiome_traditional_Euphausiacea_abund+biomass_20220930.nc'
# atlanEco_grid_ds=xr.open_dataset(os.path.join(path_AtlanECO_grid, file_Euphausia_grid), decode_times=False)


path_AtlanECO_base = '/nfs/sea/public/ftp/AtlantECO/BASE'
file_Euphausia_base = 'AtlantECO-BASE-v1_microbiome_traditional_Euphausiacea_abund+biomass_20221220.csv'
atlanEco_euphausia = pd.read_csv(os.path.join(path_AtlanECO_base, file_Euphausia_base), encoding='ISO-8859-1')

# ---- Remove CPR data (abundance not reliable)
keywords = ['270mm', 'CPR', 'Richardson'] # Keywords to filter out

# Check for any row containing either keyword
matches = {}
for col in atlanEco_euphausia.columns:
    if atlanEco_euphausia[col].dtype == object:
        mask = atlanEco_euphausia[col].astype(str).str.contains('|'.join(keywords), case=False, na=False)
        if mask.any():
            matches[col] = atlanEco_euphausia.loc[mask, col].unique().tolist()

for col, values in matches.items():
    print(f"\n🟩 Match found in column: {col}")
    for v in values:
        print(f"  → {v}")

# Remove from dataset
mask = pd.Series(False, index=atlanEco_euphausia.index)
for col in atlanEco_euphausia.columns:
    if atlanEco_euphausia[col].dtype == object:
        col_mask = atlanEco_euphausia[col].astype(str).str.contains('|'.join(keywords), case=False, na=False)
        mask |= col_mask  # combine with OR logic

# Drop rows
atlanEco_euphausia_cleaned = atlanEco_euphausia[~mask].copy()
print(f"Original dataset: {atlanEco_euphausia.shape}") #shape (1071450, 85)
print(f"Cleaned dataset : {atlanEco_euphausia_cleaned.shape}") #shape (35564, 85)

#%% ============================================ Select data of interest ============================================
# ---- Southern Ocean (south of 60°S)
month_mask = (atlanEco_euphausia_cleaned['Month'] >= 11) | (atlanEco_euphausia_cleaned['Month'] <= 4)
atlanEco_euphausia_filtered_SO = atlanEco_euphausia_cleaned[
    (atlanEco_euphausia_cleaned['decimalLatitude'] < -60) & # Extract only data south of 60°S
    (atlanEco_euphausia_cleaned['Year'] >= 1980) & # Keep only data from 1980 to 2019
    (atlanEco_euphausia_cleaned['Year'] <= 2019) &
    (month_mask) & # Select only date between 1st november and 30th April
    (atlanEco_euphausia_cleaned['ScientificName'] == 'Euphausia superba')
]

# Valid abundance
e_superba_abundance_SO = atlanEco_euphausia_filtered_SO[atlanEco_euphausia_filtered_SO['MeasurementUnit'].str.strip().str.lower() == 'ind m-3'].copy()
e_superba_abundance_SO['MeasurementValue'] = pd.to_numeric(e_superba_abundance_SO['MeasurementValue'], errors='coerce')
e_superba_abundance_SO = e_superba_abundance_SO.dropna(subset=['MeasurementValue']) # Drop Na values

# Areal density [ind/m²] = Volume density [ind/m³] × Sampling depth [m]
e_superba_abundance_SO['abundance_m2'] = e_superba_abundance_SO['MeasurementValue'] * e_superba_abundance_SO['MeanDepth']


# ---- Atlantic Sector (0-90°W)
month_mask = (atlanEco_euphausia_cleaned['Month'] >= 11) | (atlanEco_euphausia_cleaned['Month'] <= 4)
atlanEco_euphausia_filtered_Atl = atlanEco_euphausia_cleaned[
    (atlanEco_euphausia_cleaned['decimalLatitude'] < -60) & # Extract only data south of 60°S
    (atlanEco_euphausia_cleaned['decimalLongitude'] >= -100) & # Select only data in the Atlantic sector (270°E to 360°E)
    (atlanEco_euphausia_cleaned['decimalLongitude'] <= 0) &
    (atlanEco_euphausia_cleaned['Year'] >= 1980) & # Keep only data from 1980 to 2019
    (atlanEco_euphausia_cleaned['Year'] <= 2019) &
    (month_mask) & # Select only date between 1st november and 30th April
    (atlanEco_euphausia_cleaned['ScientificName'] == 'Euphausia superba')
]

# Valid abundance
e_superba_abundance_Atl = atlanEco_euphausia_filtered_Atl[atlanEco_euphausia_filtered_Atl['MeasurementUnit'].str.strip().str.lower() == 'ind m-3'].copy()
e_superba_abundance_Atl['MeasurementValue'] = pd.to_numeric(e_superba_abundance_Atl['MeasurementValue'], errors='coerce')
e_superba_abundance_Atl = e_superba_abundance_Atl.dropna(subset=['MeasurementValue']) # Drop Na values

# Areal density [ind/m²] = Volume density [ind/m³] × Sampling depth [m]
e_superba_abundance_Atl['abundance_m2'] = e_superba_abundance_Atl['MeasurementValue'] * e_superba_abundance_Atl['MeanDepth']

# %% ============================================ Invsetigate data ============================================
min_abundance = e_superba_abundance_Atl['abundance_m2'].min() # 0 ind/m2
max_abundance = e_superba_abundance_Atl['abundance_m2'].max() # 4846.107  ind/m2
mean_abundance = e_superba_abundance_Atl['abundance_m2'].mean() # 17.854926365124584 ind/m2
print(f"Minimum krill abundance: {min_abundance}")
print(f"Mean krill abundance: {mean_abundance}")
print(f"Maximum krill abundance: {max_abundance}")

plt.figure(figsize=(8, 5))
plt.hist(e_superba_abundance_Atl['abundance_m2'], bins=50, log=True, color='skyblue', edgecolor='gray')
plt.axvline(mean_abundance, color='red', linestyle='dashed', linewidth=1, label=f'Mean = {mean_abundance:.2f}')
plt.xlabel('Krill Abundance (ind/m2)')
plt.ylabel('Frequency (log scale)')
plt.title('Distribution of Euphausia superba Abundance (0-90°W)')
plt.legend()
plt.tight_layout()
plt.show()

# %% ============================================ Mean Values and Number of Samples ============================================
# --- Stats Atlantic Sector
annual_stats_Atl = (e_superba_abundance_Atl.groupby('Year').agg(mean_abundance=('abundance_m2', 'mean'), sample_count=('abundance_m2', 'count')).reset_index())
annual_stats_atl = annual_stats_Atl.dropna()

# ---- Plot ----
# --- Plot layout ---
plot = 'report'  # 'report' or 'slides'

fig_width = 6.32 if plot == 'report' else 15
fig_height = fig_width / 2 if plot == 'report' else 6

fig, ax1 = plt.subplots(figsize=(fig_width, fig_height), sharey=True)

# --- Font settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 13}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 10}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}

# Plot mean abundance (left y-axis)
color1 = '#367DD3'
ax1.set_xlabel('Year', **label_kwargs)
ax1.set_ylabel('Mean Abundance (ind/m²)', color=color1, **label_kwargs)
ms = 4 if plot == 'report' else 6  # marker size
ax1.plot(annual_stats_atl['Year'], annual_stats_atl['mean_abundance'], marker='o', color=color1, markersize=ms, label='Annual Mean')
lw = 1 if plot == 'report' else 2
ax1.axhline(e_superba_abundance_Atl['abundance_m2'].mean(), color=color1, linestyle='--', linewidth=lw, alpha=0.9, label='Overall Mean')
so_mean = e_superba_abundance_SO['abundance_m2'].mean()
line_so = ax1.axhline(so_mean, color='#A30000', linestyle='--', linewidth=lw, alpha=0.9)
ax1.tick_params(axis='y', labelcolor=color1, **tick_kwargs)
ax1.tick_params(axis='x', **tick_kwargs)
# ax1.grid(True, alpha=0.3)

# Twin axis for sample count
ax2 = ax1.twinx()
color2 = '#C2C2C2'
ax2.set_ylabel('Number of Samples', color=color2, **label_kwargs)
ax2.bar(annual_stats_atl['Year'], annual_stats_atl['sample_count'], color=color2, alpha=0.4, edgecolor='#BFC0C0', label='Sample Count')
ax2.tick_params(axis='y', labelcolor=color2, **tick_kwargs)

# Put ax1 above ax2
ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_visible(False)  # Hide ax1 background to avoid covering ax2 bar plot

# --- Legend
# --- Separate legend for Mean Southern Ocean
separate_legend = ax1.legend(
    [line_so], ['Southern Ocean'],
    loc='upper right',  # or (0.95, 0.95) with bbox_to_anchor
    frameon=True,
    fontsize=legend_kwargs.get('fontsize', 10),
    framealpha=0.8,
)

ax1.add_artist(separate_legend)  # ensures this legend appears along with the other

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
if plot=='slides':
    box_legend=(0.5, -0.05)
if plot=='report':
    box_legend=(0.5, -0.09)
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=True, fontsize=legend_kwargs.get('fontsize', 10),
           bbox_to_anchor=box_legend, borderaxespad=0, handletextpad=0.5)

# --- Title 
var_bigtitle = 'Euphausia superba: Annual Mean Abundance and Sample Count'
if plot == 'report':
    suptitle_y = 1
    fig.suptitle(f'{var_bigtitle}', y=suptitle_y, x=0.5, **maintitle_kwargs)
    fig.text(0.5, suptitle_y - 0.09, 'Atlantic Sector (0-90°W); 1980-2016', ha='center', **label_kwargs, style='italic')

else:
    suptitle_y = 0.99
    fig.suptitle(f'{var_bigtitle}', y=suptitle_y, x=0.5, **maintitle_kwargs)
    fig.text(0.5, suptitle_y - 0.09, 'Atlantic Sector (0-90°W); 1980-2016', ha='center', **label_kwargs, style='italic')

# --- Output handling ---
outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/Biomass/Abundance')
os.makedirs(outdir, exist_ok=True)
if plot == 'report':
    outfile = f"abundance_distribution_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/Biomass/Abundance/abundance_distribution_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %% ============================================ Years with Higher Mean Values ============================================
data_1982 = e_superba_abundance_SO[e_superba_abundance_SO['Year'] == 1982]
data_2012 = e_superba_abundance_SO[e_superba_abundance_SO['Year'] == 2012]

# --- Compute Statistics
mean_1982 = data_1982['abundance_m2'].mean()
mean_2012 = data_2012['abundance_m2'].mean()

print(f"1985 → N = {len(data_1982)} | Mean = {mean_1982:.2f} ind/m²")
print(f"2012 → N = {len(data_2012)} | Mean = {mean_2012:.2f} ind/m²")

# ---- Plot
# --- Plot layout ---
plot = 'report'  # 'report' or 'slides'

fig_width = 6.32 if plot == 'report' else 15
fig_height = fig_width / 2 if plot == 'report' else 6

fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

# --- Font settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 13}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 10}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 9}

# ---- 1982 ----
axes[0].hist(data_1982['abundance_m2'], bins=30, color='#9B2226', log=True)
axes[0].axvline(mean_1982, color='red', linestyle='--', linewidth=1, label=f'Mean = {mean_1982:.2f}')
axes[0].set_title('Year 1982', **subtitle_kwargs)
axes[0].set_xlabel('Abundance (ind/m²)', **label_kwargs)
axes[0].set_ylabel('Frequency (log scale)', **label_kwargs)
axes[0].legend(**legend_kwargs)
axes[0].tick_params(**tick_kwargs)

# ---- 2012 ----
axes[1].hist(data_2012['abundance_m2'], bins=30, color='#CA6702', log=True)
axes[1].axvline(mean_2012, color='red', linestyle='--', linewidth=1, label=f'Mean = {mean_2012:.2f}')
axes[1].set_title('Year 2012', **subtitle_kwargs)
axes[1].set_xlabel('Abundance (ind/m²)', **label_kwargs)
axes[1].legend(**legend_kwargs)
axes[1].tick_params(**tick_kwargs)

# --- Title ---
var_bigtitle = 'Abundance Distribution of Euphausia superba'
if plot == 'report':
    suptitle_y = 1.1
    fig.suptitle(f'{var_bigtitle}', y=suptitle_y, x=0.5, **maintitle_kwargs)
    fig.text(0.5, suptitle_y - 0.1, 'Atlantic Sector, Growth Season (1980-2016)', ha='center', **label_kwargs, style='italic')

else:
    suptitle_y = 1.05
    fig.suptitle(f'{var_bigtitle}', y=suptitle_y, x=0.5, **maintitle_kwargs)
    fig.text(0.5, suptitle_y - 0.09, 'Atlantic Sector, Growth Season (1980-2016)', ha='center', **label_kwargs, style='italic')

# --- Output handling ---
outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/Biomass/Abundance')
os.makedirs(outdir, exist_ok=True)
if plot == 'report':
    outfile = f"high_abundance_{plot}.pdf"
    # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:    
    # plt.savefig(os.path.join(os.getcwd(), f'Growth_Model/figures_outputs/Biomass/Abundance/high_abundance_{plot}.png'), dpi=500, format='png', bbox_inches='tight')
    plt.show()


# %% ============================================ Plot ============================================
# --- Convert date ---
e_superba_abundance_SO['DATE'] = pd.to_datetime(e_superba_abundance_SO['eventDate'], errors='coerce')

# --- Define combined Nov–Apr growth season across all years ---
def select_all_growth_seasons(data, start_year=1980, end_year=2019):
    mask = (
        (data['DATE'].dt.year >= start_year) & (data['DATE'].dt.year <= end_year) &
        ((data['DATE'].dt.month >= 11) | (data['DATE'].dt.month <= 4))
    )
    return data[mask].copy()

# --- Filter data ---
data_plot = select_all_growth_seasons(e_superba_abundance_SO)

if data_plot.empty:
    print("No data found for growth seasons.")
else:
    # --- Output format ---
    plot = 'slides'  # report 'slides'

    # --- Figure dimensions ---
    fig_width = 6.32 if plot == 'report' else 10
    fig_height = 5.5 if plot == 'report' else 8

    # --- Font and style kwargs ---
    maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 13}
    subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {'fontsize': 10}
    label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
    tick_kwargs = {'labelsize': 13} if plot == 'slides' else {}
    legend_kwargs = {'fontsize': 14} if plot == 'slides' else {'fontsize': 10}
    gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}

    # --- Setup figure and axes ---
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

    # --- Circular boundary for neat polar plot ---
    theta = np.linspace(np.pi / 2, np.pi, 100)
    center, radius = [0.5, 0.51], 0.5
    arc = np.vstack([np.cos(theta), np.sin(theta)]).T
    verts = np.concatenate([[center], arc * radius + center, [center]])
    circle = mpath.Path(verts)
    ax.set_boundary(circle, transform=ax.transAxes)

    # --- Map extent and features ---
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#F6F6F3', zorder=2)  # Land should be drawn above the plot
    ax.coastlines(color='black', linewidth=1)
    ax.set_facecolor("#D8D8D8")

    # --- Abundance data for color scaling ---
    abundance = data_plot['abundance_m2'].clip(lower=1e-3)

    # --- Scatter plot ---
    sc = ax.scatter(
        data_plot['decimalLongitude'],
        data_plot['decimalLatitude'],
        c=data_plot['abundance_m2'],
        cmap='inferno',
        s=10 if plot == 'report' else 20,
        transform=ccrs.PlateCarree(),
        norm=mcolors.LogNorm(vmin=abundance.min(), vmax=abundance.max()),
        alpha=0.8,
        zorder=1,
        rasterized=True
    )

    # --- Gridlines with labels ---
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--', linewidth=0.7, zorder=3)
    # gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # --- Colorbar ---
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.6, pad=0.05, extend='both')
    cbar.ax.minorticks_off()
    cbar.set_label('Abundance [ind/m²] (log scale)', **label_kwargs)

    # --- Title ---
    var_bigtitle = 'Euphausia superba Abundance (AtlanECO)'
    subtitle = 'Growth Season (Nov–Apr), 1980–2016'

    suptitle_y = 1.02 if plot == 'slides' else 1.01
    fig.suptitle(var_bigtitle, y=suptitle_y, x=0.5, ha='center', **maintitle_kwargs)
    fig.text(0.5, suptitle_y - 0.05, subtitle, ha='center', **label_kwargs, style='italic')

    plt.tight_layout()

    # --- Output handling ---
    outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/Biomass/Abundance')
    os.makedirs(outdir, exist_ok=True)

    if plot == 'report':
        outfile = "abundance_AtlanECO_report.pdf"
        # plt.savefig(os.path.join(outdir, outfile), dpi=200, format='pdf', bbox_inches='tight')
        plt.show()
    else:
        outfile = "abundance_AtlanECO_slides.png"
        plt.savefig(os.path.join(outdir, outfile), dpi=500, format='png', bbox_inches='tight')
        # plt.show()


# %% ============================================ Sampling depth ============================================
# --- Define custom colormap: white for 0, then shades of blue
cmap = plt.get_cmap('Blues')
colors_with_white = cmap(np.linspace(0.2, 1, 256))  # skip very light values
colors_with_white = np.vstack((np.array([1, 1, 1, 1]), colors_with_white))  # prepend white for 0
custom_cmap = mcolors.ListedColormap(colors_with_white)

# --- Define bins and labels
ds = xr.open_dataset(f"{path_temp}{file_var}eta200.nc")[var][1:, 0:365, 0:, :]  # Depth ROMS
depth_bins = -ds.z_rho.values

depth_labels = [f'{depth_bins[i]}–{depth_bins[i+1]}' for i in range(len(depth_bins) - 1)]
mean_depth_clipped = e_superba_abundance_Atl['MeanDepth'].clip(lower=5, upper=500)

# --- Bin the depth
e_superba_abundance_Atl['DepthBin'] = pd.cut(
    # e_superba_abundance_Atl['MeanDepth'],
    mean_depth_clipped,
    bins=depth_bins,
    labels=depth_labels,
    include_lowest=True
)

# --- Count number of samples per (year, depth bin)
sample_counts = e_superba_abundance_Atl.groupby(['DepthBin', 'Year']).size().unstack(fill_value=0)

# --- Define normalization: 0 gets its own bin
bounds = np.arange(sample_counts.values.max() + 2)  # e.g., 0 to max+1
norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

# --- Ensure full year range (optional, for consistent plotting)
full_years = np.arange(e_superba_abundance_Atl['Year'].min(), e_superba_abundance_Atl['Year'].max() + 1)
sample_counts = sample_counts.reindex(columns=full_years, fill_value=0)

# --- Set up edges for pcolormesh
year_edges = np.arange(full_years.min(), full_years.max() + 2)  # +2 for full coverage
depth_edges = np.arange(len(depth_labels) + 1)  # +1 because we have one less bin than edges

# --- Define depth edges properly (used for bin boundaries)
depth_edges = depth_bins  # These are the actual bin edges

# --- Plot
# --- Plot layout ---
plot = 'report'  # 'report' or 'slides'

fig_width = 6.32 if plot == 'report' else 10
fig_height = fig_width/1.5 if plot == 'report' else 6

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# --- Font settings ---
maintitle_kwargs = {'fontsize': 18} if plot == 'slides' else {'fontsize': 10}
subtitle_kwargs = {'fontsize': 15} if plot == 'slides' else {}
label_kwargs = {'fontsize': 14} if plot == 'slides' else {}
ticklabelsize = 13 if plot == 'slides' else 9

# --- Pcolormesh ---
pcm = ax.pcolormesh(
    year_edges,
    np.arange(len(depth_edges) - 1 + 1),  # One extra edge for pcolormesh
    sample_counts.values,
    cmap=custom_cmap,
    norm=norm,
    shading='auto'
)

# --- Format axes ---
ax.set_title('Sampling Depths', **maintitle_kwargs)
ax.set_xlabel('Year', **label_kwargs)
ax.set_ylabel('Depth [m]', **label_kwargs)

# --- Y-axis ticks ---
step = 2 
yticks = np.arange(len(depth_edges))
ytick_labels = [f'{int(d)}' if i % step == 0 else '' for i, d in enumerate(depth_edges)]

ax.set_yticks(yticks)
ax.set_yticklabels(ytick_labels)
plt.setp(ax.get_yticklabels(), fontsize=ticklabelsize)

if plot == 'slides':
    plt.setp(ax.get_yticklabels(), rotation=0, ha='right')  # Optional for slides

ax.invert_yaxis()  # Shallow depths on top

# --- Colorbar using ScalarMappable ---
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, extend='max')
cbar.set_label('Number of Samples', **label_kwargs)
cbar.ax.tick_params(labelsize=ticklabelsize)
cbar.ax.minorticks_off()

# --- Output handling ---
outdir = os.path.join(os.getcwd(), 'Growth_Model/figures_outputs/Biomass/Abundance')
os.makedirs(outdir, exist_ok=True)

outfile = f"sampling_depth_{plot}.{'png' if plot == 'slides' else 'pdf'}"
savepath = os.path.join(outdir, outfile)

# --- Save or show ---
if plot == 'report':
    # plt.savefig(savepath, dpi=200, format='pdf', bbox_inches='tight')
    plt.show()
else:
    # plt.savefig(savepath, dpi=500, format='png', bbox_inches='tight')
    plt.show()
