"""
Created on Wedn 18 March 10:46:03 2026

Answering the question:
"How is krill biomass in the southern ocean changing during MHWs?"

@author: Marguerite Larriere (mlarriere)
"""

# %% ======================== PACKAGES ========================
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


# %% ========================================================================
#                        Change in Biomass 
#    ========================================================================
 # %% ======================== Step1. Seasonal gains for the different surrogates ========================
seasonal_gain_interp = {}
for surrog in surrogate_names.keys():
    seasonal_gain_interp[surrog] = biomass_mpas_interp[surrog].isel(days=-1) - biomass_mpas_interp[surrog].isel(days=0) #shape (39, 5, 231, 1440)

# %% ======================== Step2. Change relative to climatology per grid cell (fraction) ========================
change_interp_cell = {}
for surrog in [s for s in surrogate_names if s != "clim"]:
    change_interp_cell[surrog] = (seasonal_gain_interp[surrog] - seasonal_gain_interp['clim'])/seasonal_gain_interp['clim'] #shape (39, 5)


# %% ======================== Growth ========================
krill_growth = xr.open_dataset(os.path.join(path_growth, 'growth_Atkison2006_seasonal.nc')) #shape (39, 231, 1442, 181)
growth_clim = xr.open_dataset(os.path.join(path_growth, 'growth_Atkison2006_seasonal_clim.nc'))
growth_mhws = xr.open_dataset(os.path.join(path_growth, "growth_Atkison2006_seasonal_mhws.nc"))
growth_anom_mhws = xr.open_dataset(os.path.join(path_growth, 'growth_Atkison2006_anomaly_mhw.nc'))

# Mean growth anomalies during MHWs 
for var in growth_anom_mhws.data_vars:
    growth_anom_mhws_mean = growth_anom_mhws[var].mean(dim="days", skipna=True)

# %% ======================== Temperature data ========================
ds_mhw = xr.open_dataset(os.path.join(path_combined_thesh, f"duration_AND_thresh_5mSEASON.nc"))
main_path='/nfs/sea/work/mlarriere/mhw_krill_SO'
# climSST = xr.open_dataset(os.path.join(path_clim,'climSST_surf.nc'))

temp_surf_file = os.path.join(main_path, 'temp_surf_seasonal.nc')
temp_surf_clim_file = os.path.join(main_path, "temp_clim_surf_seasonal.nc")

if not os.path.exists(temp_surf_file) and not os.path.exists(temp_surf_clim_file):
    temp_surf= xr.open_dataset(os.path.join(main_path, f"temp_surf.nc"))
    temp_surf_clim =xr.open_dataset(os.path.join(path_clim, "climSST_surf.nc"))

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
    temp_surf_clim_seasonal = temp_surf_clim['clim_sst'].sel(days=seasonal_doy)

    # Save to file
    temp_surf_seasonal.to_netcdf(temp_surf_file)
    temp_surf_clim_seasonal.to_netcdf(temp_surf_clim_file)
else:
    temp_surf_seasonal = xr.open_dataset(temp_surf_file)
    temp_surf_clim_seasonal = xr.open_dataset(temp_surf_clim_file)

# %% ======================== MHW metrics ========================
# -- Temperature anomaly
temp_anom_file = os.path.join(main_path, 'temp_anomalies_seasonal.nc')
if not os.path.exists(temp_anom_file):
    temp_anomalies = temp_surf_seasonal.temp_surf - temp_surf_clim_seasonal.clim_sst
    temp_anomalies_reset = temp_anomalies.assign_coords(days=np.arange(181))
    temp_anomalies_reset.to_netcdf(temp_anom_file)
else:
    temp_anomalies_reset = xr.open_dataset(temp_anom_file)
# print(f"Min anomaly: {float(temp_anomalies.min().values):.2f} °C") # -5.02 °C
# print(f"Max anomaly: {float(temp_anomalies.max().values):.2f} °C") # 6.10 °C

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
    
    # We consider the MHW cumulative intensity, i.e. for a 3°C event, the full event is considered (even if not full event >3°C)
    # det_cases = {
    #     '90th': det_90th,
    #     '1deg': det_90th * (ds_mhw['det_1deg'] == 1).assign_coords(years=years_coord).astype(float),
    #     '2deg': det_90th * (ds_mhw['det_2deg'] == 1).assign_coords(years=years_coord).astype(float),
    #     '3deg': det_90th * (ds_mhw['det_3deg'] == 1).assign_coords(years=years_coord).astype(float),
    #     '4deg': det_90th * (ds_mhw['det_4deg'] == 1).assign_coords(years=years_coord).astype(float),
    # }
    # det_cases = {k: v.assign_coords(days=np.arange(181)) for k, v in det_cases.items()}

    # Compute cumulative intensity 
    def compute_CI(case_name):
        return case_name, (temp_anomalies_reset * det_cases[case_name]).sum(dim='days')
    results = process_map(compute_CI, list(det_cases.keys()), max_workers=5, chunksize=1, desc='Computing CI')
    CI = {name: da for name, da in results}

    # To Dataset
    CI_ds = xr.Dataset({name: ds for name, ds in CI.items()})
    
    # Check
    # print(temp_surf_seasonal.isel(years=1999-1980, eta_rho=201, xi_rho=1083).temp_surf.values)
    # print(ds_mhw.isel(years=1999-1980, eta_rho=201, xi_rho=1083).det_1deg.values)
    # print(ds_mhw.isel(years=1999-1980, eta_rho=201, xi_rho=1083).duration.values)
    # print(temp_surf_clim_seasonal.isel(eta_rho=201, xi_rho=1083).clim_sst.values)
    # print(CI_ds['1deg'].isel(eta_rho=201, xi_rho=1083).values)


    # Save
    CI_ds.to_netcdf(cum_intensity_file)

else: 
    CI_ds = xr.open_dataset(cum_intensity_file)

# Check
print(f"Min CI (1°C): {CI_ds['1deg'].where(CI_ds['1deg'] > 0).min().values}") #0.266326904296875 °C days
print(f"Max CI (1°C): {CI_ds['1deg'].max().values}") # 397.5961456298828 °C days

print(f"Min CI (3°C): {CI_ds['3deg'].where(CI_ds['3deg'] > 0).min().values}") # 0.36895751953125 °C days
print(f"Max CI (3°C): {CI_ds['3deg'].max().values}") # 386.8313751220703 °C days


# %% ======================== MPAs data ========================
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)
mpa_dict = {"Weddell Sea": (mpas_ds.mask_ws, "#5f0f40"),
            "East Antarctic": (mpas_ds.mask_ea, "#C00225"),
            "Ross Sea": (mpas_ds.mask_rs, "#c77c27"),
            "South Orkney Islands southern shelf":  (mpas_ds.mask_o,  "#e05c8a"),
            "Antarctic Peninsula": (mpas_ds.mask_ap, "#867308")}

mpa_masks = {"WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}



# %% ======================== ROMS area ========================
ROMS_area = xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')
area_SO_surf = ROMS_area['area'].isel(z_t=0) # Select surface layer
area_SO = area_SO_surf.where(ROMS_area['lat_rho'] <= -60, drop=True)
area_SO = area_SO.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))
area_SO_np = area_SO.values #in km2 -- shape (231, 1440)
print(f'Min area in Southern Ocean: {np.nanmin(area_SO_np)}')
print(f'Max area in Southern Ocean: {np.nanmax(area_SO_np)}')

# Sum grid cell areas
total_area_SO_km2 = area_SO.sum().item()
print(f'Total Southern Ocean area south of 60°S: {total_area_SO_km2:.2f} km²')

# %% ======================== Area Affected by MHWs in MPAs [%] ========================
mpa_area_affected_perc = {}
for abbrv, (name, mask) in mpa_masks.items():
    # test
    # abbrv='RS'
    # name = mpa_masks[abbrv][0]
    # mask = mpa_masks[abbrv][1]
    print(name)

    # Initialisation
    mask_bool = mask.values.astype(bool)   # (231, 1442)
    mpa_area_affected_perc[abbrv] = {'name': name, 'ts': {}}
        
    # Total area in km2
    total_area_mpa = np.nansum(area_SO_np[mask_bool])

    for key in CI_ds:
        # key='1deg'
        yearly_ts = []
        for y in range(39):
            # y=36
            
            # Cumulative Intensity for this year
            ci = CI_ds[key].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values[y] #(231, 1440)
            
            # Area affected by MHWs
            affected_cells = mask_bool & (ci > 0)
            affected_area = np.nansum(area_SO_np[affected_cells])
            affected_area_perc = (affected_area / total_area_mpa)*100
            # print(affected_area_perc)
            if y==36: 
                print(f'Year: {1980+y}, CI key: {key}, Affected area [%]: {affected_area_perc:.2f}%')

            yearly_ts.append(affected_area_perc)

        mpa_area_affected_perc[abbrv]['ts'][key] = np.array(yearly_ts)


# %% ======================== Climatological CI ========================
# Climatological CI: Mean over 1980-2009 period
ci_clim_dict = {}
for abbrv, (name, mask) in mpa_masks.items():
    mask_bool = mask.values.astype(bool)   # (231, 1442)
    for key in CI_ds:
        ci_30yrs = (CI_ds[key].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).isel(years=slice(0, 30)).mean(dim='years', skipna=True))
        ci_clim_dict[key] = ci_30yrs.values

# %% ======================== Time series for CI and seasonal change in biomass ========================
years_coord = np.arange(1980, 2019)

# -- MPAs
# p_change_mpas = {}
# p_change_mpas_absolute={}
mpa_ci = {}
p_change_mpas_mhw = {}

for abbrv, (name, mask) in mpa_masks.items():
    # test
    # abbrv='RS'
    # name = mpa_masks[abbrv][0]
    # mask = mpa_masks[abbrv][1]

    print(name)

    mask_bool = mask.values.astype(bool) # (231, 1442)
    total_mpa_area = np.nansum(area_SO_np[mask_bool])

    mpa_ci[abbrv] = {'name': name, 'ts': {}}
    p_change_mpas_mhw[abbrv] = {'name': name, 'ts': {}}

    for key in CI_ds:
        # key='1deg'
        ci_clim = ci_clim_dict[key] 
        
        ci_key = CI_ds[key].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values
        biomass = change_interp_cell['actual'].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).biomass.values

        yearly_ci_ts = []
        yearly_bio_ts = []

        for y in range(39):
            # y=36
            # Retrieve cumulative intensity and biomass change for this year
            ci_yr = ci_key[y]  
            biomass_yr = biomass[y] #shape (10, 231, 1440)

            # Anomaly relative to climatology for each cell -- biomass gain is already relative to clim
            ci_yr_anom = ci_yr - ci_clim # shape (231, 1440)

            # Select only cells inside MPA that have been affected by MHWs 
            valid_affected_cells = (mask_bool & (ci_yr>0) & np.isfinite(ci_yr) & np.isfinite(area_SO_np)) #shape (231, 1440)

            # Spatial average (area-weighted!) and
            if np.any(valid_affected_cells) and (total_mpa_area>0):
                # Exposure-weighted response, i.e. mean biomass change for cells affected by MHWs x fraction of cells affected by MHWs
                weights_mhw = area_SO_np[valid_affected_cells] #shape (ncell, )
                ci_avg_weighted = np.nansum(ci_yr_anom[valid_affected_cells] * weights_mhw, axis=0) / total_mpa_area
                biomass_avg_weighted = np.nansum(biomass_yr[:,valid_affected_cells] * weights_mhw, axis=1) / total_mpa_area #shape (nboot,)
                biomass_avg_weighted_pct = biomass_avg_weighted *100

                yearly_ci_ts.append(ci_avg_weighted)
                yearly_bio_ts.append(biomass_avg_weighted_pct)
            else:
                yearly_ci_ts.append(np.nan)
                yearly_bio_ts.append(np.full(biomass.shape[1], np.nan))

        # print(f"Biomass mean 30yrs (1980-2009): {np.nansum(yearly_bio_ts[:30]):.2f} %")

        mpa_ci[abbrv]['ts'][key] = np.array(yearly_ci_ts)
        p_change_mpas_mhw[abbrv]['ts'][key] = np.array(yearly_bio_ts)


# %% ======================== Take median and std over bootstraps ========================
p_change_mpas_mhw_stats = {}

for abbrv in p_change_mpas_mhw:
    p_change_mpas_mhw_stats[abbrv] = {'name': p_change_mpas_mhw[abbrv]['name'],
                                      'ts': {}}

    for key in CI_ds:

        arr = p_change_mpas_mhw[abbrv]['ts'][key]   # (39, nboot)

        p_change_mpas_mhw_stats[abbrv]['ts'][key] = {
            'median': np.nanmedian(arr, axis=1),
            'std': np.nanstd(arr, axis=1)
        }

# Instead of taking the mean over MPAs, take only mean over cells inside MPA that have been affected by MHWs during y year.
# p_change_mpas_mhw = {}

# for abbrv, (name, mask) in mpa_masks.items():
    # test
    # abbrv='RS'
    # name = mpa_masks[abbrv][0]
    # mask = mpa_masks[abbrv][1]
    # print(name)

    # mask_bool = mask.values.astype(bool)

    # p_change_mpas_mhw[abbrv] = {'name': name, 'ts': {}}

    # for key in CI_ds:
    #     # key='1deg'
    #     yearly_ts = []
    #     for y in range(39):
    #         # y=36
            
    #         # Restrieve cell affected by MHWs in this year (CI > 0) and inside the MPA
    #         ci = CI_ds[key].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).values[y] #shape (231, 1440)
    #         affected_cell = mask_bool & (ci > 0)
            
    #         # Biomass change for affected cell -- spatial average (area weighting could be added in the future)
    #         biomass_y = (p_actual_change_median_cell.isel(xi_rho=slice(0, mpas_south60S.xi_rho.size)).biomass.values[y])
    #         vals = biomass_y[affected_cell]
    #         weights = area_SO_np[affected_cell]
    #         valid_cells = np.isfinite(vals) & np.isfinite(weights)
    #         if np.any(valid_cells) and np.sum(weights[valid_cells]) > 0:
    #             yearly_ts.append(np.sum(vals[valid_cells] * weights[valid_cells]) / np.sum(weights[valid_cells]))
    #         else:
    #             yearly_ts.append(np.nan)

    #     p_change_mpas_mhw[abbrv]['ts'][key] = np.array(yearly_ts)

# %% ======================== Plot TS for all MPAs ========================
ci_key = '1deg'
growth_key = 'growth_' + ci_key

# Exclude SO for 3°C threshold because no events occur there
if ci_key == '3deg':
    mpa_order = ['WS', 'EA', 'RS', 'AP']
else:
    mpa_order = ['WS', 'EA', 'RS', 'SO', 'AP']

n_mpas = len(mpa_order)

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

fig = plt.figure(figsize=(6, 3 * n_mpas))

# Outer grid: one row per MPA, with space between them
outer_gs = GridSpec(n_mpas, 1, figure=fig, hspace=0.45)

axes = []
for idx in range(n_mpas):
    # Inner grid: CI + biomass tightly packed
    inner_gs = GridSpecFromSubplotSpec(2, 1,
                                       subplot_spec=outer_gs[idx],
                                       height_ratios=[1, 1],
                                       hspace=0.07)
    axes.append((fig.add_subplot(inner_gs[0]),   # ax_ci
                 fig.add_subplot(inner_gs[1])))   # ax_bio

for idx, abbrv in enumerate(mpa_order):
    name, mask  = mpa_masks[abbrv]
    ax_ci, ax_bio = axes[idx]
    mpa_color = mpa_dict[name][1]

    # -- CI subplot
    ci_ts = mpa_ci[abbrv]['ts'][ci_key]
    ax_ci.bar(years_coord, ci_ts, color=mpa_color, alpha=0.75, width=0.9)
    ax_ci.set_ylabel(r'$\Delta$CI [°C · days]', fontsize=9)
    ax_ci.tick_params(labelsize=8, bottom=False)
    ax_ci.tick_params(axis='x', labelbottom=False)
    ax_ci.set_ylim(min(0, np.nanmin(ci_ts) * 1.1), np.nanmax(ci_ts) * 1.1)
    ax_ci.set_xlim(1980-0.5, 2018+0.5)
    ax_ci.set_xlabel('')
    # Title above the plot
    ax_ci.text(0.01, 1.18, f'{name} ({abbrv})',
               transform=ax_ci.transAxes, fontsize=10,
               fontweight='bold', va='top')
    # Add the area affected on the right y-axis
    area_ts = mpa_area_affected_perc[abbrv]['ts'][ci_key]
    ax_area = ax_ci.twinx()
    ax_area.plot(years_coord, area_ts, color='black', linestyle='--', linewidth=0.8)
    ax_area.set_ylabel('Area affected [\%]', fontsize=9)
    ax_area.tick_params(labelsize=8, bottom=False)
    ax_area.tick_params(axis='x', labelbottom=False)
    ax_area.set_ylim(0, np.nanmax(area_ts))
    ax_area.set_xlabel('')

    # -- Biomass subplot
    bio_ts  = p_change_mpas_mhw_stats[abbrv]['ts'][ci_key]['median']
    bio_std = p_change_mpas_mhw_stats[abbrv]['ts'][ci_key]['std']
    bar_colors = ['#94D2BD' if v >= 0 else '#AEA8DE' for v in bio_ts]
    ax_bio.bar(years_coord, bio_ts, width=0.9, color=bar_colors, alpha=0.85,
               yerr=bio_std, capsize=2,  error_kw={
        'elinewidth': 0.8,
        'capsize': 1.5,
        'capthick': 0.8,
        'alpha': 0.8
    })
    # ax_bio.set_ylabel(r'$\Delta$Biomass change[\%]', fontsize=9)
    ax_bio.set_ylabel(r'$\Delta$ Seasonal' +'\n'+ 'biomass gain [\%]', fontsize=9)
    ax_bio.axhline(0, color='black', lw=0.5, ls=':', alpha=0.6)
    ax_bio.tick_params(labelsize=8)
    bio_max = np.nanmax(np.abs(bio_ts)) * 1.2
    ax_bio.set_ylim(-bio_max, bio_max)

    # Add the growth anomalies
    # growth_ts = growth_change_mpas[abbrv]['ts'][growth_key]
    # ax_growth = ax_bio.twinx()
    # ax_growth.plot(years_coord, growth_ts, color='black', linestyle='--', linewidth=0.8)
    # ax_growth.set_ylabel(r'$\Delta$ growth [mm]', fontsize=9)
    # ax_growth.tick_params(labelsize=8, bottom=False)
    # ax_growth.set_ylim(0, np.nanmax(growth_ts))

    # -- Find year with highest mean growth biomass 
    # top_year = np.array(years_coord)[np.argsort(bio_ts)[-1:]].item()
    top_year = np.array(years_coord)[np.argmax(mpa_area_affected_perc[abbrv]['ts'][ci_key])]
    ax_ci.axvspan(top_year - 0.45, top_year + 0.45, color='#005F73', alpha=0.15, zorder=0)
    ax_bio.axvspan(top_year - 0.45, top_year + 0.45, color='#005F73', alpha=0.15, zorder=0)

    base_ticks = [1980, 1990, 2000, 2010, 2018]
    all_ticks  = sorted(set(base_ticks) | {top_year})
    ax_bio.set_xticks(all_ticks)
    ax_bio.set_xlim(1979.5, 2018.5)

    ax_bio.tick_params(axis='x', labelsize=8)

    for tick, label in zip(ax_bio.get_xticks(), ax_bio.get_xticklabels()):
        if int(tick) == top_year:
            label.set_color('#005F73')
            label.set_fontweight('bold')
            label.set_rotation(35)
        else:
            label.set_color('black')
            label.set_rotation(0)

# -- Single legend below all subplots
bio_handles = [
    Patch(facecolor='#94D2BD', alpha=0.85, label='Biomass gain'),
    Patch(facecolor='#AEA8DE', alpha=0.85, label='Biomass loss'),]

fig.legend(
    handles=bio_handles,
    fontsize=8, framealpha=0.85,
    loc='lower center', ncol=4,
    bbox_to_anchor=(0.5, 0.01)
)

fig.suptitle(rf'MHW $\geq$ 90th perc and {ci_key[0]}°C', fontsize=11)
fig.subplots_adjust(hspace=0.5, left=0.1, right=0.95, top=0.97, bottom=0.06)
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/fig3_TS_CI_biomass_{ci_key[0]}deg.pdf'), dpi=200, format='pdf', bbox_inches='tight')

# %% ======================== Mask Biomass change for each MPAs ========================
mpa_biomass_cells = {}
mpa_CI_cell = {}

for abbrv, (name, mask) in mpa_masks.items():
    # test
    # abbrv='AP'
    # name = mpa_masks[abbrv][0]
    # mask = mpa_masks[abbrv][1]

    # Cropping dataset to match dim
    mask_bool = mask.astype(bool)
    biomass_change_cell_reformat = change_interp_cell['actual'].isel(xi_rho=slice(0, mpas_south60S.xi_rho.size))*100
    CI_cell_reformat = CI_ds.isel(xi_rho=slice(0, mask_bool.xi_rho.size))

    # broadcast mask over years automatically
    biomass_masked = biomass_change_cell_reformat.biomass.where(mask_bool.data)
    median_biomass_masked = biomass_masked.median(dim='bootstraps', skipna=True)
    CI_masked = CI_cell_reformat['1deg'].where(mask_bool.data)

    mpa_biomass_cells[abbrv] = median_biomass_masked
    mpa_CI_cell[abbrv] = CI_masked

    print(f'{abbrv} done.')

# -- Check
abbrv = 'AP'
ci_ap = mpa_CI_cell[abbrv]

# Min (above 0)
ci_ap_pos = ci_ap.where(ci_ap > 0)
min_val = float(ci_ap_pos.min().values)
max_val = float(ci_ap.max().values)
print(f"Min CI AP (>0): {min_val:.6f}") #0.274467 °C days
print(f"Max CI AP:      {max_val:.4f}") #317.2082 °C days

# # Where/when min
# min_idx  = ci_ap_pos.argmin(dim=['years', 'eta_rho', 'xi_rho'])
# year_idx = int(min_idx['years'].values)
# eta_idx  = int(min_idx['eta_rho'].values)
# xi_idx   = int(min_idx['xi_rho'].values)
# print(f"\n-- Minimum --")
# print(f"Year:    {int(ci_ap.years[year_idx].values)}")
# print(f"eta_rho: {eta_idx}, xi_rho: {xi_idx}")
# print(f"Lon: {mpas_ds.lon_rho.values[eta_idx, xi_idx]:.2f}")
# print(f"Lat: {mpas_ds.lat_rho.values[eta_idx, xi_idx]:.2f}")

# # Where/when max
# max_idx  = ci_ap.argmax(dim=['years', 'eta_rho', 'xi_rho'])
# year_idx = int(max_idx['years'].values)
# eta_idx  = int(max_idx['eta_rho'].values)
# xi_idx   = int(max_idx['xi_rho'].values)
# print(f"\n-- Maximum --")
# print(f"Year:    {int(ci_ap.years[year_idx].values)}")
# print(f"eta_rho: {eta_idx}, xi_rho: {xi_idx}")
# print(f"Lon: {mpas_ds.lon_rho.values[eta_idx, xi_idx]:.2f}")
# print(f"Lat: {mpas_ds.lat_rho.values[eta_idx, xi_idx]:.2f}")

# %% ======================== Maps of Seasonal Biomass gain change for MPA only ========================
from skimage import measure
plot='report'
ci_key = '1deg'
if ci_key == '3deg':
    mpa_order = ['WS', 'EA', 'RS', 'AP']
else:
    mpa_order = ['WS', 'EA', 'RS', 'SO', 'AP']
n_mpas = len(mpa_order)


# Setting
lw = 1.0 if plot == 'slides' else 0.5
lw_grid = 0.7 if plot == 'slides' else 0.3
gridlabel_kwargs = {'size': 10, 'rotation': 0} if plot == 'slides' else {'size': 6, 'rotation': 0}
subtitle_kwargs  = {'fontsize': 12, 'fontweight': 'bold'}

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# Colormap
cmap_bio = LinearSegmentedColormap.from_list('purple_white_teal',
               ["#AEA8DE", "white", "#94D2BD"])
norm_bio = mcolors.TwoSlopeNorm(vmin=-50, vcenter=0, vmax=50)

# --- Figure ---
fig, axes = plt.subplots(n_mpas, 1, figsize=(4, 10),
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})


for idx, abbrv in enumerate(mpa_order):
    # print(idx, abbrv)

    ax = axes[idx]

    # -- Retrieve years of maximum biomass change for each MPA
    # top_year = np.array(years_coord)[np.argmax(p_change_mpas[abbrv]['ts'])]
    top_year = np.array(years_coord)[np.argmax(mpa_area_affected_perc[abbrv]['ts'][ci_key])]

    # max_idx  = np.argmax(p_change_mpas[abbrv]['ts'])
    # max_year = int(years_coord[max_idx])

    data = mpa_biomass_cells[abbrv].isel(years=top_year-1980)

    # Circular boundary
    ax.set_boundary(circle, transform=ax.transAxes)

    # Features
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Plot
    pcm = ax.pcolormesh(data.lon_rho, data.lat_rho, data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap_bio, norm=norm_bio,
                        rasterized=True, zorder=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=20)
    gl.xlabels_top  = False
    gl.ylabels_right = False
    gl.xlabel_style = gridlabel_kwargs
    gl.ylabel_style = gridlabel_kwargs
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # MPA boundaries
    lon_np   = mpas_ds.lon_rho.values
    lat_np   = mpas_ds.lat_rho.values
    mask_2d  = mpa_masks[abbrv][1].values  # boolean mask for this MPA

    contours = measure.find_contours(mask_2d.astype(float), 0.5)
    for contour in contours:
        eta_idx = contour[:, 0].astype(int).clip(0, lon_np.shape[0] - 1)
        xi_idx  = contour[:, 1].astype(int).clip(0, lon_np.shape[1] - 1)
        ax.plot(lon_np[eta_idx, xi_idx], lat_np[eta_idx, xi_idx],
                color=mpa_dict[mpa_masks[abbrv][0]][1],
                linewidth=0.5, transform=ccrs.PlateCarree(), zorder=2)
        
    ax.text(-0.2, 1, rf'{top_year} ({abbrv})',
                transform=ax.transAxes, fontsize=6, verticalalignment='top',
                bbox=dict(boxstyle='round, pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))

# Shared colorbar
cbar = fig.colorbar(pcm, ax=axes, orientation='horizontal', extend='both',
                    fraction=0.03, pad=0.05, shrink=0.6)

cbar.set_label('Biomass change [\%]', fontsize=9)
cbar.set_ticks(np.arange(-50, 51, 10))
cbar.ax.tick_params(labelsize=7)
plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/fig3_maps_biomass_{ci_key[0]}deg.pdf'), dpi=200, format='pdf', bbox_inches='tight')




# %% ======================== Maps of CI for MPA only ========================
from skimage import measure
plot='report'
ci_key = '1deg'
if ci_key == '3deg':
    mpa_order = ['WS', 'EA', 'RS', 'AP']
else:
    mpa_order = ['WS', 'EA', 'RS', 'SO', 'AP']
n_mpas = len(mpa_order)


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
fig, axes = plt.subplots(n_mpas, 1, figsize=(4, 10),
                         subplot_kw={'projection': ccrs.SouthPolarStereo()})

for idx, abbrv in enumerate(mpa_order):
    ax = axes[idx]

    # top_idx  = np.argmax(p_change[abbrv]['ts'][ci_key])
    # top_year = int(years_coord[top_idx])
    # abbrv='WS'
    # top_year = np.array(years_coord)[np.argmax(p_change_mpas[abbrv]['ts'])]
    top_year = np.array(years_coord)[np.argmax(mpa_area_affected_perc[abbrv]['ts'][ci_key])]
    data = mpa_CI_cell[abbrv].sel(years=top_year)
    
    # -- Colormaps (1 per MPA)
    mpa_name  = mpa_masks[abbrv][0]
    mpa_color = mpa_dict[mpa_name][1]

    rgb = mcolors.to_rgb(mpa_color)
    pastel = tuple(0.85 + 0.15 * c for c in rgb)   # very close to white
    dark   = tuple(0.35 * c for c in rgb)            # much darker

    import matplotlib.colors as mcolors
    from matplotlib.colors import LogNorm

    cmap_mpa = LinearSegmentedColormap.from_list(f'cmap_{abbrv}', [pastel, mpa_color, dark])
    cmap_mpa.set_under('white') # make 0 white   
    data_vals = data.values
    positive_vals = data_vals[data_vals > 0]

    from matplotlib.colors import Normalize

    positive_vals = data_vals[data_vals > 0]

    if positive_vals.size == 0:
        norm_mpa = Normalize(vmin=1, vmax=2)  # dummy safe range
    else:
        vmin = np.nanpercentile(positive_vals, 5)
        vmax = np.nanpercentile(positive_vals, 90)

        # safety fallback if flat field
        if vmax <= vmin:
            vmax = vmin + 1e-6

    norm_mpa = Normalize(vmin=vmin, vmax=vmax)
   
    # -- Plot CI
    plot_data = data.where(data > 0)
    cmap_mpa.set_bad((0,0,0,0)) # make NaNs transparent
    pcm = ax.pcolormesh(data.lon_rho, data.lat_rho, plot_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap_mpa, norm=norm_mpa,
                        rasterized=True, zorder=2)
    
    # CI=0
    zero_mask = xr.where(data == 0, 1, np.nan)
    ax.pcolormesh(data.lon_rho, data.lat_rho, zero_mask,
                  transform=ccrs.PlateCarree(),
                  cmap=mcolors.ListedColormap(['white']), # make 0 white
                  rasterized=True, zorder=1)

    # -- Setup features
    # Circular boundary
    ax.set_boundary(circle, transform=ax.transAxes)

    # Features
    ax.coastlines(color='black', linewidth=lw, zorder=5)
    ax.add_feature(cfeature.LAND, zorder=4, facecolor='#F6F6F3')
    ax.set_facecolor('lightgrey')

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--', linewidth=lw_grid, zorder=20)
    gl.xlabels_top   = False
    gl.ylabels_right = False
    gl.xlabel_style  = gridlabel_kwargs
    gl.ylabel_style  = gridlabel_kwargs
    gl.xformatter    = LongitudeFormatter()
    gl.yformatter    = LatitudeFormatter()

    # -- MPA boundary
    lon_np  = mpas_ds.lon_rho.values
    lat_np  = mpas_ds.lat_rho.values
    mask_2d = mpa_masks[abbrv][1].values

    contours = measure.find_contours(mask_2d.astype(float), 0.5)
    for contour in contours:
        eta_idx = contour[:, 0].astype(int).clip(0, lon_np.shape[0] - 1)
        xi_idx  = contour[:, 1].astype(int).clip(0, lon_np.shape[1] - 1)
        ax.plot(lon_np[eta_idx, xi_idx], lat_np[eta_idx, xi_idx],
                color='black', linewidth=0.5,
                transform=ccrs.PlateCarree(), zorder=2)

    ax.text(-0.2, 1, rf'{top_year} ({abbrv})',
            transform=ax.transAxes, fontsize=6, verticalalignment='top',
            bbox=dict(boxstyle='round, pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))

    # -- Colorbars 
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical',
                        extend='max', fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('CI [°C days]', fontsize=6)
    
    

    # if positive_vals.size > 0:
    #     # ticks = [1, 5, 10, 30, 100, 300]
    #     # ticks = [t for t in ticks if t <= np.nanmax(positive_vals)]

    #     # cbar.set_ticks(ticks)
    #     # cbar.set_ticklabels([str(t) for t in ticks])

    #     from matplotlib.ticker import NullLocator
    #     cbar.ax.yaxis.set_minor_locator(NullLocator())
    # else:
    #     cbar.set_ticks([])

    cbar.ax.tick_params(labelsize=6)

# -- Legend
from matplotlib.patches import Patch
no_mhw_patch = Patch(facecolor='white', edgecolor='gray', linewidth=0.5,
                     label='No MHWs (CI = 0)')
fig.legend(handles=[no_mhw_patch], loc='lower center',
           bbox_to_anchor=(0.5, 0.08), fontsize=7, framealpha=0.85)

plt.show()
# plt.savefig(os.path.join(os.getcwd(), f'D_Paper_Scripts/figures/results/fig3_maps_CI_{ci_key[0]}deg.pdf'), dpi=200, format='pdf', bbox_inches='tight')


# %% ======================== Plot Time series ========================
ci_key = '1deg'

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
