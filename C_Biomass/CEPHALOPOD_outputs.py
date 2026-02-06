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
path_surrogates = os.path.join(path_biomass, f'surrogates')
path_masslength = os.path.join(path_surrogates, f'mass_length')
path_cephalopod = os.path.join(path_biomass, 'CEPHALOPOD')

# %% ====================== Prepare biomass data ======================
# --- Load biomass from CEPHALOPOD
# 5 different algorithms, 10 bootstraps per algo
biomass_data = xr.open_dataset('/net/meso/work/aschickele/CEPHALOPOD/output/Marguerite_tot_krill_WOA_2026-01-12 17:27:04.013515/Euphausiacea/tot_krill_biomass.nc') #shape (coords, model, time) = (d1=64800, d2=5*10,  d3=12)
# print(f'max: {biomass_data.max()} mgC/m3') #7426.0 mgC/m3
# print(f'min: {biomass_data.min()} mgC/m3') #-6.576 mgC/m3

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
biomass_cephalopod = xr.Dataset(data_vars=dict(total_krill_biomass=(["months", "algo_bootstrap", "lat", "lon"], arr_global)), 
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

biomass_cephalopod_60S = xr.Dataset(data_vars=dict(total_krill_biomass=(["months", "algo_bootstrap", "lat", "lon"], arr_60S_flipped)),
                                    coords=dict( months=np.arange(1, 13), algo_bootstrap=np.arange(1, 51), lat=lat_60S_flipped, lon=np.linspace(-179.5, 179.5, 360)))

biomass_cephalopod_60S.attrs = biomass_cephalopod.attrs.copy()
biomass_cephalopod_60S.attrs.update({"model_extent": "Southern Ocean",})

# -- Euphausia superba biomass
# Euphausia = 80% of total krill
biomass_cephalopod_60S_euphausia = xr.Dataset(data_vars=dict(euphausia_biomass=(["months", "algo_bootstrap", "lat", "lon"], arr_60S_flipped * 0.8)),
                                              coords=dict(months=np.arange(1, 13), algo_bootstrap=np.arange(1, 51), lat=lat_60S_flipped, lon=np.linspace(-179.5, 179.5, 360)))

biomass_cephalopod_60S_euphausia.attrs = {"description": "Biomass of Euphausia superba assuming 80% of total krill biomass",
                                          "units": "mg C m-3",}

# -- Save 
output_file_biomass_tot_krill = os.path.join(path_cephalopod, "total_krill_biomass_SO.nc")
if not os.path.exists(output_file_biomass_tot_krill):
    biomass_cephalopod_60S.to_netcdf(output_file_biomass_tot_krill, engine="netcdf4")

output_file_biomass_euphausia = os.path.join(path_cephalopod, "euphausia_biomass_SO.nc")
if not os.path.exists(output_file_biomass_euphausia):
    biomass_cephalopod_60S_euphausia.to_netcdf(output_file_biomass_euphausia, engine="netcdf4")


# %% ====================== Interpolation functions ======================
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt

def fill_lat_trend(biomass_da, roms_mask):
    filled = biomass_da.copy()
    
    # Extract latitude array 
    lats = biomass_da.lat_rho[:, 0].values  # shape (231,)
    
    for i in range(biomass_da.shape[1]):  # loop over xi_rho 
        valid = np.isfinite(biomass_da[:, i]) & roms_mask[:, i]
        if valid.sum() < 2:
            continue  # skip if not enough points to fit
        
        # Linear fit along latitude
        coeff = np.polyfit(lats[valid], biomass_da[:, i][valid], deg=1)
        trend = np.polyval(coeff, lats)
        
        # Fill NaNs along latitude
        filled[:, i] = np.where(valid, biomass_da[:, i], trend)
        
    return filled


def fill_lon_trend(biomass_da, roms_mask):
    # Test
    # biomass_da=biomass_small.euphausia_biomass
    # roms_mask=roms_ocean_mask.values

    filled = biomass_da.copy()
    
    # Extract Longitude array 
    lons = biomass_da.lon_rho[0, :].values  # shape (1442,)
    
    for i in range(biomass_da.shape[0]): #loop over eta
        valid = np.isfinite(biomass_da[i, :]) & roms_mask[i, :]
        if np.sum(valid) < 2:
            continue 

        # Linear fit along longitude
        coeff = np.polyfit(lons[valid], biomass_da[i, :][valid], deg=1)
        trend = np.polyval(coeff, lons)

        filled[i, :] = np.where(valid, biomass_da[i, :], trend)
        
        
    return filled


def process_algorithm(a):
    # Test
    # a=0
    # print(f"Processing algorithm {a}")
    biomass_algo = biomass_regridded.euphausia_biomass.isel(algo_bootstrap=a)
    biomass_algo_monthly = biomass_algo.groupby("months").median("days")
    
    filled_monthly = []
    
    for m in biomass_algo_monthly.months.values:
        # Test
        # m=1

        biomass_mth = biomass_algo_monthly.sel(months=m)

        # Step1. Filling NA along latitudes
        biomass_lat = fill_lat_trend(biomass_mth, roms_ocean_mask.values) #shape (231, 1442)

        # Step2. Filling NA along longitudes
        biomass_lon = fill_lon_trend(biomass_mth, roms_ocean_mask.values) #shape (231, 1442)

        # Weighted combination using inverse distance --> Interpolation error variance increases with distance from data support
        both_valid = np.isfinite(biomass_lat) & np.isfinite(biomass_lon)
        mask_lat = np.isfinite(biomass_lat.values)
        mask_lon = np.isfinite(biomass_lon.values)

        d_lat = distance_transform_edt(~mask_lat)
        d_lon = distance_transform_edt(~mask_lon)

        w_lat = 1 / (d_lat + 1) #shape (231, 1442)
        w_lon = 1 / (d_lon + 1) #shape (231, 1442)

        biomass_latlon = xr.where(both_valid, (w_lat * biomass_lat + w_lon * biomass_lon) / (w_lat + w_lon),
                                    xr.where(np.isfinite(biomass_lat), biomass_lat, biomass_lon))

        # biomass_latlon = xr.where(both_valid, 0.3 * biomass_lat + 0.7 * biomass_lon, 
        #                     xr.where(np.isfinite(biomass_lat), biomass_lat, biomass_lon))

        # biomass_latlon = 0.5 * biomass_lat + 0.5 * biomass_lon
    
        # Step3. Gaussian smoothing
        biomass_smooth = gaussian_filter(biomass_latlon, sigma=1)
        
        da = xr.DataArray(biomass_smooth, dims=biomass_mth.dims, coords=biomass_mth.coords, attrs=biomass_mth.attrs)
        filled_monthly.append(da)
    
    filled_monthly = xr.concat(filled_monthly, dim="months")
    filled_monthly = filled_monthly.assign_coords(months=biomass_algo_monthly.months)
    filled_monthly = filled_monthly.where(roms_ocean_mask)
    
    return filled_monthly.expand_dims(algo_bootstrap=[a])


# %% ====================== Regridding Biomass and Abundance to ROMS grid ======================
import xesmf as xe
from tqdm.contrib.concurrent import process_map

output_file_biomass_regrid= os.path.join(path_cephalopod, "euphausia_biomass_SO_regridded.nc")
output_file_biomass_regrid_interp= os.path.join(path_cephalopod, "euphausia_biomass_SO_regrid_interp.nc")

if not (os.path.exists(output_file_biomass_regrid) and os.path.exists(output_file_biomass_regrid_interp)):

    # ===================== Prepare data =====================
    # Load dataset with correct grid
    area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc')['area'].isel(z_t=0)
    area_SO = area_roms.where(area_roms['lat_rho'] <= -60, drop=True) #shape (231, 1442)

    # From monthly to daily dataset
    # Repeat monthly value for each day of the month
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_index = np.concatenate([np.repeat(month, days_in_month[month-1]) for month in range(1, 13)])
    assert day_index.shape[0] == 365
    day_index_xr = xr.DataArray(day_index, dims="days", name="month")
    biomass_daily = biomass_cephalopod_60S_euphausia.sel(months=day_index_xr) #shape: (365, 50, 30, 360)

    # Check if the same 
    diff = biomass_daily.isel(days=304) - biomass_daily.isel(days=304+14)
    print(float(diff.euphausia_biomass.max()), float(diff.euphausia_biomass.min()))

    # Select only austral summer and early spring
    jan_april_biomass = biomass_daily.sel(days=slice(0, 120))
    jan_april_biomass.coords['days'] = jan_april_biomass.coords['days'] 
    nov_dec_biomass = biomass_daily.sel(days=slice(304, 366))
    nov_dec_biomass.coords['days'] = np.arange(304, 365)
    biomass_daily_austral = xr.concat([nov_dec_biomass, jan_april_biomass], dim="days") #shape: (181, 50, 30, 360)

    # Fix longitudes
    # ROMS (24.125, 383.875) - put to (0, 360)
    roms_fixed = area_SO.assign_coords(lon_rho=(area_SO.lon_rho % 360)) #min lon_rho = 0.125 

    # CEPHALOPOD longitude (-180, 180) - put to (0, 360)
    biomass_fixed = biomass_daily_austral.assign_coords(lon=((biomass_daily_austral.lon % 360))).sortby("lon") #min long = 0.5

    # ===================== Regridding =====================
    # Target grids
    in_ds = xr.Dataset(
        {"lon": (("lon",), biomass_fixed.lon.values),
        "lat": (("lat",), biomass_fixed.lat.values),})

    out_ds = xr.Dataset(
        {"lon": (("eta_rho", "xi_rho"), roms_fixed.lon_rho.values),
        "lat": (("eta_rho", "xi_rho"), roms_fixed.lat_rho.values),})

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

    # Add coordinates (lat, lon) from ROMS
    biomass_regridded = biomass_regridded.assign_coords(
        lon_rho=(("eta_rho", "xi_rho"), area_SO.lon_rho.values),
        lat_rho=(("eta_rho", "xi_rho"), area_SO.lat_rho.values))

    # Add Attributes
    if 'regrid_method' in biomass_regridded.attrs:
        del biomass_regridded.attrs['regrid_method']
    biomass_regridded.attrs.update({"description": "Biomass of Euphausia superba assuming 80% of total krill biomass",
                                    "regridding": "Nearest neighbors, using bilinear method.",
                                    "units": "mg C m-3",})

    # Save to file 
    biomass_regridded.to_netcdf(output_file_biomass_regrid, engine="netcdf4")

    # ===================== Interpolate NAs values =====================
    # Mask the land from ROMS
    roms_ocean_mask = roms_fixed > 0   #True False - shape (231, 1442)
   
    # -- Test interpolation
    biomass_regridded = xr.open_dataset(os.path.join(path_cephalopod, "euphausia_biomass_SO_regridded.nc"))
    biomass_regridded_small_extent = biomass_regridded.isel(days=0, algo_bootstrap=0, eta_rho=slice(150, 220), xi_rho=slice(1000,1200)).euphausia_biomass
    roms_mask = roms_ocean_mask.isel(eta_rho=slice(150, 220), xi_rho=slice(1000,1200))
        
    filled_lat_na = fill_lat_trend(biomass_regridded_small_extent, roms_mask)
    filled_lon_na = fill_lon_trend(biomass_regridded_small_extent, roms_mask)

    # Check if both valid
    both_valid = np.isfinite(filled_lat_na) & np.isfinite(filled_lon_na)
    mask_lat = np.isfinite(filled_lat_na.values)
    mask_lon = np.isfinite(filled_lon_na.values)
    d_lat = distance_transform_edt(~mask_lat)
    d_lon = distance_transform_edt(~mask_lon)
    w_lat = 1 / (d_lat + 1)
    w_lon = 1 / (d_lon + 1)
    filled_latlon_na = xr.where(both_valid, (w_lat * filled_lat_na + w_lon * filled_lon_na) / (w_lat + w_lon),
                                xr.where(np.isfinite(filled_lat_na), filled_lat_na, filled_lon_na))
    biomass_smooth = xr.DataArray(gaussian_filter(filled_latlon_na.values, sigma=1), coords=filled_latlon_na.coords,
                                  dims=filled_latlon_na.dims, name="biomass_smooth", attrs=filled_latlon_na.attrs)
    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)

    # 1. Original (regridded subset)
    pcm0 = axes[0].pcolormesh(biomass_regridded_small_extent.lon_rho, biomass_regridded_small_extent.lat_rho, 
                            biomass_regridded_small_extent, shading="nearest", cmap="viridis")
    axes[0].set_title("Regridded")
    axes[0].set_ylabel("Lat")

    # 2. Latitude NaNs filled
    pcm1 = axes[1].pcolormesh(filled_lat_na.lon_rho, filled_lat_na.lat_rho, filled_lat_na,
                            shading="nearest", cmap="viridis")
    axes[1].set_title("Latitude filled")

    # 3. Longitude NaNs filled
    pcm2 = axes[2].pcolormesh(filled_lon_na.lon_rho, filled_lon_na.lat_rho, filled_lon_na,
                            shading="nearest", cmap="viridis")
    axes[2].set_title("Longitude filled")

    # 4. Weighted interpolation
    pcm3 = axes[3].pcolormesh(filled_latlon_na.lon_rho, filled_latlon_na.lat_rho, filled_latlon_na,
                            shading="nearest", cmap="viridis")
    axes[3].set_title("Lat and Long weighted filling")

    # 5. Gaussian smoothing
    pcm4 = axes[4].pcolormesh(biomass_smooth.lon_rho, biomass_smooth.lat_rho, biomass_smooth,
                            shading="nearest", cmap="viridis")
    axes[4].set_title("Smoothed")

    # Shared formatting
    for ax in axes:
        ax.set_xlabel("Lon")

    plt.tight_layout()
    plt.show()

    # Run in parallel 
    biomass_interp_list = process_map(process_algorithm, np.arange(50), max_workers=10, desc='Interpolation Algo')

    # Concatenate results together
    biomass_interp = xr.concat(biomass_interp_list, dim="algo_bootstrap")
    biomass_interp = biomass_interp.assign_coords(lon_rho=(("eta_rho", "xi_rho"), area_SO.lon_rho.values),
                                                  lat_rho=(("eta_rho", "xi_rho"), area_SO.lat_rho.values))
    
    # Back to daily dataset
    months_order = np.array([11, 12, 1, 2, 3, 4])
    days_per_month = np.array([30, 31, 31, 28, 31, 30])
    day_to_month = np.concatenate([np.repeat(m, d) for m, d in zip(months_order, days_per_month)])
    assert day_to_month.size == 181
    day_to_month_xr = xr.DataArray(day_to_month, dims="days", name="months")
    biomass_interp_daily = biomass_interp.sel(months=day_to_month_xr)
    biomass_interp_daily = biomass_interp_daily.transpose('days', 'algo_bootstrap', 'eta_rho', 'xi_rho')  #shape (181, 50, 231, 1442)

    # To Dataset
    biomass_interp_daily_ds = biomass_interp_daily.to_dataset(name="euphausia_biomass")
    biomass_interp_daily_ds = biomass_interp_daily_ds.reset_index(["algo_bootstrap"])
    biomass_interp_daily_ds.attrs.update({"description": "Biomass of Euphausia superba assuming 80% of total krill biomass",
                                          "regridding": "Nearest neighbors, using bilinear method.",
                                          "interpolation": "Nan values filled using linear trend fits along latitude and longitude.\n"
                                                           "Latitude- and longitude-based estimated values are combined with inverse distance weighting.\n"
                                                           "Finally, Gaussian smoothing (with sigma = 1).",
                                          "units": "mg C m-3",})

    # Save to file
    biomass_interp_daily_ds.to_netcdf(output_file_biomass_regrid_interp, engine="netcdf4")

    visualisation = True
    if visualisation:
        # ---- Median over longitude (just for plotting, small example)
        data_before = biomass_fixed.isel(algo_bootstrap=0, days=0).euphausia_biomass
        data_after = biomass_regridded.isel(algo_bootstrap=0, days=0).euphausia_biomass
        data_filled = biomass_interp_daily_ds.isel(algo_bootstrap=0, days=0).euphausia_biomass

        # ---- Figure setup
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.05)

        # Circular boundary
        theta = np.linspace(0, 2*np.pi, 200)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * 0.5 + 0.5)

        # Colorscale
        vmax = np.nanpercentile(data_before, 95)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        titles = ["Before regridding", "After regridding", "After Nan filling"]
        datasets = [data_before, data_after, data_filled]

        for i, (ax, data, title) in enumerate(zip(axes, datasets, titles)):
            ax.set_extent([0, 360, -90, -60], crs=ccrs.PlateCarree())
            ax.set_boundary(circle, transform=ax.transAxes)
            ax.set_anchor('C')

            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}
            gl.xlabels_top = False
            gl.ylabels_right = False
            if title != "Before regridding":
                gl.ylabels_left = False

            if i == 0:
                pcm = ax.pcolormesh(biomass_fixed.lon, biomass_fixed.lat, data,
                                    transform=ccrs.PlateCarree(), cmap="inferno", norm=norm)
            else:
                pcm = ax.pcolormesh(biomass_regridded.lon_rho, biomass_regridded.lat_rho, data,
                                    transform=ccrs.PlateCarree(), cmap="inferno", norm=norm)

            ax.set_title(title, fontsize=11)

        # Colorbar
        cbar = fig.colorbar(pcm, ax=axes, orientation='vertical', shrink=0.6, pad=0.05, extend='max')
        cbar.set_label("Biomass [mg C m$^{-3}$]", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        fig.suptitle("Euphausia superba biomass - day 0, bootstrap 0", fontsize=16, y=0.93)
        plt.show()


else:
    biomass_regridded = xr.open_dataset(output_file_biomass_regrid)
    biomass_regrid_interp = xr.open_dataset(output_file_biomass_regrid_interp)


# %% ====================== CEPHALOPOD biomass in the MPAS ======================
# This is use afterwards to select MHW events happening in area where Biomass exists.
mpas_ds =xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/MPA_mask.nc') #shape (434, 1440)
# South of 60°S
south_mask = (mpas_ds['lat_rho'] <= -60)
mpas_south60S =  mpas_ds.where(south_mask, drop=True) #shape (231, 1440)

mpa_masks = {"RS": ("Ross Sea", mpas_south60S.mask_rs),
             "SO": ("South Orkney Islands southern shelf", mpas_south60S.mask_o),
             "EA": ("East Antarctic", mpas_south60S.mask_ea),
             "WS": ("Weddell Sea", mpas_south60S.mask_ws),
             "AP": ("Antarctic Peninsula", mpas_south60S.mask_ap),}


# Mask biomass in the MPAs (for only 1 algo_bootstrap)
biomass_mpa_regrid_vars = {}
biomass_mpa_interp_vars = {}

for abbrv, (name, mask_2d) in mpa_masks.items():
    # test
    # abbrv, (name, mask_2d) = "RS", ("Ross Sea", mpas_south60S.mask_rs)

    print(f"Masking biomass for {name} ({abbrv})")

    # Ensure boolean mask
    mask_2d = mask_2d.astype(bool)

    # Select 1 boostrap
    biomass_regridded_1algo = biomass_regridded.isel(algo_bootstrap=0).euphausia_biomass
    biomass_regrid_interp_1algo = biomass_regrid_interp.isel(algo_bootstrap=0).euphausia_biomass

    # Reformat
    biomass_regridded_reformat = biomass_regridded_1algo.isel(xi_rho=slice(0, mask_2d.xi_rho.size))
    biomass_interp_reformat = biomass_regrid_interp_1algo.isel(xi_rho=slice(0, mask_2d.xi_rho.size))

    # Mask biomass (broadcast over days & bootstrap)
    biomass_regridded_mpa = biomass_regridded_reformat.where(mask_2d)
    biomass_interp_mpa = biomass_interp_reformat.where(mask_2d)

    biomass_mpa_regrid_vars[f"biomass_{abbrv}"] = biomass_regridded_mpa
    biomass_mpa_interp_vars[f"biomass_{abbrv}"] = biomass_interp_mpa

# To Datasets
biomass_mpa_regrid_ds = xr.Dataset(data_vars=biomass_mpa_regrid_vars,
                                   coords={"days": biomass_regridded_reformat.days,
                                           "lon_rho": (("eta_rho", "xi_rho"), mpas_south60S.lon_rho.data),
                                           "lat_rho": (("eta_rho", "xi_rho"), mpas_south60S.lat_rho.data), },
                                   attrs={"description": "Euphausia superba biomass masked to MPAs (south of 60°S)",
                                          "algo_bootstrap": 0,
                                          "regridding": biomass_regridded.attrs.get("regridding", ""),
                                          "units": "mg C m-3",
                                          "note": "Biomass available only inside each MPA; NaN elsewhere.",})


biomass_mpa_interp_ds = xr.Dataset(data_vars=biomass_mpa_interp_vars,
                                   coords={"days": biomass_interp_reformat.days,
                                           "lon_rho": (("eta_rho", "xi_rho"), mpas_south60S.lon_rho.data),
                                           "lat_rho": (("eta_rho", "xi_rho"), mpas_south60S.lat_rho.data),},
                                   attrs={"description": "Interpolated Euphausia superba biomass masked to MPAs (south of 60°S)",
                                          "algo_bootstrap": 0,
                                          "regridding": biomass_regrid_interp.attrs.get("regridding", ""),
                                          "units": "mg C m-3",
                                          "note": "Biomass available only inside each MPA; NaN elsewhere.",})

# Save to file
out_regrid = os.path.join(path_cephalopod, "biomass_regridded_1algo_mpa.nc")
out_interp = os.path.join(path_cephalopod, "biomass_regridded_interp_1algo_mpa.nc")
biomass_mpa_regrid_ds.to_netcdf(out_regrid)
biomass_mpa_interp_ds.to_netcdf(out_interp)


# %% ====================== Spread of models ======================
# --- Prepare data 
# Volume grid cells (ROMS)
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc') #in km3

# Mean volume on the first 100m depth
# volume_roms_surf = volume_roms['volume'].isel(z_rho=0)
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 

# Mask latitudes south of 60°S
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)

# Total Biomass
total_biomass_mgC = (biomass_regrid_interp.euphausia_biomass * volume_60S_SO_100m * 1e9).sum(dim=("eta_rho","xi_rho")) #shape: (days:181, algo_bootstrap:50)

# Convert metric 
total_biomass_tons = total_biomass_mgC / (1e9)
total_biomass_Mt = total_biomass_mgC / (1e9*1e6)

days_in_month = [30, 31, 31, 28, 30, 31]  # Nov, Dec, Jan, Feb, Mar, Apr
month_labels = ["Nov","Dec","Jan","Feb","Mar","Apr"]
month_edges = np.cumsum([0]+days_in_month)

# Assign month index to each day
month_idx = np.zeros(total_biomass_Mt.days.size, dtype=int)
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
        data = total_biomass_Mt.isel(days=days_sel, algo_bootstrap=slice(start, end)).values.flatten()
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
ax.set_ylabel("Total Biomass [Mt C]", fontsize=12)
ax.set_xlabel("Months", fontsize=12)
ax.set_title("Euphausia superba total biomass in the Southern Ocean\nSpread of models and bootstraps per month", fontsize=14)

# Legend
ax.legend(handles=handles, loc='upper center', title="Medians", fontsize=11, ncol=3, title_fontsize=12)

plt.tight_layout()
plt.show()



# %% ================================= Convert to Density =================================
# --- Set up and data
proportion = {'juvenile': 0.20, 'immature': 0.3, 'mature': 0.3, 'gravid':0.2}
clim_krillmass_SO = xr.open_dataset(os.path.join(path_masslength, "clim_mass_stages_SO.nc")) #shape (181, 231, 1442)

# --- Area ROMS south 60°S
area_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/area.nc') #in km2
area_SO_surf = area_roms['area'].isel(z_t=0)
area_60S_SO = area_SO_surf.where(area_roms['lat_rho'] <= -60, drop=True)

# --- Volume ROMS south 60°S
volume_roms =  xr.open_dataset('/home/jwongmeng/work/ROMS/scripts/coords/volume.nc') #in km3
volume_roms_100m = volume_roms['volume'].isel(z_rho=slice(0, 14)).sum(dim='z_rho') 
volume_60S_SO_100m = volume_roms_100m.where(volume_roms['lat_rho'] <= -60, drop=True)


# --- Conversion
print(f'Original: {biomass_regridded.isel(days=0, algo_bootstrap=0, eta_rho=200, xi_rho=1000).euphausia_biomass.values:.3f} mgC/m3')
# Step1. Carbon fraction in Euphausia Superba -- Data from Farber-Lorda et al. (2009)
C_frac_stage = {'juvenile': (0.4989, 0.0250), 'males': (0.4756, 0.0266), 'mature females': (0.4756, 0.0297), 'spent females': (0.5299, 0.0267),} # [fraction C / mg biomass]
mean_C_fraction = np.mean(np.array([v[0] for v in C_frac_stage.values()]))
propagated_sd = np.sqrt(np.sum(np.array([v[1] for v in C_frac_stage.values()])**2) / len(np.array([v[1] for v in C_frac_stage.values()])))
print(f"Mean C fraction: {mean_C_fraction:.4f}")
print(f"Propagated SD: {propagated_sd:.4f}")

# From carbon mass to dry mass
dry_biomass = biomass_regridded / mean_C_fraction #[mg/m3]
print(f'Step1: {dry_biomass.isel(days=0, algo_bootstrap=0, eta_rho=200, xi_rho=1000).euphausia_biomass.values:.3f} mg/m3')

# Step2. Volume and area
# From mg/m3 to mg/m2
biomass_mgm2 = dry_biomass.euphausia_biomass * (volume_60S_SO_100m * 1e9) / (area_60S_SO * 1e6) #shape (181, 50, 231, 1442)
print(f'Step2: {biomass_mgm2.isel(days=0, algo_bootstrap=0, eta_rho=200, xi_rho=1000).values:.3f} mg/m2')


# Step3. Krill mass
# Krill mean mass [mg]
mean_mass = sum(clim_krillmass_SO[stage] * proportion[stage] for stage in proportion) # shape (181, 231, 1442)

# Mean over algorithms
biomass_mgm2_mean = biomass_mgm2.mean(dim="algo_bootstrap")# shape (181, 231, 1442)

# To ind/m2
krill_density_daily = biomass_mgm2_mean / mean_mass
print(f'Step3: {krill_density_daily.isel(days=0, eta_rho=200, xi_rho=1000).values:.3f} ind/m2')

krill_density_monthly = krill_density_daily.groupby("months").mean(dim="days") # shape (months: 6, 231, 1442)

# %% ================================= Plot Density concentration =================================
# ===== Month selection =====
months_sel = [11, 12, 1, 2, 3, 4]
month_labels = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
nplots = len(months_sel)

# ===== Custom colormap =====
vmin, vmax= np.nanpercentile(krill_density_monthly, [5, 95])

# ===== Figure layout =====
ncols = 3
nrows = int(np.ceil(nplots / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.2 * nrows), subplot_kw=dict(projection=ccrs.SouthPolarStereo()))
fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.92, wspace=0.05, hspace=0.1)
axes = np.atleast_1d(axes).flatten()

# ===== Circular boundary =====
theta = np.linspace(0, 2 * np.pi, 200)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + 0.5)

# ===== Plot =====
for i, (m, label) in enumerate(zip(months_sel, month_labels)):
    ax = axes[i]

    data = krill_density_monthly.sel(months=m)

    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=3)

    pcm = ax.pcolormesh(data.lon_rho, data.lat_rho, data.values, transform=ccrs.PlateCarree(),
                        cmap='YlGnBu_r', vmin=vmin, vmax=vmax, shading="auto")

    # Gridlines
    gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.7, linestyle="--", linewidth=0.4)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 7, 'rotation': 0}
    gl.ylabel_style = {'size': 7, 'rotation': 0}

    ax.set_title(label, fontsize=11)

# ===== Remove unused axes =====
for ax in axes[nplots:]:
    ax.remove()

# ===== Shared colorbar & title =====
cbar = fig.colorbar(pcm, ax=axes[:nplots], orientation="vertical", shrink=0.8, pad=0.05, extend="both")
cbar.set_label("Density [ind$\cdot$ $m^{-2}$]", fontsize=14)
cbar.ax.tick_params(labelsize=12)

fig.suptitle("Euphausia superba density (approximation)\nMedian over algorithms and bootstraps", fontsize=16, y=1.03, x=0.4)
plt.show()



# %%
