"""
Created on Sund 18 Oct 09:13:45 2025

Functions for krill length and mass calculations

@author: Marguerite Larriere (mlarriere)
"""

import numpy as np
import xarray as xr

from functools import partial
from tqdm.contrib.concurrent import process_map
import os
from datetime import datetime

from B_Growth_Model.Atkinson2006_model import length_Atkinson2006

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
STAGE_LENGTHS = {
    "juvenile": 25,
    "immature": 30,
    "mature": 40,
    "gravid": 45
    }

STAGE_LIST = list(STAGE_LENGTHS.keys())

# ---------------------------------------------------------
# Length-to-mass conversion
# ---------------------------------------------------------
def length_to_mass(length_array, p=10**(-4.19), r=3.89):
    """Convert krill length to mass in mg."""
    return p * length_array**r


# ---------------------------------------------------------
# Calculate length and mass for one year
# ---------------------------------------------------------
def length_single_year(yr, stage, chla_ds, temp_ds, sensitivity_param=None, sensitivity=0):
    """
    Calculate krill length and mass for one year and maturity stage.
    """
    chla = chla_ds.chla.isel(years=yr)
    temp = temp_ds.avg_temp.isel(years=yr)

    length = length_Atkinson2006(chla=chla, temp=temp, initial_length=STAGE_LENGTHS[stage], maturity_stage=stage, sensitivity_param=sensitivity_param, sensitivity=sensitivity) #(181, 231, 1442)
    mass = length_to_mass(length)

    return length, mass


# ------------------------------------------------------------------
# Calculate length and mass across all years in one world/surrogates
# ------------------------------------------------------------------
def compute_world(stage, chla_ds, temp_ds, label, max_workers=4, sensitivity_param=None, sensitivity=0):
    """
    Calculate daily length and mass for one stage and for each grid cell under one environmental scenario.

    Returns:
    length_out_da, mass_out_da : xarray.DataArray --- Dimensions: years, days, eta_rho, xi_rho
    """
     
    print(f"  → {label} | "
        f"{sensitivity_param or 'baseline'} | {sensitivity:+.0%}")

    # Compute length and mass - results=list of length 39
    func = partial(length_single_year, stage=stage, chla_ds=chla_ds, temp_ds=temp_ds, sensitivity_param=sensitivity_param, sensitivity=sensitivity)
    results = process_map(func, range(39), max_workers=max_workers, chunksize=1, desc=f"{label} | {stage}")

    # Initialisation
    nyears, nday, neta, nxi = 39, 181, 231, 1442
    length_out_da = xr.DataArray(np.empty((nyears, nday, neta, nxi), dtype=np.float32),
                                 dims=("years", "days", "eta_rho", "xi_rho"),
                                 coords={"years": np.arange(nyears),
                                         "lat": chla_ds.lat_rho,
                                         "lon": chla_ds.lon_rho,},)

    mass_out_da = xr.DataArray(np.empty((nyears, nday, neta, nxi), dtype=np.float32),
                               dims=("years", "days", "eta_rho", "xi_rho"),
                               coords=length_out_da.coords,)

    for y, (length, mass) in enumerate(results):
        length_out_da[y, :, :, :] = length.values.astype(np.float32)
        mass_out_da[y, :, :, :]   = mass.values.astype(np.float32)

        del length, mass

    return length_out_da, mass_out_da


# -------------------------------------------------------------
# Calculate all environmental scenarios for one maturity stage
# -------------------------------------------------------------
def compute_stage(stage, worlds, sensitivity_param=None, sensitivity=0):
    results = {}

    for world_key, (chla_ds, temp_ds, label, workers) in worlds.items():

        length, mass = compute_world(stage=stage, chla_ds=chla_ds, temp_ds=temp_ds, label=label, max_workers=workers,
                                     sensitivity_param=sensitivity_param, sensitivity=sensitivity,)

        results[f"{world_key}_length"] = length
        results[f"{world_key}_mass"] = mass

    return results


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------
def save_if_not_exists(ds, filepath, description=None, temperature=None, chla = None, units=None):
    """
    Save xarray Dataset to NetCDF.
    """
     
    # create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # if the file exists, remove it 
    if os.path.exists(filepath):
        print(f"Removing existing file: {filepath}")
        os.remove(filepath)

    # Metadata
    if description is not None:
        ds.attrs["description"] = description
    if units is not None:
        ds.attrs["units"] = units
    if temperature is not None:
        ds.attrs["temperature"] = temperature
    if chla is not None:
        ds.attrs["chla"] = chla
    ds.attrs["date"] = datetime.today().strftime('%d %B %Y')

    # Write
    ds.to_netcdf(filepath, mode="w", engine="netcdf4")
    print(f"Saved: {filepath}")