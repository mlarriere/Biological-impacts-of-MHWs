'''
Created on Friday 23 October 12:40 2026

Functions for attribution of seasonal biomass changes

@author: Marguerite Larriere
'''


import numpy as np
import xarray as xr

from functools import partial
from tqdm.contrib.concurrent import process_map
import os
from datetime import datetime


# ------------------------------------------------------------------------------------------------------
#                                      Southern Ocean functions
# -------------------------------------------------------------------------------------------------------

def compute_attributions_SO(biomass, surrogate_names, output_dir):
    """
    Compute and save: 1. Seasonal biomass gain      
                      2. Biomass change relative to climatology
                      3. Attribution of MHWs, warming, and residual
    for the full Southern Ocean.
    """
    # Paths
    seasonal_gain_path = os.path.join(output_dir, "seasonal_gain_SO.nc")
    change_biomass_path = os.path.join(output_dir, "change_biomass_SO.nc")
    attributions_path = os.path.join(output_dir, "attributions_SO.nc")

    # Step 1: Seasonal biomass gain
    if os.path.exists(seasonal_gain_path):
        print("Loading existing seasonal biomass gain...")
        with xr.open_dataset(seasonal_gain_path) as ds:
            seasonal_gain_ds = ds.load()
    else:
        print("Computing seasonal biomass gain...")
        seasonal_gain = {}

        for surrog in surrogate_names:
            seasonal_gain[surrog] = (biomass[surrog].isel(days=-1) - biomass[surrog].isel(days=0))["biomass"]

        seasonal_gain_ds = xr.Dataset(seasonal_gain)
        seasonal_gain_ds.to_netcdf(seasonal_gain_path)
        print(f"Saved: {seasonal_gain_path}")

    # Step 2: Change relative to climatology
    if os.path.exists(change_biomass_path):
        print("Loading existing biomass change...")
        with xr.open_dataset(change_biomass_path) as ds:
            change_biomass_ds = ds.load()
    else:
        print("Computing biomass change relative to climatology...")
        change_biomass = {}

        for surrog in surrogate_names:
            if surrog != "clim":
                change_biomass[surrog] = ((seasonal_gain_ds[surrog] - seasonal_gain_ds["clim"]) / seasonal_gain_ds["clim"])

        change_biomass_ds = xr.Dataset(change_biomass)
        change_biomass_ds.to_netcdf(change_biomass_path)
        print(f"Saved: {change_biomass_path}")

    # Step 3: Attribution
    if os.path.exists(attributions_path):
        print("Loading existing SO attributions...")
        with xr.open_dataset(attributions_path) as ds:
            impact_ds = ds.load()
    else:
        print("Computing SO attributions...")

        impact_mhws = (change_biomass_ds["actual"] - change_biomass_ds["climtrend"])
        impact_warming = (change_biomass_ds["actual"] - change_biomass_ds["nowarming"])
        residual = ( change_biomass_ds["actual"] - impact_mhws - impact_warming)

        impact_ds = xr.Dataset({"attrib_mhws": impact_mhws,
                                "attrib_warming": impact_warming,
                                "residual": residual})

        impact_ds.to_netcdf(attributions_path)
        print(f"Saved: {attributions_path}")

    return seasonal_gain_ds, change_biomass_ds, impact_ds

def mask_attributions_MHW_SO(path_combined_thesh, output_dir):
    """ Biomass change and attribution for MHW cells only (masking) """

    os.makedirs(output_dir, exist_ok=True)

    # Input files
    fpath_biomass_change = os.path.join(output_dir, "change_biomass_SO.nc")
    fpath_attributions = os.path.join(output_dir, "attributions_SO.nc")

    # Output files
    fpath_bio = os.path.join(output_dir, "biomass_change_wrt_clim_mhws_cells.nc")
    fpath_attrib_mhws = os.path.join(output_dir, "attrib_mhws_mhws_cells.nc")
    fpath_attrib_warming = os.path.join(output_dir, "attrib_warming_mhws_cells.nc")
    fpath_residuals = os.path.join(output_dir, "residuals_attrib_mhws_cells.nc")

    output_files = [fpath_bio, fpath_attrib_mhws, fpath_attrib_warming, fpath_residuals]

    # If files already exists, load datasets
    if all(os.path.exists(f) for f in output_files):
        print("Loading existing files...")

        with xr.open_dataset(fpath_attrib_mhws) as ds:
            mhws_ds = ds.load()

        with xr.open_dataset(fpath_attrib_warming) as ds:
            warming_ds = ds.load()

        with xr.open_dataset(fpath_residuals) as ds:
            residual_ds = ds.load()

        with xr.open_dataset(fpath_bio) as ds:
            bio_ds = ds.load()

        return mhws_ds, warming_ds, residual_ds, bio_ds

    # -- Load datasets to mask
    # Biomass change relative to climatology
    with xr.open_dataset(fpath_biomass_change) as ds:
        change_biomass_ds = ds.load()

    # Attributions
    with xr.open_dataset(fpath_attributions) as ds:
        impact_ds = ds.load()

    # Biomass change with 'reference' environmental variables
    bio = change_biomass_ds["actual"]

    # -- Load MHW
    print("Loading MHW event masks...")
    mhw_fpath = os.path.join(path_combined_thesh, "duration_AND_thresh_5mSEASON.nc")
    with xr.open_dataset(mhw_fpath) as mhw_events_surface:

        # Base mask: cells with MHW events
        duration_mask = mhw_events_surface["duration"] > 0

        # Detection masks for each absolute threshold
        det = {i: mhw_events_surface[f"det_{i}deg"] == 1 for i in [1, 2, 3, 4]}

        # Cells that experienced each MHW at least once
        mhws_masks = {"90perc": duration_mask.any(dim="days"), **{f"{i}deg": (duration_mask & det[i]).any(dim="days") for i in [1, 2, 3, 4]}}

        # Load masks into memory before closing the file
        mhws_masks = {name: mask.astype(bool).load() for name, mask in mhws_masks.items()}


    # Apply masks to the attribution and biomass datasets
    masked_attributions = {varname: {f"{varname}_{suffix}": impact_ds[varname].where(mask) for suffix, mask in mhws_masks.items()} for varname in ["attrib_mhws", "attrib_warming", "residual"]}
    biomass_dict = {f"biomass_{suffix}": bio.where(mask) for suffix, mask in mhws_masks.items()}

    # Create output datasets
    mhws_ds = xr.Dataset(data_vars=masked_attributions["attrib_mhws"],
                         attrs={"description": ("Contribution of MHWs to seasonal biomass change, only for cells affected by MHWs. i°C= absolute threshold.")})

    warming_ds = xr.Dataset(data_vars=masked_attributions["attrib_warming"],
                            attrs={"description": ("Contribution of long-thermal trend to seasonal biomass change, only for cells affected by MHWs. i°C= absolute threshold")})

    residual_ds = xr.Dataset(data_vars=masked_attributions["residual"],
                             attrs={"description": ("Residual, i.e. potential Chla contributions, only for cells affected by MHWs. i°C= absolute threshold.")})

    bio_ds = xr.Dataset(data_vars=biomass_dict,
                        attrs={"description": ("Seasonal biomass change relative to climatology, only for cells affected by MHWs. i°C= absolute threshold.")})

    # Save output datasets
    mhws_ds.to_netcdf(fpath_attrib_mhws)
    warming_ds.to_netcdf(fpath_attrib_warming)
    residual_ds.to_netcdf(fpath_residuals)
    bio_ds.to_netcdf(fpath_bio)

    print("Saved all MHW-masked datasets (Southern Ocean).")

    return mhws_ds, warming_ds, residual_ds, bio_ds

# ------------------------------------------------------------------------------------------------------
#                                            MPAs functions
# -------------------------------------------------------------------------------------------------------
def seasonal_gain_mpa(args):
    """Compute seasonal gain for all surrogates of one MPA and save to netCDF."""
    abbrv, mpa_masks, biomass_mpas, surrogate_names, output_dir = args

    mpa_name, _ = mpa_masks[abbrv]
    fpath = os.path.join(output_dir, f'seasonal_gain_{abbrv}.nc')

    if not os.path.exists(fpath):
        gain = {}
        for surrog in surrogate_names.keys():
            gain[surrog] = (biomass_mpas[abbrv][surrog].isel(days=-1) - biomass_mpas[abbrv][surrog].isel(days=0)).biomass  # shape (39, 10, 231, 1440)

        xr.Dataset(gain).to_netcdf(fpath)
        print(f"{abbrv} -- computed and saved.")

    ds = xr.open_dataset(fpath)
    result = {surrog: ds[surrog] for surrog in surrogate_names}
    ds.close()
    return abbrv, result

def biomass_change_mpa(args):
    """Fractional change relative to climatology per grid cell."""
    abbrv, seasonal_gain_mpa, surrogate_names, output_dir = args

    fpath = os.path.join(output_dir, f'bio_change_{abbrv}.nc')

    if not os.path.exists(fpath):
        bio_change = {}
        for surrog in [s for s in surrogate_names if s != "clim"]:
            bio_change[surrog] = ((seasonal_gain_mpa[abbrv][surrog] - seasonal_gain_mpa[abbrv]['clim']) / seasonal_gain_mpa[abbrv]['clim'])  # shape (39, 10, eta_rho, xi_rho)

        xr.Dataset(bio_change).to_netcdf(fpath)
        print(f"[{abbrv}] Computed and saved.")

    ds = xr.open_dataset(fpath)
    result = {surrog: ds[surrog] for surrog in surrogate_names if surrog != "clim"}
    ds.close()
    print(f"[{abbrv}] Loaded.")
    return abbrv, result

def contribution_mpa(args):
    """Compute MHW and warming contributions for one MPA."""
    abbrv, bio_change_interp_mpa, output_dir = args

    fpath = os.path.join(output_dir, f'attrib_{abbrv}.nc')

    if not os.path.exists(fpath):
        # abbrv='WS'

        contrib_mhws = bio_change_interp_mpa[abbrv]['actual'] - bio_change_interp_mpa[abbrv]['climtrend']  # shape (39, 10, 231, 1440)
        contrib_warming = bio_change_interp_mpa[abbrv]['actual'] - bio_change_interp_mpa[abbrv]['nowarming']  # shape (39, 10, 231, 1440)
        residual = bio_change_interp_mpa[abbrv]['actual'] - contrib_mhws - contrib_warming #residuals are mainly the chla 'contribution' according to the Atkinson model

        # Quick check 
        # check_sum = contrib_mhws.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200) + contrib_warming.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200) + residual.isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200)
        # print((check_sum-p_change_interp_mpa[abbrv]['actual'].isel(years=35, bootstraps=0, eta_rho=50, xi_rho=1200)).values) #should be close to 0

        ds_out = xr.Dataset(data_vars={'mhws': (contrib_mhws.dims, contrib_mhws.data,
                                                 {'description': 'Contribution of MHWs [%] to the change in seasonal krill biomass.'}),
                                        'warming': (contrib_warming.dims, contrib_warming.data,
                                                    {'description': 'Contribution of long-term warming [%] to the change in seasonal krill biomass.'}),
                                        'residual': (residual.dims, residual.data,
                                                     {'description': 'Residual term computed as reference - mhws - warming. Represents Chla contributions and an unexplained part.'})
                                        },
                            )
        ds_out.to_netcdf(fpath)
        # xr.Dataset({'mhws': contrib_mhws, 'warming': contrib_warming, 'residual': residual}).to_netcdf(fpath)
        print(f"[{abbrv}] Computed and saved.")

    ds = xr.open_dataset(fpath)
    contrib_mhws = ds['mhws']
    contrib_warming = ds['warming']
    residual = ds['residual']
    ds.close()
    print(f"[{abbrv}] Loaded.")

    return abbrv, {'mhws': contrib_mhws,
                   'warming': contrib_warming,
                   'residual': residual}


def mask_mhws_contribution_mpa(args):
    """Mask attribution and biomass-change datasets for MHWs cells for one MPA."""

    abbrv, mpa_masks, attrib_ds_mpa, bio_change_interp_mpa, path_combined_thesh = args

    print(f"     {mpa_masks[abbrv][0]}")

    # --- Load MHW masks
    fpath = os.path.join(path_combined_thesh, "mpas", "interpolated", f"duration_AND_thresh_{abbrv}.nc")

    with xr.open_dataset(fpath) as mhw:
        duration_mask = mhw["duration"] > 0
        det = {i: mhw[f"det_{i}deg"] == 1 for i in [1, 2, 3, 4]}

        mhws_masks = {
            "90perc": duration_mask.any("days"),
            "1deg": (duration_mask & det[1]).any("days"),
            "2deg": (duration_mask & det[2]).any("days"),
            "3deg": (duration_mask & det[3]).any("days"),
            "4deg": (duration_mask & det[4]).any("days"),
        }

        # Load coordinate data before closing the file
        lon_rho = mhw["lon_rho"].load()
        lat_rho = mhw["lat_rho"].load()

        # Load masks into memory
        mhws_masks = {name: mask.load() for name, mask in mhws_masks.items()}

    # --- Select attribution data for this MPA
    attrib_ds = attrib_ds_mpa[abbrv]
    bio = bio_change_interp_mpa[abbrv]["actual"]

    coords = {
        "lon_rho": lon_rho,
        "lat_rho": lat_rho,
        "years": attrib_ds["attrib_mhws"].years,
        "bootstraps": attrib_ds["attrib_mhws"].bootstraps,
        "eta_rho": attrib_ds["attrib_mhws"].eta_rho,
        "xi_rho": attrib_ds["attrib_mhws"].xi_rho,
    }

    # --- Build masked attribution datasets
    def build(varname, desc):
        data_vars = {f"{varname}_{suffix}": attrib_ds[varname].where(mask) for suffix, mask in mhws_masks.items()}

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "mpa": mpa_masks[abbrv][0],
                "description": (f"{desc} change in biomass; cells selected have experienced MHWs.\ni°C: Absolute threshold.")})

    ds_mhws = build("attrib_mhws", "Contribution of MHWs to")
    ds_warming = build("attrib_warming", "Contribution of long-term thermal trend to")
    ds_residual = build("residual", "Residuals of attributions to")

    # --- Build masked biomass dataset
    ds_bio = xr.Dataset(data_vars={f"biomass_{suffix}": bio.where(mask) for suffix, mask in mhws_masks.items()},
        coords=coords,
        attrs={"mpa": mpa_masks[abbrv][0],
               "description": ("Seasonal biomass change relative to climatology, only for cells affected by MHWs. i°C= absolute threshold")})

    return ds_mhws, ds_warming, ds_residual, ds_bio