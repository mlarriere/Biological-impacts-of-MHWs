"""
Created on Sund 18 Oct 14:22:45 2025

Functions for krill biomass from krill mass

@author: Marguerite Larriere (mlarriere)
"""

import numpy as np
import xarray as xr
from functools import partial
from tqdm.contrib.concurrent import process_map
import gc

# Population proportions
DEFAULT_PROPORTION = {
    "juvenile": 0.20,
    "immature": 0.30,
    "mature": 0.30,
    "gravid": 0.20,
}


# ---------------------------------------------------------
# Biomass for one bootstrap
# ---------------------------------------------------------
def biomass_k(k, growth_fact_pop, initial_biomass_ds):
    """Calculate daily biomass for one CEPHALOPOD bootstrap."""
    n_days = 181

    B0 = initial_biomass_ds["euphausia_biomass"].isel(days=0, bootstraps=k).values

    B_k = np.empty((n_days, *B0.shape), dtype=np.float32)
    B_k[0] = B0

    for t in range(1, n_days):
        B_k[t] = B_k[t - 1] * growth_fact_pop[t - 1]

    return B_k


# ---------------------------------------------------------
# Biomass evolution for one year
# ---------------------------------------------------------
def evolution_biomass_yr(year_idx, ds_mass, initial_biomass_ds, proportion):
    """Calculate biomass for one year (all bootstraps)."""
    # test
    # year_idx = 36
    # ds_mass = clim_krillmass_SO

    # -- Climatology case, i.e. no years dim
    has_years = 'years' in ds_mass.dims
    n_days = 181

    # -- Compute growth factor
    growth_fact_pop = np.zeros((n_days-1, ds_mass.eta_rho.size, ds_mass.xi_rho.size), dtype=np.float32) #shape (180, 231, 1442)
    for stage, p in proportion.items():
        if has_years:
            # Specific year
            ds_mass_yr = ds_mass[stage].isel(years=year_idx).values
        else:
            ds_mass_yr = ds_mass[stage].values # climatology  

        g_fact_stage = ds_mass_yr[1:] / ds_mass_yr[:-1] 
        growth_fact_pop += p * g_fact_stage # shape (180, 231, 1442)

    # -- Biomass timeseries for every models and algorithms
    # Loop over the algo and boostraps -- in parallel
    func = partial(biomass_k, growth_fact_pop=growth_fact_pop, initial_biomass_ds=initial_biomass_ds)
    results = process_map(func, range(10), max_workers=10, chunksize=1, desc='Bootstraps')

    # Extract data
    B_all = np.stack(results, axis=0) #shape (10, 181, 231, 1442)
    
    # To Datsaset
    biomass_ds = xr.Dataset(data_vars=dict(biomass=(("bootstraps", "days", "eta_rho", "xi_rho"), B_all)),
                            coords=dict(bootstraps=np.arange(10), 
                                        days=ds_mass.days,
                                        lat_rho=ds_mass.lat_rho,
                                        lon_rho=ds_mass.lon_rho,))
    
    biomass_ds.attrs.update({"Description": "Evolution of krill biomass weighted according to population proportions.\n"\
                                            "Biomass time series were computed for all 50 algorithm–bootstrap "\
                                            "realisations of the Cephalopod model. For each algorithm, the "\
                                            "bootstrap realisations were then averaged to obtain a single time series per algorithm.",
                            "Units": "mg/m3",})
    
    
    # Clean memory
    del B_all
    gc.collect()

    return biomass_ds


# ---------------------------------------------------------
# Biomass time series for one environmental scenario
# ---------------------------------------------------------
import os
def compute_biomass_surrogates(ds_mass, initial_biomass_ds,  label, proportion, output_file, mean_C_fraction, propagated_sd, description, max_workers=10):
    # test
    # max_workers=10
    # proportion=proportion
    # ds_mass=clim_krillmass_SO_expanded
    # label='Climatology'  
    # label_da='clim'  
    # output_folder= os.path.join(path_biomass_ts_SO)
    
    print(f"  → {label}")
    
    # output_file= os.path.join(output_folder, f'biomass_{label_da}.nc')

    if not os.path.exists(output_file):
        # Prepare function
        func = partial(evolution_biomass_yr, ds_mass=ds_mass, initial_biomass_ds=initial_biomass_ds, proportion=proportion)

        # -- Climatology case, i.e. no years dim
        if 'years' not in ds_mass.dims:
            raise ValueError(f"ds_mass for '{label}' has no 'years' dimension.")

        # Run in parallel
        B_algo_da = process_map(func, range(39), max_workers=max_workers, chunksize=1, desc=f"{label} | Biomass ") #len = nyears and shape (10, 181, 231, 1442)

        # Extract data
        B_algo_all_ds = xr.concat(B_algo_da, dim="years")  # shape: (39, 10, 181, 231, 1442)

        # Put together into dataset
        B_algo_all_ds.attrs.update({"Surrogate": label,
                                    "Description": description,
                                    "Population Proportions": ", ".join(f"{k} : {v*100:.0f}%" for k, v in proportion.items()),
                                    "Initial Biomass": "Cephalopod output for krill total (euphausia biomass ~80% of it).\nRegridded to ROMS resolution (0.25°) and interpolated.",
                                    "Assumptions": "Fixed stage proportions, no mortality, no recruitment, no stage transitions within a growth season.",
                                    "Units": "mg/m3",
                                    'Conversion to dry weight': f"Using mean C fraction of {mean_C_fraction:.4f} (SD: {propagated_sd:.4f}) from Farber-Lorda et al. (2009).",})
            
        # Save to file
        B_algo_all_ds.to_netcdf(output_file, mode="w", engine="netcdf4")
        
        # Clean memory
        del B_algo_da
        gc.collect()


    else:
        print(f"  → {label} already exists, skipping.")
        return