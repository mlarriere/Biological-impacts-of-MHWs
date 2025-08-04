import numpy as np
import xarray as xr

def growth_Atkison2006(chla_data, temp_data, maturity_stage=None, length=None):
    """
    Computes daily krill growth rate using Atkinson et al. (2006).
    If maturity_stage is given, a default a-term is used,
    but growth is still calculated using the actual body length provided.
    """

    if maturity_stage:
        if maturity_stage == 'juvenile':
            a = -0.158
        elif maturity_stage == 'immature':
            a = -0.192
        elif maturity_stage == 'mature':
            a = np.mean([-0.196, -0.216])
        elif maturity_stage == 'gravid':
            a = -0.216
        else:
            raise ValueError(f"Unknown maturity stage: {maturity_stage}")
        if length is None:
            raise ValueError("Length must be provided if using maturity_stage")
    elif length is not None:
        a = -0.192  # Default intercept
    else:
        raise ValueError("Either 'maturity_stage' or 'length' must be provided.")

    # Length terms
    b, c = 0.00674, -0.000101

    # Food and temperature terms
    d, e = 0.377, 0.321
    f, g = 0.013, -0.0115

    growth = a + b * length + c * length**2 + (d * chla_data) / (e + chla_data) + f * temp_data + g * temp_data**2

    return growth

def length_Atkison2006(chla, temp, initial_length, intermoult_period=10, maturity_stage=None):
    """
    Simulates krill length over time using the Atkinson et al. (2006) model.
    Accepts either a maturity_stage or just an initial length.
    """
   
    n_days = chla.sizes['days'] #181
    is_gridded = 'eta_rho' in chla.dims #True

    if is_gridded:
        shape = chla.isel(days=0).shape
        length = xr.DataArray(np.full((n_days, *shape), np.nan),
                              dims=("days", "eta_rho", "xi_rho"),
                              coords={"days": chla.days, "lat_rho": chla.lat_rho, "lon_rho": chla.lon_rho})
    else:
        length = xr.DataArray(np.full((n_days,), np.nan), dims=("days",), coords={"days": chla.days})

    length[0] = initial_length

    for t in range(1, n_days):
        length[t] = length[t - 1]  # Default: no growth

        if t % intermoult_period == 0 and t >= intermoult_period:
            chl_slice = chla.isel(days=slice(t - intermoult_period, t))
            tmp_slice = temp.isel(days=slice(t - intermoult_period, t))

            valid_mask = (
                (chl_slice.count(dim='days') == intermoult_period)
                & (tmp_slice.count(dim='days') == intermoult_period)
            )
            # Ensure boolean and no NaNs in mask
            valid_mask = valid_mask.fillna(False).astype(bool)

            # print(f"Day {t}: valid_mask sum = {valid_mask.sum().item()}, shape = {valid_mask.shape}")

            if valid_mask.any():
                chl_mean = chl_slice.mean(dim='days', skipna=True)
                tmp_mean = tmp_slice.mean(dim='days', skipna=True)

                growth = growth_Atkison2006(
                    chla_data=chl_mean,
                    temp_data=tmp_mean,
                    maturity_stage=maturity_stage,
                    length=length[t - 1]
                )

                new_len = length[t - 1] + growth

                # Check shapes before where
                # print(f"Day {t}: length[t-1] shape: {length[t - 1].shape}, growth shape: {growth.shape}, new_len shape: {new_len.shape}")

                length[t] = xr.where(valid_mask & (new_len > 0), new_len, length[t - 1])

                # mean_len_valid = length[t].where(valid_mask).mean().item()
                # print(f"Day {t}: Mean length (valid pixels only): {mean_len_valid:.3f}")

            else:
                # print(f"Day {t}: No valid pixels found.")
                length[t] = length[t - 1]


    return length
