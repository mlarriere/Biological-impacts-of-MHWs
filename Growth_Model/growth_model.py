import numpy as np
import xarray as xr

def growth_Atkison2006(chla_data, temp_data):
    '''Function to calculate growth according to Atkinson et al. (2006) for given chlorophyll and temperature data'''

    # Defining constants
    # ---- Coefficients of models predicting DGR and GI from length, food, and temperature in Eq. 4 (Atkinson et al., 2006), Here we use model4, i.e. sex and maturity considered (krill length min 35mm)
    a, std_a= np.mean([-0.196, -0.216]), 0.156  # constant term. mean value between males and mature females 

    # Length
    b, std_b = 0.00674,  0.00611 #linear term 
    c, std_c = -0.000101, 0.000071 #quadratic term 

    # Food
    d, std_d = 0.377, 0.087 #maximum term
    e, std_e = 0.321, 0.232 #half saturation constant

    # Temperature
    f, std_f = 0.013, 0.0163 #linear term
    g, std_g = -0.0115, 0.00420 #quadratic term 

    # H = #random effect for unexplained variation

    length=35 # mean body length in adult krill (Michael et al. 2021 / Tarling 2020)
    # print(type(a), type(b), type(c), type(d), type(e), type(f), type(g))
    # print(type(length))
    
    # Growth equation 
    growth = a + b* length + c *length**2 + (d*chla_data)/(e+chla_data) + f*temp_data + g*temp_data**2 #[mm] -daily

    return growth

print("Before defining length_Atkison2006")

def length_Atkison2006(chla, temp, initial_length, intermoult_period= 10):
    # Get dimensions
    n_days = chla.sizes['days'] #181 days
    shape = chla.isel(days=0).shape #(106, 161)

    # Initialisation - dataarray to store length
    length = xr.DataArray(np.full((n_days, *shape), np.nan), 
                          dims=("days", "eta_rho", "xi_rho"),
                          coords={"days": chla.days, "lat_rho": chla.lat_rho, "lon_rho": chla.lon_rho})

    # First step -- initial length (hypothethis - on Nov1st krill length = 35mm)
    length[0] = initial_length

    # Simulate growth day by day
    for t in range(1, n_days):
        # Only apply growth at the end of the intermoult period
        if t >= intermoult_period and t % intermoult_period == 0:
            # Average CHLA and TEMP over the last intermoult_period days
            chl_period = chla.isel(days=slice(t - intermoult_period, t))
            tmp_period = temp.isel(days=slice(t - intermoult_period, t))
            
            # Compute mean over time
            chl_mean = chl_period.mean(dim='days')
            tmp_mean = tmp_period.mean(dim='days')

            prev_len = length[t-1]
            growth = growth_Atkison2006(chl_mean, tmp_mean)  # Growth model - Eq. 4 (Atkinson et al. 2006)

            new_len = prev_len + growth

            # Check if length <= 0 => if so, krill die, set length to np.nan
            new_len = xr.where(new_len <= 0, np.nan, new_len)

            length[t] = new_len

        else:
            length[t] = length[t - 1]

    return length

print("length_Atkison2006 defined")
