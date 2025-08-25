# Biological Impacts of Marine Heatwaves (MHWs)

## Overview
This repository contains the code developed for a **master’s thesis** on the biological impacts of **Marine Heatwaves (MHWs)** on **Antarctic krill** in the Southern Ocean.  
The project investigates how environmental drivers, such as **chlorophyll-a concentrations** and **temperature**, influence the **daily growth rate of krill**, and evaluates the impact of MHWs on their growth.


## Repository Structure
The repository is organized into three main sections:

### 1. Marine_HeatWaves
Code to define **MHW events** and compute their characteristics, such as frequency, intensity and duration. We based our MHWs definition on the work of **Hobday et al. (2016)**

### 2. Growth_Model
Scripts implementing the **krill growth model**. Here we used the empirical model by **Atkinson et al. (2006)** and computes daily growth rates using environmental drivers simulated using ROMS model.

### 3. Biomass
Scripts for **krill biomass computation**, including krill population definition, length to mass relationship, biomass density calculations and biomass estimation.

## Usage
- Ensure all dependencies are installed (Python packages)
- Follow the order:  
  1. Compute MHW characteristics  
  2. Run growth model  
  3. Calculate biomass  

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
