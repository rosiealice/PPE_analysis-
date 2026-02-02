#!/usr/bin/env python
"""
Script to create videos of FATES timeseries from NorESM output files
Supports both global maps and point-based bar chart animations
Reads files individually from the lnd/hist directory and creates an animated visualization
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import shutil
import sys
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Point
from shapely.prepared import prep

# ============================================================================
# HELPER FUNCTIONS (Extracted to avoid duplication)
# ============================================================================

def find_nearest_point(target_lat, target_lon, lats, lons):
    """Find the nearest grid point to target lat/lon"""
    valid_mask = ~(np.isnan(lats) | np.isnan(lons))
    if not np.any(valid_mask):
        print("ERROR: All lat/lon coordinates are NaN!")
        return 0, np.nan, np.nan
    
    distances = np.full_like(lats, np.inf)
    distances[valid_mask] = np.sqrt((lats[valid_mask] - target_lat)**2 + 
                                    (lons[valid_mask] - target_lon)**2)
    idx = np.argmin(distances)
    return idx, lats[idx], lons[idx]

def get_country(lat, lon):
    """Get country name from lat/lon coordinates using cartopy"""
    try:
        import cartopy.io.shapereader as shpreader
        # Convert longitude to -180 to 180 range if needed
        if lon > 180:
            lon = lon - 360
        
        shpfilename = shpreader.natural_earth(resolution='110m',
                                             category='cultural',
                                             name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        countries = reader.records()
        point = Point(lon, lat)
        
        for country in countries:
            if prep(country.geometry).contains(point):
                return country.attributes['NAME']
        return "Unknown"
    except Exception as e:
        return "Unknown"

def get_us_state(lat, lon):
    """Get US state name from lat/lon coordinates using cartopy"""
    try:
        import cartopy.io.shapereader as shpreader
        # Convert longitude to -180 to 180 range if needed
        if lon > 180:
            lon = lon - 360
        
        # Load US states shapefile
        shpfilename = shpreader.natural_earth(resolution='110m',
                                             category='cultural',
                                             name='admin_1_states_provinces')
        reader = shpreader.Reader(shpfilename)
        states = reader.records()
        point = Point(lon, lat)
        
        for state in states:
            # Only check US states
            if state.attributes.get('admin') == 'United States of America':
                if prep(state.geometry).contains(point):
                    return state.attributes['name']
        return None  # Not in USA
    except Exception as e:
        print(f"Error getting US state: {e}")
        return None

def read_location_file(pft_name, max_sites=8):
    """Read location data from PFT-specific file
    
    Args:
        pft_name: Name of PFT (e.g., 'bet', 'bdt', 'ent')
        max_sites: Maximum number of sites to read
        
    Returns:
        List of (lat, lon, name) tuples
    """
    pft_file = f'/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/pft_grid_cells/{pft_name}_grid_cells.txt'
    locations = []
    
    try:
        with open(pft_file, 'r') as f:
            site_count = 0
            for line in f:
                line = line.strip()
                if not line or line.startswith('lat'):  # Skip empty lines and header
                    continue
                if site_count >= max_sites:
                    break
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        site_count += 1
                        locations.append((lat, lon, f"{pft_name.upper()}{site_count}"))
                    except ValueError:
                        continue
        return locations
    except FileNotFoundError:
        print(f"WARNING: File not found: {pft_file}")
        return []

# ============================================================================
# Configuration
DATA_PATH = "/datalake/NS9560K/rosief/"
CASENAME="i1850.ne16pg3_tn14.fatesnocomp.noresm3_0_beta08.CPLHIST.2025-12-18"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta08.CPLHIST_soilc_c4g_auto_d2bl1_c4MS_dmx_wsat.2026-01-07"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CRUJRA.2026-01-08/"
CASENAME="i1850.ne16pg3_tn14.fatesSP.noresm3_0_beta09.CPLHIST__BT3_LSP_MS3.25_BT1EBT.2026-01-23"
CASENAME="i2000.ne16pg3_tn14.fatesSP.noresm3_0_beta09.CPLHIST.2026-01-08"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CRUJRA_BT3_LSP_MS3.25_BT1EBT_bias28j.2026-01-28"
#CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.BT3_LSP_MS3.25_BT1EBT_bias30j_LU2000.2026-01-30"
#CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CRUJRA_BT3_LSP_MS3.25_BT1EBT_bias28j_LU2000.2026-01-30"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.2026-01-08"
CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.31j.2026-02-01"
#CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CRUJRA_BT3_LSP_MS3.25_BT1EBT.2026-01-24"
#CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.BT3_LSP.2026-01-08"


DATA_DIR = Path(f"{DATA_PATH}/{CASENAME}/lnd/hist/")

SAME_COLOR_SCALE = True  # Set to True to use same color scale for all variables
cmap_choose = 'YlGn'
vminv = 1; vmaxv=99

OUTPUT_FORMAT = "gif"  # "gif" or "mp4"

# Loop over multiple cases
for case in [6]:  # Specify which cases to run, e.g., [0, 1, 2], [5] for point dynamics, or [6] for szpf
    print(f"\n{'='*80}")
    print(f"PROCESSING CASE {case}")
    print(f"{'='*80}\n")
    
    if case == 0:
        # Variable names - can be a single string or a list of strings
        CASE_NAME = "hydraulic_mortality"
        VAR_NAMES = ["FATES_VEGC_LU"]  # or 
        SCALING_FACTORS = {var:1 for var in VAR_NAMES}
        YEAR_RANGE = (0, 30) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 12 # Process every Nth timestep (1=all, 2=every other, 3=every third, etc.)
        tint=3
        UNIT_LABELS = {var: "m2/m2" for var in VAR_NAMES}

    elif case == 1:
        CASE_NAME = "Cfluxes"
        VAR_NAMES = ["FATES_GPP", 
        "FATES_NPP",
        "FATES_LITTER_IN" ,
        "FATES_LITTER_OUT",
        "FATES_FIRE_CLOSS",
        "FATES_HET_RESP"]
        SCALING_FACTORS = {var: 3600*24*365 for var in VAR_NAMES}
        YEAR_RANGE = (25, 30) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep (1=all, 2=every other, 3=every third, etc.)
        tint=3
        UNIT_LABELS = {var: "KgC/m2/yr" for var in VAR_NAMES}

    elif case == 2:
        CASE_NAME = "mortality_sources"
        VAR_NAMES = ["FATES_MORTALITY_FIRE_CFLUX_PF", 
        "FATES_MORTALITY_HYDRAULIC_CFLUX_PF",
        "FATES_MORTALITY_CSTARV_CFLUX_PF",
        "FATES_MORTALITY_BACKGROUND_CFLUX_PF",
        "FATES_MORTALITY_SENESCENCE_CFLUX_PF",
        "FATES_HARVEST_WOODPROD_C_FLUX"]
        # Scaling factors for each variable (applied to convert units)
        # Default: 3600*24*365 converts from per-second to per-year
        SCALING_FACTORS = {var: 3600*24*365 for var in VAR_NAMES}
        YEAR_RANGE = (25, 30) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep (1=all, 2=every other, 3=every third, etc.)
        tint=5
        OUTPUT_FORMAT = "mp4"  # "gif" or "mp4"
        UNIT_LABELS = {var: "KgC/m2/yr" for var in VAR_NAMES}

    elif case == 3:
        CASE_NAME = "fire_variables"
        VAR_NAMES = ["FATES_FIRE_CLOSS",
         "FATES_FIRE_INTENSITY",
        "FATES_NESTEROV_INDEX",
        "FATES_IGNITIONS",
        "FATES_FUEL_AMOUNT",
        "FATES_BURNFRAC"]
        SCALING_FACTORS = {var: 1 for var in VAR_NAMES}
        YEAR_RANGE = (25, 30) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep 
        tint=3  # tie interval
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
        UNIT_LABELS = {var: "KgC/m2/yr" for var in VAR_NAMES}
        SAME_COLOR_SCALE = False  # Set to True to use same color scale for all variables
        vmaxv=97

    elif case == 4:
        CASE_NAME = "water_carbon"
        VAR_NAMES = ["FATES_GPP",
         "QVEGT",
        "BTRAN",
        "QFLX_EVAP_TOT"]
        SCALING_FACTORS = {var: 1 for var in VAR_NAMES}
        YEAR_RANGE = (0, 10) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep 
        tint=5  # tie interval
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
        UNIT_LABELS = {var: "KgC/m2/yr" for var in VAR_NAMES}
        SAME_COLOR_SCALE = False  # Set to True to use same color scale for all variables
        vmaxv=98

    elif case == 5:
        VAR_NAMES = ["FATES_VEGC_ABOVEGROUND_SZ"]
        #VAR_NAMES = ["FATES_NPLANT_SZ"]
        #VAR_NAMES = ["FATES_VEGC_BASAL_AREA"]\
        CASE_NAME = VAR_NAMES[0]+'points'
        BACKG_VAR="BTRAN"
        fatesarea="FATES_AREA_PLANTS"
        SCALING_FACTORS = {var: 1 for var in VAR_NAMES}
        UNIT_LABELS = {var: "cm2/ha" for var in VAR_NAMES}
        YEAR_RANGE = (0, 40) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep 
        tint=100  # time interval in ms
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
        PLOT_TYPE = "point_bars"  # Special plot type for bar charts at points
        TREE_STYLE = "cloud"  # "christmas" for Christmas tree shape, "cloud" for cloud-shaped crowns
        TREE_WIDTH = 1.5  # Control variable for tree width (default is 0.3, higher = wider trees)
        SHOW_TEMPERATURE = True  # Show temperature indicator bar and text
        SHOW_BTRAN = False  # Show BTRAN indicator box at bottom
        
        # Control variable for PFTs to plot - can be a list like ['ent', 'bet', 'bdt'] or single string 'ent'
        PFT_LIST = ['ent','bet','bdt']  # Options: 'ent', 'bet', 'bdt', 'ndt', 'sht', etc.
        
        # Convert to list if single string
        if isinstance(PFT_LIST, str):
            PFT_LIST = [PFT_LIST]


    elif case == 6:
        CASE_NAME = "vegc_szpf_points"
        CASE_NAME = "AGprod_szpf_points"
        CASE_NAME = "DDBH__szpf_points"

        VAR_NAMES = ["FATES_VEGC_SZPF"]
        VAR_NAMES = ["FATES_ABOVEGROUND_PROD_SZPF"]
        VAR_NAMES = ["FATES_DDBH_SZPF"]
        UNIT_LABELS = {var: "" for var in VAR_NAMES}
        SCALING_FACTORS = {var: 1 for var in VAR_NAMES}
        YEAR_RANGE = (36, 38) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep 
        tint=200  # time interval in ms
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
       
        PLOT_TYPE = "point_bars_szpf"  # Stacked bar charts for size x PFT multiplexed data
        # Reference variables to get dimensions
        SIZE_REF_VAR = "FATES_MORTALITY_BACKGROUND_SZ"  # Variable to get number of size classes
        PFT_REF_VAR = "FATES_VEGC_PF"   # Variable to get number of PFT classes
        
        # Control variable for PFTs to plot - can be a list like ['ent', 'bet', 'bdt'] or single string 'ent'
        PFT_LIST = ['ent', 'bet', 'bdt']  # Options: 'ent', 'bet', 'bdt', 'ndt', 'sht', etc.
        
        # Convert to list if single string
        if isinstance(PFT_LIST, str):
            PFT_LIST = [PFT_LIST]

    elif case == 7:
        VAR_NAMES = ["FATES_MORTALITY_BACKGROUND_SZ","FATES_MORTALITY_FIRE_SZ", "FATES_MORTALITY_CSTARV_SZ","FATES_MORTALITY_FREEZING_SZ","FATES_MORTALITY_SENESCENCE_SZ"]

        VAR_NAMES = ["FATES_NPP_PF","FATES_MORTALITY_FIRE_CFLUX_PF","FATES_MORTALITY_HYDRAULIC_CFLUX_PF","FATES_MORTALITY_CSTARV_CFLUX_PF","FATES_MORTALITY_BACKGROUND_CFLUX_PF","FATES_MORTALITY_SENESCENCE_CFLUX_PF"]
        CASE_NAME = 'M ortality_sources_pf'
        BACKG_VAR = "BTRAN"
        fatesarea = "FATES_AREA_PLANTS"
        SCALING_FACTORS = {var: 1 for var in VAR_NAMES}  # Convert from per-second to per-year
        SCALING_FACTORS[0]=0.01
        UNIT_LABELS = {var: "m2/yr" for var in VAR_NAMES}
        YEAR_RANGE = (58, 60)  # Year range to process
        FILE_INTERVAL = 1  # Process every Nth timestep
        tint = 150  # time interval in ms
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
        PLOT_TYPE = "point_scatter_sz"  # Scatter plot for size class data
        SHOW_TEMPERATURE = True  # Show temperature indicator bar and text
        SHOW_BTRAN = False  # Show BTRAN indicator box at bottom
        
        # Control variable for PFTs to plot
        PFT_LIST = ['ent', 'bet', 'bdt']  # Options: 'ent', 'bet', 'bdt', 'ndt', 'sht', etc.
        
        # Convert to list if single string
        if isinstance(PFT_LIST, str):
            PFT_LIST = [PFT_LIST]

    elif case == 8:
        # Case 8: Line graph version of case 7 - time on x-axis, each pft/sz as a line
        VAR_NAMES = ["FATES_MORTALITY_FIRE_CFLUX_PF","FATES_MORTALITY_HYDRAULIC_CFLUX_PF","FATES_MORTALITY_CSTARV_CFLUX_PF","FATES_MORTALITY_BACKGROUND_CFLUX_PF",
        "FATES_MORTALITY_SENESCENCE_CFLUX_PF"]
        #VAR_NAMES = ["FATES_MORTALITY_BACKGROUND_SZ","FATES_MORTALITY_FIRE_SZ", "FATES_MORTALITY_CSTARV_SZ","FATES_MORTALITY_FREEZING_SZ","FATES_MORTALITY_SENESCENCE_SZ"]
        #VAR_NAMES = ["FATES_VEGC_PF","FATES_LEAFC_PF"]
        #VAR_NAMES = ["FATES_GPP_PF","FATES_NPP_PF"]
        CASE_NAME = 'Mortality_sources_pf_timeseries'
        #CASE_NAME = 'Mortality_sources_sz_timeseries'
        #CASE_NAME = 'vegc_pf_timeseries'
        #CASE_NAME = 'carbon_fluxes_pf_timeseries'
        BACKG_VAR = "BTRAN"
        fatesarea = "FATES_AREA_PLANTS"
        
        # Auto-detect units based on variable names and set scaling factors
        UNIT_LABELS = {}
        SCALING_FACTORS = {}
        for var in VAR_NAMES:
            if 'CFLUX' in var or 'GPP' in var or 'NPP' in var:
                # Carbon flux variables - convert from gC/m2/s to KgC/m2/yr
                SCALING_FACTORS[var] = 3600*24*365
                UNIT_LABELS[var] = "KgC/m2/yr"
            elif any(mort_type in var for mort_type in ['MORTALITY_BACKGROUND_SZ', 'MORTALITY_FIRE_SZ', 'MORTALITY_CSTARV_SZ', 'MORTALITY_FREEZING_SZ', 'MORTALITY_SENESCENCE_SZ', 'MORTALITY_HYDRAULIC_SZ']):
                # Mortality rate variables (per year)
                SCALING_FACTORS[var] = 1
                UNIT_LABELS[var] = "fraction/yr"
            else:
                # Default
                SCALING_FACTORS[var] = 1
                UNIT_LABELS[var] = "unknown units"
        
        YEAR_RANGE = (58, 60)  # Year range to process
        FILE_INTERVAL = 1  # Process every Nth timestep
        tint = 150  # time interval in ms
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
        PLOT_TYPE = "point_line_sz"  # Line plot with time on x-axis
        SHOW_TEMPERATURE = True  # Show temperature indicator bar and text
        SHOW_BTRAN = False  # Show BTRAN indicator box at bottom
        
        # Control variable for PFTs to plot
        PFT_LIST = ['ent', 'bet', 'bdt']  # Options: 'ent', 'bet', 'bdt', 'ndt', 'sht', etc.
        
        # Convert to list if single string
        if isinstance(PFT_LIST, str):
            PFT_LIST = [PFT_LIST]


    # Convert to list if single string provided
    if isinstance(VAR_NAMES, str):
        VAR_NAMES = [VAR_NAMES]

    # Set default plot type if not specified
    if 'PLOT_TYPE' not in locals():
        PLOT_TYPE = "map"


    OUTPUT_PATH = "/datalake/NS9560K/www/diagnostics/noresm/rosief/"+CASENAME+"/gifs"
    file_ext = "gif" if OUTPUT_FORMAT == "gif" else "mp4"
    year_range_str = f"_y{YEAR_RANGE[0]}-{YEAR_RANGE[1]}" if YEAR_RANGE is not None else ""
    OUTPUT_FILE = OUTPUT_PATH+ "/fates_" + CASE_NAME + year_range_str + "_timeseries." + file_ext
    # Unit labels for each variable (applied when scaling is used)
    # If not specified, original units from the file will be used

    print(f"Reading data from: {DATA_DIR}")
    print(f"Variables to plot: {VAR_NAMES}")
    print(f"Same color scale for all: {SAME_COLOR_SCALE}")
    print(f"Output file: {OUTPUT_FILE}")
    if YEAR_RANGE is not None:
        print(f"Processing years {YEAR_RANGE[0]} to {YEAR_RANGE[1]}")
    print(f"File interval: Using every {FILE_INTERVAL} timestep(s)")

    # Get list of all NetCDF files
    print("\nFinding data files...")
    try:
        file_pattern = str(DATA_DIR / "*h0a*.nc")
        all_files = sorted(glob.glob(file_pattern))
    
        if not all_files:
            print(f"\nERROR: No files found matching pattern: {file_pattern}")
            exit(1)
    
        print(f"Found {len(all_files)} files total")
    
        # Build a list of (file, time_index, time_value) tuples for each timestep
        # Only scan files within the specified year range
        file_time_map = []
    
        # Determine timestep range based on year range
        # YEAR_RANGE determines temporal coverage, FILE_INTERVAL determines sampling rate
        start_timestep = None
        end_timestep = None
        if YEAR_RANGE is not None:
            start_timestep = YEAR_RANGE[0] * 12
            end_timestep = YEAR_RANGE[1] * 12
            n_raw_timesteps = end_timestep - start_timestep
            expected_output = n_raw_timesteps // FILE_INTERVAL
            print(f"Will process timesteps {start_timestep} to {end_timestep} (years {YEAR_RANGE[0]}-{YEAR_RANGE[1]}), using every {FILE_INTERVAL} timestep → ~{expected_output} output frames")
    
        print("Building timestep list from filenames...")
    
        # Build list directly from filenames - each file is one timestep
        timestep_counter = 0
        for i, file_path in enumerate(all_files):
            # Skip timesteps before the start of the range
            if start_timestep is not None and timestep_counter < start_timestep:
                timestep_counter += 1
                continue
        
            # Stop if we've reached the end of the range
            if end_timestep is not None and timestep_counter >= end_timestep:
                break
        
            # Only add timestep if it matches the interval
            if (timestep_counter - (start_timestep or 0)) % FILE_INTERVAL == 0:
                # Open file only to get the time value
                with xr.open_dataset(file_path, decode_times=False) as ds:
                    file_time_map.append((file_path, 0, ds.time.values[0]))
        
            timestep_counter += 1
        
            if (timestep_counter % 50 == 0 or timestep_counter == 1):
                print(f"\rProcessed {timestep_counter} timesteps, selected {len(file_time_map)}...", end='', flush=True)
    
        print(f"\n\nProcessed {timestep_counter} timesteps, selected {len(file_time_map)} for animation")
    
        n_times = len(file_time_map)
        print(f"Processing {n_times} timesteps")
    
        # Store metadata for each variable
        var_metadata = {}
        n_vars = len(VAR_NAMES)
    
        # Open first file to get metadata for all variables
        print("\nReading metadata from first file...")
        with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
            for VAR_NAME in VAR_NAMES:
                # Check if variable exists
                if VAR_NAME not in ds.variables:
                    print(f"\nERROR: Variable {VAR_NAME} not found in dataset")
                    print(f"Available variables: {list(ds.data_vars)}")
                    exit(1)
            
                print(f"\n{VAR_NAME}:")
                gpp_var = ds[VAR_NAME]
                print(f"  dims: {gpp_var.dims}")
                print(f"  shape (single timestep): {gpp_var.isel(time=0).shape}")
            
                # Get spatial dimensions (only need to do this once)
                if 'lat' not in var_metadata and 'lon' not in var_metadata:
                    if 'lat' in gpp_var.dims and 'lon' in gpp_var.dims:
                        use_cartopy = True
                        var_metadata['lat'] = ds.lat.values
                        var_metadata['lon'] = ds.lon.values
                    elif 'ncol' in gpp_var.dims or 'lndgrid' in gpp_var.dims:
                        use_cartopy = False
                        grid_dim = 'ncol' if 'ncol' in gpp_var.dims else 'lndgrid'
                        print(f"  Unstructured grid detected ({grid_dim} dimension)")
                        if 'lat' in ds.variables and 'lon' in ds.variables:
                            var_metadata['lat'] = ds.lat.values
                            var_metadata['lon'] = ds.lon.values
                        else:
                            print(f"ERROR: lat/lon coordinates not found in dataset")
                            exit(1)
                    else:
                        print(f"ERROR: Cannot determine spatial dimensions from {gpp_var.dims}")
                        exit(1)
                    var_metadata['use_cartopy'] = use_cartopy
            
                # Determine spatial dimension name
                if 'ncol' in gpp_var.dims:
                    spatial_dim = 'ncol'
                elif 'lndgrid' in gpp_var.dims:
                    spatial_dim = 'lndgrid'
                elif 'lat' in gpp_var.dims:
                    spatial_dim = 'lat'
                else:
                    spatial_dim = None
            
                # Check if variable has extra dimensions
                n_dims = len(gpp_var.dims)
                has_extra_dim = n_dims > 2
                extra_dims = []
                split_extra_dim = False
                if has_extra_dim:
                    extra_dims = [d for d in gpp_var.dims if d != 'time' and d != spatial_dim]
                    # If only one variable, create subplots for each member of extra dimension
                    if len(VAR_NAMES) == 1 and len(extra_dims) == 1:
                        split_extra_dim = True
                        print(f"  Will create subplots for each member of dimension: {extra_dims[0]}")
                    else:
                        print(f"  Will sum over extra dimensions: {extra_dims}")
            
                # Get first frame(s)
                first_frame_data = gpp_var.isel(time=0)
                if has_extra_dim and not split_extra_dim:
                    first_frame = first_frame_data.sum(dim=extra_dims).values
                else:
                    first_frame = first_frame_data.values
            
                # Apply scaling factor
                scaling_factor = SCALING_FACTORS.get(VAR_NAME, 1.0)
                print(f"  Applying scaling factor {scaling_factor} to {VAR_NAME}")
                first_frame = first_frame * scaling_factor
            
                # Get units (use custom label if provided, otherwise original units)
                units = UNIT_LABELS.get(VAR_NAME, gpp_var.attrs.get("units", "units not specified"))
                print(f"  Units for {VAR_NAME}: {units}")
            
                # Store metadata
                var_metadata[VAR_NAME] = {
                    'scaling_factor': scaling_factor,
                    'units': units,
                    'has_extra_dim': has_extra_dim,
                    'extra_dims': extra_dims,
                    'split_extra_dim': split_extra_dim,
                    'first_frame': first_frame,
                    'dims': gpp_var.dims
                }
            
                # If splitting extra dimension, store dimension size
                if split_extra_dim:
                    extra_dim_name = extra_dims[0]
                    var_metadata[VAR_NAME]['extra_dim_size'] = len(ds[extra_dim_name])
                    print(f"  {extra_dim_name} has {len(ds[extra_dim_name])} members")
    
        # Compute color scales from last timestep
        print("\nComputing color scales from last timestep...")
        last_file_path, last_time_idx, last_time_val = file_time_map[-1]
    
        if SAME_COLOR_SCALE:
            # Compute global min/max across all variables
            all_values = []
            with xr.open_dataset(last_file_path, decode_times=False) as ds_last:
                for VAR_NAME in VAR_NAMES:
                    last_frame_data = ds_last[VAR_NAME].isel(time=last_time_idx)
                    if var_metadata[VAR_NAME]['split_extra_dim']:
                        # Include all members of extra dimension
                        last_frame = last_frame_data.values
                    elif var_metadata[VAR_NAME]['has_extra_dim']:
                        last_frame = last_frame_data.sum(dim=var_metadata[VAR_NAME]['extra_dims']).values
                    else:
                        last_frame = last_frame_data.values
                    # Apply scaling factor
                    last_frame = last_frame * var_metadata[VAR_NAME]['scaling_factor']
                    all_values.append(last_frame.flatten())
        
            all_values = np.concatenate(all_values)
            global_vmin = float(np.nanpercentile(all_values, vminv))
            global_vmax = float(np.nanpercentile(all_values, vmaxv))
            print(f"  Global range: {global_vmin:.4f} to {global_vmax:.4f}")
        
            # Set same limits for all variables/subplots
            for VAR_NAME in VAR_NAMES:
                var_metadata[VAR_NAME]['vmin'] = global_vmin
                var_metadata[VAR_NAME]['vmax'] = global_vmax
        else:
            # Compute per-variable min/max
            with xr.open_dataset(last_file_path, decode_times=False) as ds_last:
                for VAR_NAME in VAR_NAMES:
                    last_frame_data = ds_last[VAR_NAME].isel(time=last_time_idx)
                    if var_metadata[VAR_NAME]['split_extra_dim']:
                        # Include all members when computing color scale
                        last_frame = last_frame_data.values
                    elif var_metadata[VAR_NAME]['has_extra_dim']:
                        last_frame = last_frame_data.sum(dim=var_metadata[VAR_NAME]['extra_dims']).values
                    else:
                        last_frame = last_frame_data.values
                    # Apply scaling factor
                    last_frame = last_frame * var_metadata[VAR_NAME]['scaling_factor']
                    # Combine first and last frame for better color scale
                    all_vals = np.concatenate([var_metadata[VAR_NAME]['first_frame'].flatten(), last_frame.flatten()])
                    vmin = float(np.nanpercentile(all_vals, 0))
                    vmax = float(np.nanpercentile(all_vals, 100))
                    var_metadata[VAR_NAME]['vmin'] = vmin
                    var_metadata[VAR_NAME]['vmax'] = vmax
                    print(f"  {VAR_NAME}: {vmin:.4f} to {vmax:.4f}")
    
        lat = var_metadata['lat']
        lon = var_metadata['lon']
        use_cartopy = var_metadata['use_cartopy']
        
        print(f"\nSpatial coordinate info:")
        print(f"  lat shape: {lat.shape}, min/max: {np.nanmin(lat):.2f}/{np.nanmax(lat):.2f}")
        print(f"  lon shape: {lon.shape}, min/max: {np.nanmin(lon):.2f}/{np.nanmax(lon):.2f}")
    
        # Check if this is a point-based plot
        if 'PLOT_TYPE' in locals() and PLOT_TYPE == "point_bars":
            print("\n" + "="*60)
            print("CREATING POINT-BASED BAR CHART ANIMATION")
            print("="*60)
            
            def plot_pft_animation(pft_name, pft_list, var_names, var_metadata, file_time_map, 
                                  scaling_factors, unit_labels, tree_style, tree_width, backg_var, 
                                  fatesarea, output_file, output_format, tint, start_timestep, 
                                  file_interval, show_temperature=True, show_btran=False):
                """
                Create animation for a single PFT.
                
                Args:
                    pft_name: Name of the PFT (e.g., 'bet', 'bdt', 'ent')
                    pft_list: List of all PFTs being processed
                    var_names: List of variable names to plot
                    var_metadata: Metadata dictionary containing lat/lon and other info
                    file_time_map: List of (file_path, time_idx, time_val) tuples
                    scaling_factors: Dictionary of scaling factors for each variable
                    unit_labels: Dictionary of unit labels for each variable
                    tree_style: Tree visualization style ('christmas' or 'cloud')
                    tree_width: Width scaling factor for trees
                    backg_var: Background variable name (e.g., 'BTRAN')
                    fatesarea: FATES area variable name
                    output_file: Output file path template
                    output_format: Output format ('gif' or 'mp4')
                    tint: Time interval in milliseconds
                    start_timestep: Starting timestep
                    file_interval: File interval for processing
                """
                pft_upper = pft_name.upper()
                print(f"\n{'='*60}")
                print(f"Processing PFT: {pft_upper}")
                print(f"{'='*60}\n")
                
                # Read locations from PFT-specific file using helper function
                print(f"Reading locations from PFT file for {pft_upper}...")
                LOCATIONS = read_location_file(pft_name, max_sites=8)
                
                if not LOCATIONS:
                    print(f"Skipping PFT: {pft_upper}")
                    return
                
                print(f"Loaded {len(LOCATIONS)} locations for {pft_upper}")
            
                # Find nearest grid points to requested locations
                location_indices = []
                lats_array = var_metadata['lat']
                lons_array = var_metadata['lon']
                for target_lat, target_lon, name in LOCATIONS:
                    idx, actual_lat, actual_lon = find_nearest_point(target_lat, target_lon, lats_array, lons_array)
                    location_indices.append((idx, actual_lat, actual_lon, name))
                    print(f"Location '{name}': target=({target_lat}, {target_lon}), actual=({actual_lat:.2f}, {actual_lon:.2f}), idx={idx}")
                
                VAR_NAME = var_names[0]  # Assume single variable for point plots
                
                # Extract size dimension information
                with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                    var_data = ds[VAR_NAME]
                    
                    print(f"\nVariable '{VAR_NAME}' dimensions: {var_data.dims}")
                    print(f"Available dimensions in dataset: {list(ds.dims.keys())}")
                    
                    # Get units from netcdf file if not already specified or if default
                    if VAR_NAME not in unit_labels or unit_labels[VAR_NAME] == "cm2/ha":
                        if hasattr(var_data, 'units'):
                            unit_labels[VAR_NAME] = var_data.units
                            print(f"Using units from netcdf: {var_data.units}")
                        else:
                            unit_labels[VAR_NAME] = "units"
                            print("No units attribute found in netcdf file")
                    
                    # Find the size or PFT dimension - look for common patterns
                    size_dim = None
                    for dim in var_data.dims:
                        dim_lower = dim.lower()
                        if any(pattern in dim_lower for pattern in ['size', 'sz', 'fates_levscls', 'levscls']):
                            size_dim = dim
                            break
                    
                    # If no size dimension found, check for PFT dimension
                    if size_dim is None:
                        for dim in var_data.dims:
                            dim_lower = dim.lower()
                            if any(pattern in dim_lower for pattern in ['pft', 'fates_levpft', 'levpft']):
                                size_dim = dim
                                print(f"Note: Using PFT dimension '{size_dim}' instead of size dimension")
                                break
                    
                    if size_dim is None:
                        print(f"ERROR: Could not find size or PFT dimension in variable {VAR_NAME}")
                        print(f"Variable dimensions: {var_data.dims}")
                        print(f"Looking for dimensions containing: 'size', 'sz', 'fates_levscls', 'levscls', 'pft', or 'levpft'")
                        return
                    
                    n_sizes = len(ds[size_dim])
                    print(f"Found dimension '{size_dim}' with {n_sizes} bins")
                    
                    # Check if this is a PFT dimension
                    is_pft_dim = 'pft' in size_dim.lower()
                    
                    # Get labels based on dimension type
                    if is_pft_dim:
                        # For PFT dimensions, create simple PFT labels
                        pft_names = ['BET Tropical', 'BET Temperate', 'BDT Tropical', 'BDT Temperate', 
                                   'NET Temperate', 'NDT Boreal', 'NET Boreal', 'BES Temperate', 
                                   'BDS Temperate', 'BDS Boreal', 'C3 Arctic Grass', 'C3 Grass', 'C4 Grass']
                        # Use actual PFT names if available, otherwise use generic labels
                        if n_sizes <= len(pft_names):
                            size_labels = [f"PFT{i+1}" for i in range(n_sizes)]  # Simple labels
                        else:
                            size_labels = [f"PFT{i+1}" for i in range(n_sizes)]
                    elif f"{size_dim}_bounds" in ds.variables or f"fates_{size_dim}_bounds" in ds.variables:
                        size_bounds_var = f"{size_dim}_bounds" if f"{size_dim}_bounds" in ds.variables else f"fates_{size_dim}_bounds"
                        size_bounds = ds[size_bounds_var].values
                        size_labels = [f"{size_bounds[i,0]:.1f}-{size_bounds[i,1]:.1f}" for i in range(n_sizes)]
                    else:
                        # Use the dimension values directly (e.g., diameter in cm)
                        size_values = ds[size_dim].values
                        size_labels = [f"{val:.0f}" for val in size_values]
                
                # Setup figure with subplots for each location
                n_locations = len(location_indices)
                n_cols = 4
                n_rows = 2
                
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 8))
                
                # Set the entire figure background with a smooth gradient
                # Create a custom colormap with more colors for smoother, constant vertical tone change
                from matplotlib.colors import LinearSegmentedColormap
                # Green gradient for main background
                funky_colors = ['#e8f5e9', '#c8e6c9', '#a5d6a7', '#81c784', '#66bb6a', '#4caf50', '#43a047', '#388e3c', '#2e7d32', '#1b5e20']  # Smooth light to dark green
                funky_cmap = LinearSegmentedColormap.from_list('funky_green', funky_colors)
                fig.patch.set_facecolor('#c8e6c9')  # Base green color
                
                if n_locations == 1:
                    axs = [axs]
                else:
                    axs = axs.flatten()
                
                # Add overall title for the PFT with variable name
                display_name = var_names[0].replace('FATES_', '')
                fig.suptitle(f'{pft_upper} Points - {display_name}', fontsize=20, fontweight='bold', y=0.98)
                
                # Initialize bar charts
                bar_containers = []
                tree_collections = []  # Store tree patch collections for each subplot
                x_pos = np.arange(n_sizes)
                
                def create_tree_shape(x_center, height, base_width=0.3, style="christmas", pft_type=None):
                    """Create a tree-shaped polygon that scales with height
                    
                    Args:
                        x_center: X position of the tree center
                        height: Height value where the TOP of the tree should be plotted
                        base_width: Base width scaling factor
                        style: 'christmas' for Christmas tree or 'cloud' for cloud-shaped crown
                        pft_type: PFT name for PFT-specific customization (e.g., 'bet', 'bdt')
                    """
                    if height <= 0:
                        return None  # No visible tree for zero values
                    
                    # Override style for specific PFTs
                    if pft_type == "ent":
                        style = "christmas"  # ENTs always use Christmas tree shape
                    
                    # Scale tree width based on height
                    width_scale = 0.3 + (height / 100.0) * 0.5  # Gradually increase width
                    width = base_width * width_scale
                    
                    # Tree trunk (bottom 15%) and crown proportions
                    trunk_proportion = 0.15
                    
                    if style == "cloud":
                        # PFT-specific parameters for cloud-shaped canopy
                        if pft_type == "bet":
                            # Bet PFT: wider and less deep canopy with flat top
                            crown_proportion = 0.65  # Shallower canopy (was 0.85)
                            canopy_width_multiplier = 1.1  # Even wider canopy (was 0.85)
                            trunk_proportion = 0.35  # Taller trunk to compensate
                        else:
                            # Default tropical tree parameters
                            crown_proportion = 0.85
                            canopy_width_multiplier = 0.6
                        
                        # Calculate absolute heights - tree top is at 'height', base is below
                        tree_total_height = height
                        trunk_height = tree_total_height * trunk_proportion
                        crown_height = tree_total_height * crown_proportion
                        trunk_width = width * 0.15
                        
                        # Base of tree starts at 0, top ends at specified height
                        tree_base = 0
                        crown_base = trunk_height
                        crown_top = height
                        
                        # Create a rounded, umbrella-like canopy using many points
                        n_points = 40  # More points for smoother curve
                        
                        vertices = [
                            # Trunk (starts at base, goes up to crown)
                            [x_center - trunk_width/2, tree_base],
                            [x_center - trunk_width/2, crown_base],
                        ]
                        
                        # Create smooth rounded crown using semi-ellipse with bumps
                        for i in range(n_points + 1):
                            # Angle from 0 to pi (left to right across canopy)
                            angle = np.pi * i / n_points
                            
                            # Base ellipse shape for tropical canopy (wider than tall)
                            x_offset = width * np.cos(angle) * canopy_width_multiplier
                            
                            # Height follows an elliptical arc with slight asymmetry
                            if pft_type == "bet":
                                # Oak-like shape: rounded with multiple lobes
                                # Base rounded shape
                                base_height = crown_base + crown_height * np.sin(angle)
                                # Add multiple large lobes for oak-like appearance (5-7 main lobes)
                                lobes = crown_height * 0.18 * np.sin(angle * 6)
                                # Add smaller variations for natural irregularity
                                small_bumps = crown_height * 0.06 * np.sin(angle * 13)
                                y_pos = base_height + lobes + small_bumps
                            else:
                                base_height = crown_base + crown_height * (0.3 + 0.7 * np.sin(angle))
                                # Add subtle bumps for natural look
                                bump = crown_height * 0.08 * np.sin(angle * 5)
                                y_pos = base_height + bump
                            
                            vertices.append([x_center + x_offset, y_pos])
                        
                        # Complete the shape back to trunk
                        vertices.append([x_center + trunk_width/2, crown_base])
                        vertices.append([x_center + trunk_width/2, tree_base])
                        
                    else:  # Christmas tree style (default)
                        # Christmas tree crown proportions
                        crown_proportion = 0.85
                        trunk_proportion = 0.15
                        
                        # Calculate absolute heights - tree top is at 'height', base is below
                        tree_total_height = height
                        trunk_height = tree_total_height * trunk_proportion
                        crown_height = tree_total_height * crown_proportion
                        trunk_width = width * 0.15
                        
                        # Base of tree starts at 0, top ends at specified height
                        tree_base = 0
                        
                        n_tiers = 4  # Number of branch tiers
                        
                        # Define tree vertices (trunk + layered Christmas tree crown)
                        vertices = [
                            # Trunk (starts at base, goes up)
                            [x_center - trunk_width/2, tree_base],
                            [x_center - trunk_width/2, trunk_height],
                        ]
                        
                        # Create layered tiers for Christmas tree effect
                        for i in range(n_tiers):
                            # Each tier gets progressively smaller toward the top
                            tier_bottom = trunk_height + (i * crown_height / n_tiers)
                            tier_top = trunk_height + ((i + 1) * crown_height / n_tiers)
                            tier_mid = (tier_bottom + tier_top) / 2
                            
                            # Width decreases as we go up
                            bottom_width_ratio = 1.0 - (i * 0.7 / n_tiers)
                            top_width_ratio = 1.0 - ((i + 1) * 0.7 / n_tiers)
                            mid_width_ratio = (bottom_width_ratio + top_width_ratio) / 2
                            
                            # Left side of tier - going up
                            vertices.append([x_center - width/2 * bottom_width_ratio, tier_bottom])
                            vertices.append([x_center - width/2 * mid_width_ratio * 0.7, tier_mid])
                        
                        # Top point of tree
                        vertices.append([x_center, height])
                        
                        # Right side - coming back down
                        for i in range(n_tiers - 1, -1, -1):
                            tier_bottom = trunk_height + (i * crown_height / n_tiers)
                            tier_top = trunk_height + ((i + 1) * crown_height / n_tiers)
                            tier_mid = (tier_bottom + tier_top) / 2
                            
                            bottom_width_ratio = 1.0 - (i * 0.7 / n_tiers)
                            top_width_ratio = 1.0 - ((i + 1) * 0.7 / n_tiers)
                            mid_width_ratio = (bottom_width_ratio + top_width_ratio) / 2
                            
                            vertices.append([x_center + width/2 * mid_width_ratio * 0.7, tier_mid])
                            vertices.append([x_center + width/2 * bottom_width_ratio, tier_bottom])
                        
                        # Back to trunk
                        vertices.append([x_center + trunk_width/2, trunk_height])
                        vertices.append([x_center + trunk_width/2, tree_base])
                    
                    return Polygon(vertices, closed=True)
                
                for loc_idx, (grid_idx, actual_lat, actual_lon, name) in enumerate(location_indices):
                    ax = axs[loc_idx]
                    
                    # Set initial x-axis limits based on data
                    ax.set_xlim(-0.5, n_sizes - 0.5)
                    
                    # Add funky shaded green background with vertical gradient (top to bottom)
                    # Create a gradient array that goes from light (top) to dark (bottom)
                    gradient_array = np.linspace(0, 1, 512).reshape(512, 1)  # More steps for smoother gradient
                    gradient = ax.imshow(gradient_array, extent=[-0.5, n_sizes - 0.5, 0, 100], 
                                         aspect='auto', cmap=funky_cmap, alpha=0.6, zorder=0,
                                         origin='upper', interpolation='bilinear')  # Smooth interpolation
                    
                    # Get first frame data for this location
                    with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                        var_data = ds[VAR_NAME].isel(time=0)
                        
                        # Determine spatial dimension
                        spatial_dim = None
                        if 'ncol' in var_data.dims:
                            spatial_dim = 'ncol'
                        elif 'lndgrid' in var_data.dims:
                            spatial_dim = 'lndgrid'
                        
                        if spatial_dim:
                            # Unstructured grid - select by spatial index
                            data = var_data.isel({spatial_dim: grid_idx}).values
                        else:
                            # Regular grid - this shouldn't happen for FATES data
                            print(f"Warning: Could not determine spatial dimension for location {name}")
                            data = np.zeros(n_sizes)
                        
                        data = data * scaling_factors[VAR_NAME]
                    
                    # Create tree-shaped patches for each bar
                    trees = []
                    for i, height in enumerate(data):
                        tree = create_tree_shape(x_pos[i], height, base_width=tree_width, style=tree_style, pft_type=pft_name)
                        if tree is not None:
                            trees.append(tree)
                    
                    # Create collection of tree patches (empty list is ok)
                    tree_collection = PatchCollection(trees if trees else [Polygon([[0,0]])], facecolors='forestgreen', 
                                                     edgecolors='darkgreen', linewidths=1.5, zorder=2)
                    ax.add_collection(tree_collection)
                    tree_collections.append((tree_collection, trees))
                    bar_containers.append(trees)  # Store trees for update
                    
                    ax.set_xlabel('Size Class: cm', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'{unit_labels[VAR_NAME]}', fontsize=12, fontweight='bold')
                    display_name = VAR_NAME.replace('FATES_', '')
                    ax.set_title(f'{display_name} - Year 0, Month 01', 
                               fontsize=14, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(size_labels, rotation=45, ha='right')
                    ax.grid(axis='y', alpha=0.3, zorder=1)
                    
                    # Add location info text to top right of this subplot
                    country_name = get_country(actual_lat, actual_lon)
                    us_state = get_us_state(actual_lat, actual_lon)
                    if us_state:
                        location_text = f"{us_state}, USA - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
                    else:
                        location_text = f"{country_name} - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
                    ax.text(0.98, 0.98, location_text, transform=ax.transAxes, 
                           fontsize=9, fontweight='bold', ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=10)
                    
                    # Set y-axis limit based on max across all timesteps and locations
                    ax.set_ylim(0, None)  # Will update after scanning data
                
                # Hide unused subplots
                for idx in range(n_locations, len(axs)):
                    axs[idx].axis('off')
                
                plt.tight_layout()
                
                # Determine y-axis limits by scanning all data
                print("\nDetermining y-axis limits...")
                max_value = 0
                n_times = len(file_time_map)
                # Sample first and last files to determine data range
                sample_files = []
                if n_times > 0:
                    sample_files.append(file_time_map[0])  # First file
                if n_times > 1:
                    sample_files.append(file_time_map[-1])  # Last file
                
                for file_path, time_idx, _ in sample_files:
                    with xr.open_dataset(file_path, decode_times=False) as ds:
                        var_data = ds[VAR_NAME].isel(time=time_idx)
                        # Determine spatial dimension
                        spatial_dim = None
                        if 'ncol' in var_data.dims:
                            spatial_dim = 'ncol'
                        elif 'lndgrid' in var_data.dims:
                            spatial_dim = 'lndgrid'
                        
                        for grid_idx, _, _, _ in location_indices:
                            if spatial_dim:
                                data = var_data.isel({spatial_dim: grid_idx}).values
                            else:
                                data = var_data.values
                            data = data * scaling_factors[VAR_NAME]
                            max_value = max(max_value, np.nanmax(data))
                
                # Add BTRAN boxes at the bottom of each subplot (in negative Y space)
                btran_box_height = max_value * 0.1 if show_btran else 0  # Height of BTRAN indicator box
                btran_boxes = []  # Store references to BTRAN boxes for animation updates
                temp_bars = []  # Store references to temperature bars for animation updates
                temp_bar_texts = []  # Store references to temperature text labels
                yax_scaler=1.2
                for loc_idx, ax in enumerate(axs[:n_locations]):
                    ax.set_ylim(-btran_box_height, max_value * yax_scaler)
                    # Update background gradient extent to cover full plot area
                    if len(ax.images) > 0:
                        # Create new gradient array properly sized for the actual y-range
                        gradient_array = np.linspace(0, 1, 512).reshape(512, 1)  # More steps for smoother gradient
                        ax.images[0].set_data(gradient_array)
                        ax.images[0].set_extent([-0.5, n_sizes - 0.5, -btran_box_height, max_value * yax_scaler])
                    
                    # Add BTRAN indicator box at the bottom
                    if show_btran:
                        # Get initial BTRAN value for this location
                        grid_idx, _, _, _ = location_indices[loc_idx]
                        with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                            if backg_var in ds.variables:
                                btran_data = ds[backg_var].isel(time=0)
                                # Determine spatial dimension
                                if 'ncol' in btran_data.dims:
                                    btran_val = btran_data.isel(ncol=grid_idx).values
                                elif 'lndgrid' in btran_data.dims:
                                    btran_val = btran_data.isel(lndgrid=grid_idx).values
                                else:
                                    btran_val = 0.5  # Default if spatial dim not found
                                
                                # Divide by FATES_AREA_PLANTS
                                if fatesarea in ds.variables:
                                    fatesarea_data = ds[fatesarea].isel(time=0)
                                    if 'ncol' in fatesarea_data.dims:
                                        fatesarea_val = fatesarea_data.isel(ncol=grid_idx).values
                                    elif 'lndgrid' in fatesarea_data.dims:
                                        fatesarea_val = fatesarea_data.isel(lndgrid=grid_idx).values
                                    else:
                                        fatesarea_val = 1.0
                                    if fatesarea_val > 0:
                                        btran_val = btran_val / fatesarea_val
                            else:
                                print(f"Warning: {backg_var} not found in dataset")
                                btran_val = 0.5  # Default value
                        
                        # Create blue box with intensity proportional to BTRAN (dark blue at 1, white at 0)
                        # Use a blue color that scales from white (0) to dark blue (1)
                        blue_intensity = btran_val  # 0 to 1
                        box_color = plt.cm.Blues(blue_intensity * 0.9 + 0.1)  # Map to blues colormap (0.1-1.0 range)
                        
                        btran_box = ax.fill_between(
                            [-0.5, n_sizes - 0.5], 
                            [-btran_box_height, -btran_box_height], 
                            [0, 0],
                            color=box_color, 
                            alpha=0.8, 
                            zorder=5,
                            linewidth=2,
                            edgecolor='navy'
                        )
                        btran_boxes.append(btran_box)
                        
                        # Add BTRAN label
                        ax.text(n_sizes/2 - 0.5, -btran_box_height/2, f'{backg_var}: {btran_val:.2f}', 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color='darkblue', zorder=6)
                    else:
                        btran_boxes.append(None)
                    
                    # Add temperature indicator as a red bar on the right side
                    if show_temperature:
                        # Get initial TSA value for this location
                        temp_var = 'TSA'
                        with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                            if temp_var in ds.variables:
                                temp_data = ds[temp_var].isel(time=0)
                                # Determine spatial dimension
                                if 'ncol' in temp_data.dims:
                                    temp_val = temp_data.isel(ncol=grid_idx).values
                                elif 'lndgrid' in temp_data.dims:
                                    temp_val = temp_data.isel(lndgrid=grid_idx).values
                                else:
                                    temp_val = 273.1  # Default 0°C in Kelvin
                                
                                # Convert from Kelvin to Celsius if needed
                                temp_celsius = float(temp_val)
                                if temp_celsius > 100:  # Likely in Kelvin
                                    temp_celsius = temp_celsius - 273.15
                            else:
                                print(f"Warning: {temp_var} not found in dataset")
                                temp_celsius = 0.0  # Default value
                        
                        # Create red temperature bar on the right side
                        # Map temperature to bar height: -20°C (min) -> 25°C (max) = 0 to max_value * 1.1
                        temp_min, temp_max = -20, 25
                        temp_range = temp_max - temp_min
                        temp_height = ((temp_celsius - temp_min) / temp_range) * max_value * 1.1
                        temp_height = np.clip(temp_height, 0, max_value * 1.1)  # Clamp to plot range
                        
                        # Position bar on the right side of the plot
                        temp_bar_x = n_sizes - 0.2  # Right side position
                        temp_bar_width = 0.35  # Width of the bar
                        
                        # Create the red bar
                        temp_bar = ax.fill_between(
                            [temp_bar_x, temp_bar_x + temp_bar_width],
                            [0, 0],
                            [temp_height, temp_height],
                            color='red',
                            alpha=0.7,
                            zorder=15,
                            linewidth=2,
                            edgecolor='darkred'
                        )
                        temp_bars.append(temp_bar)
                        
                        # Add temperature label that follows bar but stays within bounds
                        # Position text at bar height, clamped to stay within 0.2 to 0.5 of subplot height
                        text_y_pos = np.clip(temp_height, max_value * 0.1, max_value * 0.4)
                        # Position text further to the left, over the larger size classes
                        text_x_pos = n_sizes - 2.0  # Further left, over the 100cm area
                        temp_text = ax.text(text_x_pos, text_y_pos,
                               f'{temp_celsius:.1f}°C',
                               ha='center', va='center', fontsize=9, fontweight='bold',
                               color='darkred', zorder=16,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='darkred'))
                        temp_bar_texts.append(temp_text)
                    else:
                        temp_bars.append(None)
                        temp_bar_texts.append(None)
                
                # Animation update function for bar charts
                current_file = None
                current_ds = None
                
                def update_bars(frame):
                    """Update bar charts for each location"""
                    nonlocal current_file, current_ds
                    
                    print(f"\rProcessing frame {frame+1}/{n_times}", end='', flush=True)
                    
                    file_path, time_idx, time_val = file_time_map[frame]
                    
                    if current_file != file_path:
                        if current_ds is not None:
                            current_ds.close()
                        current_ds = xr.open_dataset(file_path, decode_times=False)
                        current_file = file_path
                    
                    # Update each location's bar chart
                    for loc_idx, (grid_idx, actual_lat, actual_lon, name) in enumerate(location_indices):
                        ax = axs[loc_idx]
                        tree_collection, old_trees = tree_collections[loc_idx]
                        
                        # Update BTRAN box color
                        if show_btran and backg_var in current_ds.variables:
                            btran_data = current_ds[backg_var].isel(time=time_idx)
                            if 'ncol' in btran_data.dims:
                                btran_val = btran_data.isel(ncol=grid_idx).values
                            elif 'lndgrid' in btran_data.dims:
                                btran_val = btran_data.isel(lndgrid=grid_idx).values
                            else:
                                btran_val = 0.5
                            
                            # Divide by FATES_AREA_PLANTS
                            if fatesarea in current_ds.variables:
                                fatesarea_data = current_ds[fatesarea].isel(time=time_idx)
                                if 'ncol' in fatesarea_data.dims:
                                    fatesarea_val = fatesarea_data.isel(ncol=grid_idx).values
                                elif 'lndgrid' in fatesarea_data.dims:
                                    fatesarea_val = fatesarea_data.isel(lndgrid=grid_idx).values
                                else:
                                    fatesarea_val = 1.0
                                if fatesarea_val > 0:
                                    btran_val = btran_val / fatesarea_val
                        else:
                            btran_val = 0.5
                        
                        if show_btran:
                            # Update BTRAN box color (dark blue at 1, white at 0)
                            blue_intensity = float(btran_val)
                            box_color = plt.cm.Blues(blue_intensity * 0.9 + 0.1)
                            btran_boxes[loc_idx].set_color(box_color)
                            
                            # Update BTRAN text - find and update existing text
                            # Remove old BTRAN text and add new one
                            for text_obj in ax.texts:
                                if backg_var in text_obj.get_text():
                                    text_obj.remove()
                            ax.text(n_sizes/2 - 0.5, -btran_box_height/2, f'{backg_var}: {btran_val:.2f}', 
                                   ha='center', va='center', fontsize=10, fontweight='bold',
                                   color='darkblue', zorder=6)
                        
                        # Update temperature bar height
                        if show_temperature:
                            temp_var = 'TSA'
                        if temp_var in current_ds.variables:
                            temp_data = current_ds[temp_var].isel(time=time_idx)
                            if 'ncol' in temp_data.dims:
                                temp_val = temp_data.isel(ncol=grid_idx).values
                            elif 'lndgrid' in temp_data.dims:
                                temp_val = temp_data.isel(lndgrid=grid_idx).values
                            else:
                                temp_val = 273.1
                            
                            # Convert from Kelvin to Celsius if needed
                            temp_celsius = float(temp_val)
                            if temp_celsius > 100:  # Likely in Kelvin
                                temp_celsius = temp_celsius - 273.15
                        else:
                            temp_celsius = 0.0
                        
                        if show_temperature:
                            # Update temperature bar height based on temperature
                            temp_min, temp_max = -20, 25
                            temp_range = temp_max - temp_min
                            heightbox_scaler=0.4
                            temp_height = ((temp_celsius - temp_min) / temp_range) * max_value * yax_scaler * heightbox_scaler
                            
                            temp_height = np.clip(temp_height, 0, max_value * 1.1)
                            # Update bar by removing and recreating (easier than modifying fill_between)
                            temp_bars[loc_idx].remove()
                            temp_bar_x = n_sizes - 0.2
                            temp_bar_width = 0.35
                            temp_bar = ax.fill_between(
                                [temp_bar_x, temp_bar_x + temp_bar_width],
                                [0, 0],
                                [temp_height, temp_height],
                                color='red',
                                alpha=0.7,
                                zorder=15,
                                linewidth=2,
                                edgecolor='darkred'
                            )
                            temp_bars[loc_idx] = temp_bar
                            
                            # Update temperature text to follow bar height but stay within bounds
                            text_y_pos = np.clip(temp_height, max_value * 0.2, max_value * 0.5)
                            text_x_pos = n_sizes - 2.0
                            temp_bar_texts[loc_idx].set_position((text_x_pos, text_y_pos))
                            temp_bar_texts[loc_idx].set_text(f'{temp_celsius:.1f}°C')
                        
                        # Get data for this location and timestep
                        var_data = current_ds[VAR_NAME].isel(time=time_idx)
                        
                        # Determine spatial dimension
                        spatial_dim = None
                        if 'ncol' in var_data.dims:
                            spatial_dim = 'ncol'
                        elif 'lndgrid' in var_data.dims:
                            spatial_dim = 'lndgrid'
                        
                        if spatial_dim:
                            data = var_data.isel({spatial_dim: grid_idx}).values
                        else:
                            data = var_data.values
                        
                        data = data * scaling_factors[VAR_NAME]
                        
                        # Create new tree shapes with updated heights
                        new_trees = []
                        for i, height in enumerate(data):
                            tree = create_tree_shape(x_pos[i], height, base_width=tree_width, style=tree_style, pft_type=pft_name)
                            if tree is not None:
                                new_trees.append(tree)
                        
                        # Update the patch collection with new trees (empty list clears all)
                        if not new_trees:
                            new_trees = [Polygon([[0,0], [0,0], [0,0]], closed=True)]  # Invisible placeholder
                        tree_collection.set_paths(new_trees)
                        
                        # Update title with current time
                        actual_timestep = (start_timestep or 0) + (frame * file_interval)
                        year = actual_timestep // 12
                        month = actual_timestep % 12 + 1
                        display_name = VAR_NAME.replace('FATES_', '')
                        ax.set_title(f'Year {year}, Month {month:02d}',
                               fontsize=14, fontweight='bold')
                
                    return [tree_collections[i][0] for i in range(len(tree_collections))]
                
                # Create animation
                print(f"\nCreating bar chart animation with {n_times} frames...")
                anim = animation.FuncAnimation(
                    fig, update_bars, frames=n_times,
                    interval=tint,
                    blit=True,
                    repeat=True
                )
                
                # Create PFT-specific output file name
                pft_output_file = output_file.replace("_timeseries.", f"_{pft_name}_timeseries.")
                print(f"\n\nSaving {pft_upper} animation to {pft_output_file}...")
                
                output_dir = Path(pft_output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if output_format == "gif":
                    from matplotlib.animation import PillowWriter
                    writer = PillowWriter(fps=1000//tint)
                    anim.save(pft_output_file, writer=writer, dpi=100)
                else:
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=1000//tint, bitrate=5000)
                    anim.save(pft_output_file, writer=writer, dpi=100)
                
                print(f"Animation successfully saved to: {pft_output_file}")
                plt.close(fig)
            
            # Loop through each PFT to create separate animations
            for pft_name in PFT_LIST:
                plot_pft_animation(
                    pft_name=pft_name,
                    pft_list=PFT_LIST,
                    var_names=VAR_NAMES,
                    var_metadata=var_metadata,
                    file_time_map=file_time_map,
                    scaling_factors=SCALING_FACTORS,
                    unit_labels=UNIT_LABELS,
                    tree_style=TREE_STYLE,
                    tree_width=TREE_WIDTH,
                    backg_var=BACKG_VAR,
                    fatesarea=fatesarea,
                    output_file=OUTPUT_FILE,
                    output_format=OUTPUT_FORMAT,
                    tint=tint,
                    start_timestep=start_timestep,
                    file_interval=FILE_INTERVAL,
                    show_temperature=SHOW_TEMPERATURE,
                    show_btran=SHOW_BTRAN
                )
        
        elif 'PLOT_TYPE' in locals() and PLOT_TYPE == "point_bars_szpf":
            print("\n" + "="*60)
            print("CREATING POINT-BASED STACKED BAR CHART (SIZE x PFT)")
            print("="*60)
            
            # Process all PFTs in the list
            for pft_name in PFT_LIST:
                pft_upper = pft_name.upper()
                print(f"\n{'='*60}")
                print(f"Processing PFT: {pft_upper}")
                print(f"{'='*60}\n")
                
                # Read locations from PFT-specific file
                print(f"Reading locations from PFT file for {pft_upper}...")
                LOCATIONS = read_location_file(pft_name, max_sites=8)
                
                if not LOCATIONS:
                    print(f"Skipping PFT: {pft_upper}")
                    continue
                
                print(f"Loaded {len(LOCATIONS)} locations for {pft_upper}")
                
                # Find nearest grid points to requested locations (using module-level helper)
                location_indices = []
                for target_lat, target_lon, name in LOCATIONS:
                    idx, actual_lat, actual_lon = find_nearest_point(target_lat, target_lon, lat, lon)
                    location_indices.append((idx, actual_lat, actual_lon, name))
                    print(f"Location '{name}': target=({target_lat}, {target_lon}), actual=({actual_lat:.2f}, {actual_lon:.2f}), idx={idx}")
                
                # Determine dimensions from reference variables
                with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                    VAR_NAME = VAR_NAMES[0]  # Main multiplexed variable
                    var_data = ds[VAR_NAME]
                    
                    print(f"\nVariable '{VAR_NAME}' dimensions: {var_data.dims}")
                    
                    # Get size dimension from reference variable
                    if SIZE_REF_VAR in ds.variables:
                        size_var = ds[SIZE_REF_VAR]
                        for dim in size_var.dims:
                            dim_lower = dim.lower()
                            if any(pattern in dim_lower for pattern in ['size', 'sz', 'fates_levscls', 'levscls']):
                                size_dim = dim
                                n_sizes = len(ds[size_dim])
                                print(f"Found {n_sizes} size classes from '{SIZE_REF_VAR}' dimension '{size_dim}'")
                                break
                    else:
                        print(f"ERROR: Reference variable '{SIZE_REF_VAR}' not found")
                        exit(1)
                    
                    # Get PFT dimension from reference variable
                    if PFT_REF_VAR in ds.variables:
                        pft_var = ds[PFT_REF_VAR]
                        for dim in pft_var.dims:
                            dim_lower = dim.lower()
                            if 'pft' in dim_lower or 'fates_levpft' in dim_lower:
                                pft_dim = dim
                                n_pfts = len(ds[pft_dim])
                                print(f"Found {n_pfts} PFT classes from '{PFT_REF_VAR}' dimension '{pft_dim}'")
                                break
                    else:
                        print(f"ERROR: Reference variable '{PFT_REF_VAR}' not found")
                        exit(1)
                    
                    # Verify multiplexed dimension size
                    for dim in var_data.dims:
                        if dim not in ['time', 'ncol', 'lndgrid']:
                            multiplex_dim = dim
                            multiplex_size = len(ds[multiplex_dim])
                            print(f"Multiplexed dimension '{multiplex_dim}' has size {multiplex_size}")
                            
                            # Check both orderings
                            if multiplex_size == n_sizes * n_pfts:
                                print(f"Confirmed: {multiplex_size} = {n_sizes} sizes × {n_pfts} PFTs")
                            else:
                                print(f"WARNING: Expected {n_sizes * n_pfts} but got {multiplex_size}")
                            break
                    
                    # Get size bin labels
                    size_labels = [f"Size {i+1}" for i in range(n_sizes)]
                    for size_bounds_var in [f"{size_dim}_bounds", f"fates_{size_dim}_bounds"]:
                        if size_bounds_var in ds.variables:
                            size_bounds = ds[size_bounds_var].values
                            size_labels = [f"{size_bounds[i,0]:.1f}-{size_bounds[i,1]:.1f}" for i in range(n_sizes)]
                            break
                    
                    # Get PFT labels
                    pft_labels = [f"PFT {i+1}" for i in range(n_pfts)]
                    if 'pfts' in ds.variables or 'fates_pftname' in ds.variables:
                        pft_name_var = 'pfts' if 'pfts' in ds.variables else 'fates_pftname'
                        try:
                            pft_names = ds[pft_name_var].values
                            if hasattr(pft_names[0], 'decode'):
                                pft_labels = [name.decode('utf-8').strip() for name in pft_names]
                            else:
                                pft_labels = [str(name).strip() for name in pft_names]
                        except:
                            pass
                
                # Define colors for PFTs - use a colormap
                import matplotlib.cm as cm
                colors = cm.tab20(np.linspace(0, 1, n_pfts))
                
                # Setup figure with subplots for each location
                n_locations = len(location_indices)
                n_cols = min(n_locations, 2)
                n_rows = (n_locations + n_cols - 1) // n_cols
                
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 7*n_rows))
                if n_locations == 1:
                    axs = [axs]
                else:
                    axs = axs.flatten()
                
                # Initialize stacked bar charts
                bar_containers_by_location = []
                x_pos = np.arange(n_sizes)
                
                # Function to demultiplex data - try both orderings
                def demultiplex_data(data_1d, n_sizes, n_pfts, order='size_major'):
                    """Reshape multiplexed 1D data into 2D size x PFT array"""
                    if order == 'size_major':
                        # Each size class has all PFTs: [s0p0, s0p1, ..., s0pN, s1p0, s1p1, ...]
                        return data_1d.reshape(n_sizes, n_pfts)
                    else:
                        # Each PFT has all sizes: [p0s0, p0s1, ..., p0sN, p1s0, p1s1, ...]
                        return data_1d.reshape(n_pfts, n_sizes).T
                
                # Test both orderings with first location to determine which is correct
                with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                    var_data = ds[VAR_NAME].isel(time=0)
                    spatial_dim = 'ncol' if 'ncol' in var_data.dims else 'lndgrid'
                    test_data = var_data.isel({spatial_dim: location_indices[0][0]}).values * SCALING_FACTORS[VAR_NAME]
                    
                    # Data is PFT-major ordered: [p0s0, p0s1, ..., p0sN, p1s0, p1s1, ...]
                    multiplex_order = 'pft_major'
                    print(f"\nUsing '{multiplex_order}' ordering for demultiplexing")
                    print("(If plots look wrong, change multiplex_order to 'size_major')")
                
                for loc_idx, (grid_idx, actual_lat, actual_lon, name) in enumerate(location_indices):
                    ax = axs[loc_idx]
                    
                    # Get first frame data for this location
                    with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                        var_data = ds[VAR_NAME].isel(time=0)
                        spatial_dim = 'ncol' if 'ncol' in var_data.dims else 'lndgrid'
                        data_1d = var_data.isel({spatial_dim: grid_idx}).values * SCALING_FACTORS[VAR_NAME]
                        
                        # Demultiplex into size x PFT array
                        data_2d = demultiplex_data(data_1d, n_sizes, n_pfts, multiplex_order)
                    
                    # Create stacked bar chart
                    bar_containers = []
                    bottom = np.zeros(n_sizes)
                    for pft_idx in range(n_pfts):
                        bars = ax.bar(x_pos, data_2d[:, pft_idx], bottom=bottom,
                                     label=pft_labels[pft_idx], color=colors[pft_idx],
                                     edgecolor='black', linewidth=0.3)
                        bar_containers.append(bars)
                        bottom += data_2d[:, pft_idx]
                    
                    bar_containers_by_location.append(bar_containers)
                    
                    ax.set_xlabel('Size Class', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'{UNIT_LABELS[VAR_NAME]}', fontsize=12, fontweight='bold')
                    ax.set_title(f'{VAR_NAME} - {CASENAME}\n{name} ({actual_lat:.2f}°, {actual_lon:.2f}°) - Year 0, Month 01', 
                               fontsize=14, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(size_labels, rotation=45, ha='right')
                    ax.grid(axis='y', alpha=0.3)
                    ax.legend(loc='upper right', fontsize=8, ncol=2)
                
                # Hide unused subplots
                for idx in range(n_locations, len(axs)):
                    axs[idx].axis('off')
                
                plt.tight_layout()
                
                # Determine y-axis limits
                print("\nDetermining y-axis limits...")
                max_value = 0
                # Check last 10 frames instead of first 10
                for file_path, time_idx, _ in file_time_map[-min(10, len(file_time_map)):]:
                    with xr.open_dataset(file_path, decode_times=False) as ds:
                        var_data = ds[VAR_NAME].isel(time=time_idx)
                        spatial_dim = 'ncol' if 'ncol' in var_data.dims else 'lndgrid'
                        
                        for grid_idx, _, _, _ in location_indices:
                            data_1d = var_data.isel({spatial_dim: grid_idx}).values * SCALING_FACTORS[VAR_NAME]
                            data_2d = demultiplex_data(data_1d, n_sizes, n_pfts, multiplex_order)
                            total_per_size = data_2d.sum(axis=1)
                            max_value = max(max_value, np.nanmax(total_per_size))
                yax_scaler=1.2
                for ax in axs[:n_locations]:
                    ax.set_ylim(0, max_value * yax_scaler)
                    #ax.set_ylim(0, 0.9)  # Commented out - was overriding calculated limit
                print('max_value', max_value)
                # Animation update function
                current_file = None
                current_ds = None
                
                def update_stacked_bars(frame):
                    """Update stacked bar charts for each location"""
                    global current_file, current_ds
                    
                    print(f"\rProcessing frame {frame+1}/{n_times}", end='', flush=True)
                    
                    file_path, time_idx, time_val = file_time_map[frame]
                    
                    if current_file != file_path:
                        if current_ds is not None:
                            current_ds.close()
                        current_ds = xr.open_dataset(file_path, decode_times=False)
                        current_file = file_path
                    
                    # Update each location's stacked bar chart
                    for loc_idx, (grid_idx, actual_lat, actual_lon, name) in enumerate(location_indices):
                        ax = axs[loc_idx]
                        bar_containers = bar_containers_by_location[loc_idx]
                        
                        # Get data for this location and timestep
                        var_data = current_ds[VAR_NAME].isel(time=time_idx)
                        spatial_dim = 'ncol' if 'ncol' in var_data.dims else 'lndgrid'
                        data_1d = var_data.isel({spatial_dim: grid_idx}).values * SCALING_FACTORS[VAR_NAME]
                        data_2d = demultiplex_data(data_1d, n_sizes, n_pfts, multiplex_order)
                        
                        # Update bar heights for each PFT
                        bottom = np.zeros(n_sizes)
                        for pft_idx in range(n_pfts):
                            bars = bar_containers[pft_idx]
                            pft_data = data_2d[:, pft_idx]
                            for bar, height, bot in zip(bars, pft_data, bottom):
                                bar.set_height(height)
                                bar.set_y(bot)
                            bottom += pft_data
                        
                        # Update title with current time
                        actual_timestep = (start_timestep or 0) + (frame * FILE_INTERVAL)
                        year = actual_timestep // 12
                        month = actual_timestep % 12 + 1
                        ax.set_title(f'{VAR_NAME} - {CASENAME}\n{name} ({actual_lat:.2f}°, {actual_lon:.2f}°) - Year {year}, Month {month:02d}',
                                   fontsize=14, fontweight='bold')
                    
                    return [bar for loc_bars in bar_containers_by_location for bars in loc_bars for bar in bars]
                
                # Create animation
                print(f"\nCreating stacked bar chart animation with {n_times} frames...")
                anim = animation.FuncAnimation(
                    fig, update_stacked_bars, frames=n_times,
                    interval=tint,
                    blit=True,
                    repeat=True
                )
                
                # Save animation
                pft_output_file = OUTPUT_FILE.replace('.gif', f'_{pft_upper}.gif').replace('.mp4', f'_{pft_upper}.mp4')
                print(f"\nSaving {pft_upper} animation to {pft_output_file}...")
                
                output_dir = Path(pft_output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if OUTPUT_FORMAT == "mp4":
                    import shutil
                    if shutil.which('ffmpeg') is None:
                        print("WARNING: ffmpeg not found. Falling back to GIF format.")
                        pft_output_file = pft_output_file.replace('.mp4', '.gif')
                        writer = animation.PillowWriter(fps=1000/tint)
                        anim.save(pft_output_file, writer=writer, dpi=100)
                    else:
                        writer = animation.FFMpegWriter(fps=1000/tint, metadata=dict(artist='FATES Analysis'), bitrate=1800)
                        anim.save(pft_output_file, writer=writer, dpi=100)
                elif OUTPUT_FORMAT == "gif":
                    writer = animation.PillowWriter(fps=1000/tint)
                    anim.save(pft_output_file, writer=writer, dpi=100)
                
                print(f"Animation saved successfully!")
                plt.close(fig)
            
        elif 'PLOT_TYPE' in locals() and PLOT_TYPE == "point_scatter_sz":
            print("\n" + "="*60)
            print("CREATING POINT-BASED SCATTER PLOT (SIZE CLASS)")
            print("="*60)
            
            # Process all PFTs in the list
            for pft_name in PFT_LIST:
                pft_upper = pft_name.upper()
                print(f"\n{'='*60}")
                print(f"Processing PFT: {pft_upper}")
                print(f"{'='*60}\n")
                
                # Read locations from PFT-specific file
                print(f"Reading locations from PFT file for {pft_upper}...")
                LOCATIONS = read_location_file(pft_name, max_sites=8)
                
                if not LOCATIONS:
                    print(f"Skipping PFT: {pft_upper}")
                    continue
                
                print(f"Loaded {len(LOCATIONS)} locations for {pft_upper}")
                
                # Find nearest grid points to requested locations
                location_indices = []
                for target_lat, target_lon, name in LOCATIONS:
                    idx, actual_lat, actual_lon = find_nearest_point(target_lat, target_lon, lat, lon)
                    location_indices.append((idx, actual_lat, actual_lon, name))
                    print(f"Location '{name}': target=({target_lat}, {target_lon}), actual=({actual_lat:.2f}, {actual_lon:.2f}), idx={idx}")
                
                n_locations = len(location_indices)
                n_vars = len(VAR_NAMES)
                
                # Get size dimension info from first file
                with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                    first_var = VAR_NAMES[0]
                    var_data = ds[first_var]
                    
                    # Find size or PFT dimension
                    size_dim = None
                    for dim in var_data.dims:
                        if 'size' in dim.lower() or 'levscls' in dim.lower():
                            size_dim = dim
                            break
                    
                    # If no size dimension found, check for PFT dimension
                    if size_dim is None:
                        for dim in var_data.dims:
                            if 'pft' in dim.lower() or 'levpft' in dim.lower():
                                size_dim = dim
                                print(f"Note: Using PFT dimension '{size_dim}' instead of size dimension")
                                break
                    
                    if size_dim is None:
                        print(f"ERROR: Could not find size or PFT dimension in {first_var}")
                        print(f"Available dimensions: {var_data.dims}")
                        exit(1)
                    
                    n_sizes = len(ds[size_dim])
                    is_pft_dim = 'pft' in size_dim.lower()
                    dim_type = "PFT classes" if is_pft_dim else "size classes"
                    print(f"\nFound {n_sizes} {dim_type} in dimension '{size_dim}'")
                    
                    # Determine spatial dimension
                    if 'ncol' in var_data.dims:
                        spatial_dim = 'ncol'
                    elif 'lndgrid' in var_data.dims:
                        spatial_dim = 'lndgrid'
                    else:
                        print(f"ERROR: No spatial dimension found")
                        exit(1)
                    
                    print(f"Using spatial dimension: {spatial_dim}")
                
                # Create labels based on dimension type
                if is_pft_dim:
                    size_labels = [f"PFT{i+1}" for i in range(n_sizes)]
                else:
                    size_labels = [f"SC{i+1}" for i in range(n_sizes)]
                x_pos = np.arange(n_sizes)
                
                # Define colors and markers for each variable
                var_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                var_markers = ['o', 's', '^', 'D', 'v', 'p']
                
                # Setup figure - 4 columns by 2 rows grid layout (like case 5)
                n_cols = 4
                n_rows = 2
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
                axs = axs.flatten()
                
                # Initialize scatter plots
                scatter_artists = []
                for loc_idx in range(n_locations):
                    loc_scatters = []
                    grid_idx, actual_lat, actual_lon, name = location_indices[loc_idx]
                    ax = axs[loc_idx]
                    
                    # Plot all variables on the same axes with different colors/markers
                    for var_idx, VAR_NAME in enumerate(VAR_NAMES):
                        color = var_colors[var_idx % len(var_colors)]
                        marker = var_markers[var_idx % len(var_markers)]
                        
                        # Initial scatter plot (will be updated in animation)
                        scatter = ax.scatter(x_pos, np.zeros(n_sizes), s=100, alpha=0.7, 
                                           color=color, marker=marker, edgecolors='black', 
                                           linewidths=1.5, label=VAR_NAME.replace('FATES_', ''))
                        loc_scatters.append(scatter)
                    
                    # Setup plot
                    ax.set_xlabel('Size Class', fontsize=12, fontweight='bold')
                    # Use the first variable's units (assuming all have same units)
                    ax.set_ylabel(f'{UNIT_LABELS[VAR_NAMES[0]]}', fontsize=12, fontweight='bold')
                    ax.set_title(f'{name} - Year 0, Month 01', 
                               fontsize=14, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(size_labels, rotation=45, ha='right')
                    
                    ax.grid(axis='both', alpha=0.3, zorder=1)
                    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
                    
                    # Add location info text
                    country_name = get_country(actual_lat, actual_lon)
                    us_state = get_us_state(actual_lat, actual_lon)
                    if us_state:
                        location_text = f"{us_state}, USA - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
                    else:
                        location_text = f"{country_name} - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
                    ax.text(0.02, 0.98, location_text, transform=ax.transAxes, 
                           fontsize=9, fontweight='bold', ha='left', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=10)
                    
                    scatter_artists.append(loc_scatters)
                
                # Hide unused subplots
                for idx in range(n_locations, len(axs)):
                    axs[idx].axis('off')
                
                plt.tight_layout()
                
                # Determine y-axis limits by scanning first and last files
                print("\nDetermining y-axis limits...")
                max_value_overall = 0
                sample_files = []
                if n_times > 0:
                    sample_files.append(file_time_map[0])
                if n_times > 1:
                    sample_files.append(file_time_map[-1])
                
                for file_path, time_idx, _ in sample_files:
                    with xr.open_dataset(file_path, decode_times=False) as ds:
                        for VAR_NAME in VAR_NAMES:
                            var_data = ds[VAR_NAME].isel(time=time_idx)
                            for grid_idx, _, _, _ in location_indices:
                                data = var_data.isel({spatial_dim: grid_idx}).values
                                data = data * SCALING_FACTORS[VAR_NAME]
                                max_value_overall = max(max_value_overall, np.nanmax(data))
                
                # Set y-axis limits (same for all locations since all variables share same axes)
                yax_scaler = 1.2
                for loc_idx in range(n_locations):
                    axs[loc_idx].set_ylim(0, max_value_overall * yax_scaler)
                
                # Animation update function
                def update_scatter(frame):
                    file_path, time_idx, _ = file_time_map[frame]
                    
                    with xr.open_dataset(file_path, decode_times=False) as ds:
                        # Update each subplot
                        for loc_idx in range(n_locations):
                            grid_idx, _, _, name = location_indices[loc_idx]
                            ax = axs[loc_idx]
                            
                            for var_idx, VAR_NAME in enumerate(VAR_NAMES):
                                scatter = scatter_artists[loc_idx][var_idx]
                                
                                # Get data for this location and variable
                                var_data = ds[VAR_NAME].isel(time=time_idx)
                                data = var_data.isel({spatial_dim: grid_idx}).values
                                data = data * SCALING_FACTORS[VAR_NAME]
                                
                                # Update scatter plot positions
                                scatter.set_offsets(np.c_[x_pos, data])
                            
                            # Update title with time (once per location)
                            actual_timestep = (start_timestep or 0) + (frame * FILE_INTERVAL)
                            year = actual_timestep // 12
                            month = actual_timestep % 12 + 1
                            ax.set_title(f'{name} - Year {year}, Month {month:02d}',
                                       fontsize=14, fontweight='bold')
                    
                    return [scatter for loc_scatters in scatter_artists for scatter in loc_scatters]
                
                # Create animation
                print(f"\nCreating scatter plot animation with {n_times} frames...")
                anim = animation.FuncAnimation(
                    fig, update_scatter, frames=n_times,
                    interval=tint,
                    blit=True,
                    repeat=True
                )
                
                # Save animation
                pft_output_file = OUTPUT_FILE.replace('.gif', f'_{pft_upper}.gif').replace('.mp4', f'_{pft_upper}.mp4')
                print(f"\nSaving {pft_upper} animation to {pft_output_file}...")
                
                output_dir = Path(pft_output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if OUTPUT_FORMAT == "mp4":
                    import shutil
                    if shutil.which('ffmpeg') is None:
                        print("WARNING: ffmpeg not found. Falling back to GIF format.")
                        pft_output_file = pft_output_file.replace('.mp4', '.gif')
                        writer = animation.PillowWriter(fps=1000/tint)
                        anim.save(pft_output_file, writer=writer, dpi=100)
                    else:
                        writer = animation.FFMpegWriter(fps=1000/tint, metadata=dict(artist='FATES Analysis'), bitrate=1800)
                        anim.save(pft_output_file, writer=writer, dpi=100)
                elif OUTPUT_FORMAT == "gif":
                    writer = animation.PillowWriter(fps=1000/tint)
                    anim.save(pft_output_file, writer=writer, dpi=100)
                
                print(f"Animation saved successfully!")
                plt.close(fig)
            
        elif 'PLOT_TYPE' in locals() and PLOT_TYPE == "point_line_sz":
            print("\n" + "="*60)
            print("CREATING POINT-BASED LINE PLOT (TIME SERIES) - STATIC")
            print("="*60)
            
            # Process all PFTs in the list
            for pft_name in PFT_LIST:
                pft_upper = pft_name.upper()
                print(f"\n{'='*60}")
                print(f"Processing PFT: {pft_upper}")
                print(f"{'='*60}\n")
                
                # Read locations from PFT-specific file
                print(f"Reading locations from PFT file for {pft_upper}...")
                LOCATIONS = read_location_file(pft_name, max_sites=8)
                
                if not LOCATIONS:
                    print(f"Skipping PFT: {pft_upper}")
                    continue
                
                print(f"Loaded {len(LOCATIONS)} locations for {pft_upper}")
                
                # Find nearest grid points to requested locations
                location_indices = []
                for target_lat, target_lon, name in LOCATIONS:
                    idx, actual_lat, actual_lon = find_nearest_point(target_lat, target_lon, lat, lon)
                    location_indices.append((idx, actual_lat, actual_lon, name))
                    print(f"Location '{name}': target=({target_lat}, {target_lon}), actual=({actual_lat:.2f}, {actual_lon:.2f}), idx={idx}")
                
                n_locations = len(location_indices)
                n_vars = len(VAR_NAMES)
                
                # Get size dimension info from first file
                with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
                    first_var = VAR_NAMES[0]
                    var_data = ds[first_var]
                    
                    # Find size or PFT dimension
                    size_dim = None
                    for dim in var_data.dims:
                        if 'size' in dim.lower() or 'levscls' in dim.lower():
                            size_dim = dim
                            break
                    
                    # If no size dimension found, check for PFT dimension
                    if size_dim is None:
                        for dim in var_data.dims:
                            if 'pft' in dim.lower() or 'levpft' in dim.lower():
                                size_dim = dim
                                print(f"Note: Using PFT dimension '{size_dim}' instead of size dimension")
                                break
                    
                    if size_dim is None:
                        print(f"ERROR: Could not find size or PFT dimension in {first_var}")
                        print(f"Available dimensions: {var_data.dims}")
                        exit(1)
                    
                    n_sizes = len(ds[size_dim])
                    is_pft_dim = 'pft' in size_dim.lower()
                    dim_type = "PFT classes" if is_pft_dim else "size classes"
                    print(f"\nFound {n_sizes} {dim_type} in dimension '{size_dim}'")
                    
                    # Determine spatial dimension
                    if 'ncol' in var_data.dims:
                        spatial_dim = 'ncol'
                    elif 'lndgrid' in var_data.dims:
                        spatial_dim = 'lndgrid'
                    else:
                        print(f"ERROR: No spatial dimension found")
                        exit(1)
                    
                    print(f"Using spatial dimension: {spatial_dim}")
                
                # Create labels based on dimension type
                if is_pft_dim:
                    size_labels = [f"PFT{i+1}" for i in range(n_sizes)]
                else:
                    size_labels = [f"SC{i+1}" for i in range(n_sizes)]
                
                # Define distinct colors for each variable
                # Using a custom color palette with high contrast
                distinct_colors = [
                    '#1f77b4',  # Blue
                    '#ff7f0e',  # Orange
                    '#2ca02c',  # Green
                    '#d62728',  # Red
                    '#9467bd',  # Purple
                    '#8c564b',  # Brown
                    '#e377c2',  # Pink
                    '#7f7f7f',  # Gray
                    '#bcbd22',  # Olive
                    '#17becf',  # Cyan
                ]
                var_colors = [distinct_colors[i % len(distinct_colors)] for i in range(n_vars)]
                
                # Define markers for each PFT/size class
                markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', '<', '>', 'd']
                
                # Load all data for all timesteps
                print("\nLoading data for all timesteps...")
                all_data = {}  # Dict: (loc_idx, var_idx) -> array of shape (n_times, n_sizes)
                time_values = []  # Store time values for x-axis
                
                for frame_idx, (file_path, time_idx, time_val) in enumerate(file_time_map):
                    if frame_idx % 10 == 0:
                        print(f"\rLoading timestep {frame_idx+1}/{n_times}...", end='', flush=True)
                    
                    with xr.open_dataset(file_path, decode_times=False) as ds:
                        # Store time value (convert to months since start)
                        actual_timestep = (start_timestep or 0) + (frame_idx * FILE_INTERVAL)
                        time_values.append(actual_timestep)
                        
                        for loc_idx, (grid_idx, _, _, _) in enumerate(location_indices):
                            for var_idx, VAR_NAME in enumerate(VAR_NAMES):
                                var_data = ds[VAR_NAME].isel(time=time_idx)
                                data = var_data.isel({spatial_dim: grid_idx}).values
                                data = data * SCALING_FACTORS[VAR_NAME]
                                
                                key = (loc_idx, var_idx)
                                if key not in all_data:
                                    all_data[key] = []
                                all_data[key].append(data)
                
                print(f"\nData loading complete!")
                
                # Convert lists to arrays
                for key in all_data:
                    all_data[key] = np.array(all_data[key])  # Shape: (n_times, n_sizes)
                
                time_values = np.array(time_values)
                
                # Setup figure - one subplot per location
                n_cols = 4
                n_rows = 2
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
                axs = axs.flatten()
                
                # Add CASENAME as figure title
                fig.suptitle(f'{CASENAME}', fontsize=16, fontweight='bold', y=0.995)
                
                # Plot all data at once (static plot)
                # One subplot per site, with variable=color and PFT=marker
                print("\nCreating line plots...")
                
                for loc_idx in range(n_locations):
                    grid_idx, actual_lat, actual_lon, name = location_indices[loc_idx]
                    ax = axs[loc_idx]
                    
                    # Get location info for title
                    country_name = get_country(actual_lat, actual_lon)
                    us_state = get_us_state(actual_lat, actual_lon)
                    if us_state:
                        location_str = f"{us_state}, USA"
                    else:
                        location_str = country_name
                    
                    # Plot each combination of variable and PFT/size class
                    for var_idx, VAR_NAME in enumerate(VAR_NAMES):
                        color = var_colors[var_idx % len(var_colors)]
                        var_label = VAR_NAME.replace('FATES_', '').replace('MORTALITY_', 'M_').replace('_CFLUX_PF', '')
                        
                        key = (loc_idx, var_idx)
                        data_all_times = all_data[key]  # Shape: (n_times, n_sizes)
                        
                        for size_idx in range(n_sizes):
                            marker = markers[size_idx % len(markers)]
                            y_data = data_all_times[:, size_idx]
                            
                            # Create label for legend (only for first variable to show markers, or first size to show colors)
                            if var_idx == 0 and size_idx < n_sizes:
                                # Show marker legend
                                label = f"{size_labels[size_idx]}"
                            elif size_idx == 0 and var_idx < n_vars:
                                # Show color legend
                                label = var_label
                            else:
                                label = None
                            
                            ax.plot(time_values, y_data, linewidth=1.5, alpha=0.7, 
                                   color=color, marker=marker, markersize=4, markevery=max(1, len(time_values)//20),
                                   label=label)
                    
                    # Setup plot
                    ax.set_xlabel('Time (Year-Month)', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'{UNIT_LABELS[VAR_NAMES[0]]}', fontsize=12, fontweight='bold')
                    ax.set_title(f'{name} - {location_str}\nLat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°', 
                               fontsize=12, fontweight='bold')
                    
                    # Format x-axis to show years and months
                    # Create custom tick labels showing Year-Month
                    n_ticks = min(10, len(time_values))  # Show up to 10 tick marks
                    tick_indices = np.linspace(0, len(time_values)-1, n_ticks, dtype=int)
                    tick_positions = time_values[tick_indices]
                    tick_labels = [f'Y{int(t//12)}-M{int(t%12+1):02d}' for t in tick_positions]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                    
                    ax.grid(axis='both', alpha=0.3, zorder=1)
                    
                    # Create custom legend with two sections
                    from matplotlib.lines import Line2D
                    # Variable legend (colors)
                    var_legend_elements = [Line2D([0], [0], color=var_colors[i % len(var_colors)], linewidth=2, 
                                                  label=VAR_NAMES[i].replace('FATES_', '').replace('MORTALITY_', 'M_').replace('_CFLUX_PF', ''))
                                          for i in range(n_vars)]
                    # PFT/Size legend (markers)
                    pft_legend_elements = [Line2D([0], [0], color='gray', marker=markers[i % len(markers)], 
                                                  linewidth=0, markersize=6, label=size_labels[i])
                                          for i in range(n_sizes)]
                    
                    # Combine legends
                    all_legend_elements = var_legend_elements + pft_legend_elements
                    ax.legend(handles=all_legend_elements, loc='upper right', fontsize=7, ncol=2, framealpha=0.9)
                    
                    # Set x-axis limits to full time range
                    ax.set_xlim(time_values[0], time_values[-1])
                    
                    # Add location info text
                    country_name = get_country(actual_lat, actual_lon)
                    us_state = get_us_state(actual_lat, actual_lon)
                    if us_state:
                        location_text = f"{us_state}, USA - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
                    else:
                        location_text = f"{country_name} - Lat: {actual_lat:.2f}°, Lon: {actual_lon:.2f}°"
                    ax.text(0.02, 0.98, location_text, transform=ax.transAxes, 
                           fontsize=9, fontweight='bold', ha='left', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=10)
                
                # Hide unused subplots
                for idx in range(n_locations, len(axs)):
                    axs[idx].axis('off')
                
                # Determine y-axis limits from all data
                print("Determining y-axis limits...")
                max_value_overall = 0
                for loc_idx in range(n_locations):
                    for var_idx in range(n_vars):
                        key = (loc_idx, var_idx)
                        data_all_times = all_data[key]  # Shape: (n_times, n_sizes)
                        max_value_overall = max(max_value_overall, np.nanmax(data_all_times))
                
                # Set y-axis limits for all subplots
                yax_scaler = 1.2
                for loc_idx in range(n_locations):
                    axs[loc_idx].set_ylim(0, max_value_overall * yax_scaler)
                
                plt.tight_layout()
                
                # Save static plot as PNG
                pft_output_file = OUTPUT_FILE.replace('.gif', f'_{pft_upper}.png').replace('.mp4', f'_{pft_upper}.png')
                print(f"\nSaving {pft_upper} plot to {pft_output_file}...")
                
                output_dir = Path(pft_output_file).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                fig.savefig(pft_output_file, dpi=150, bbox_inches='tight')
                print(f"Plot saved successfully!")
                plt.close(fig)
            
        else:
            # Original map-based plotting code
            # Create figure and subplots
            print("\nSetting up animation...")
            # Determine number of subplots needed
            n_subplots = 0
            for VAR_NAME in VAR_NAMES:
                if var_metadata[VAR_NAME]['split_extra_dim']:
                    n_subplots += var_metadata[VAR_NAME]['extra_dim_size']
                else:
                    n_subplots += 1
    
            # Landscape layout: prefer 3-4 columns for wider-than-tall arrangement
            if n_subplots == 2:
                n_cols = 2
                n_rows = 2
            else:
                n_cols = min(n_subplots, 4) if n_subplots > 2 else min(n_subplots, 2)
                if(n_subplots==6): 
                    n_cols=3
                n_rows = (n_subplots + n_cols - 1) // n_cols  # Ceiling division
    
            # Always use cartopy projection for map background
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(14*n_cols, 8*n_rows), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})
    
            # Make axs always a list for consistent indexing
            if n_subplots == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
    
            # Initialize plots for each variable/subplot
            ims = []
            cbars = []
            subplot_info = []  # Store (VAR_NAME, extra_dim_idx) for each subplot
            subplot_idx = 0
    
            for VAR_NAME in VAR_NAMES:
                meta = var_metadata[VAR_NAME]
                vmin = meta['vmin']
                vmax = meta['vmax']
        
                # Determine how many subplots for this variable
                if meta['split_extra_dim']:
                    n_members = meta['extra_dim_size']
                    print(f"\n{VAR_NAME}: Creating {n_members} subplots (one per {meta['extra_dims'][0]})")
                    print(f"  Color scale: {vmin:.4f} to {vmax:.4f}")
                else:
                    n_members = 1
                    print(f"\n{VAR_NAME}: Color scale: {vmin:.4f} to {vmax:.4f}")
        
                for member_idx in range(n_members):
                    ax = axs[subplot_idx]
            
                    # Extract the appropriate data slice for first frame
                    if meta['split_extra_dim']:
                        first_frame = meta['first_frame'][member_idx, :] * meta['scaling_factor']
                    else:
                        first_frame = meta['first_frame'] * meta['scaling_factor']
            
                    # Setup map features - add coastlines and country borders
                    ax.coastlines(linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.2, edgecolor='gray')
                    if use_cartopy and 'lat' in meta['dims']:
                        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            
                    # Initialize plot
                    if use_cartopy and 'lat' in meta['dims']:
                        im = ax.pcolormesh(lon, lat, first_frame, 
                                          transform=ccrs.PlateCarree(),
                                          cmap=cmap_choose, vmin=vmin, vmax=vmax)
                    else:
                        # Unstructured grid - use scatter plot with map background
                        im = ax.scatter(lon, lat, c=first_frame,
                                       cmap=cmap_choose, vmin=vmin, vmax=vmax, s=7,
                                       transform=ccrs.PlateCarree())
            
                    ims.append(im)
                    subplot_info.append((VAR_NAME, member_idx if meta['split_extra_dim'] else None))
            
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
                    cbar.set_label(f'{meta["units"]}', fontsize=14, fontweight='bold')
                    cbars.append(cbar)
            
                    # Set title with year-month (relative to start)
                    # Frame 0 represents timestep 0, accounting for FILE_INTERVAL
                    timestep = 0 * FILE_INTERVAL
                    year = timestep // 12 + 1
                    month = timestep % 12 + 1
                    time_str = f'Year {year}, Month {month:02d}'
                    if meta['split_extra_dim']:
                        extra_dim_name = meta['extra_dims'][0]
                        ax.set_title(f'{VAR_NAME} [{extra_dim_name}={member_idx}] - {time_str}', fontsize=16, fontweight='bold')
                    else:
                        ax.set_title(f'{VAR_NAME} - {time_str}', fontsize=16, fontweight='bold')
            
                    subplot_idx += 1
    
            # Hide any extra subplots
            for idx in range(n_subplots, len(axs)):
                axs[idx].axis('off')
    
            plt.tight_layout()
    
            # Cache for keeping files open
            current_file = None
            current_ds = None
    
            # Animation update function
            def update_frame(frame):
                """Update function for animation - loads data on demand"""
                global current_file, current_ds
        
                print(f"\rProcessing frame {frame+1}/{n_times}", end='', flush=True)
        
                # Get file and time index for this frame
                file_path, time_idx, time_val = file_time_map[frame]
        
                # Only open a new file if we've moved to a different file
                if current_file != file_path:
                    if current_ds is not None:
                        current_ds.close()
                    current_ds = xr.open_dataset(file_path, decode_times=False)
                    current_file = file_path
        
                # Update each subplot
                artists = []
                for idx, (VAR_NAME, member_idx) in enumerate(subplot_info):
                    meta = var_metadata[VAR_NAME]
                    ax = axs[idx]
                    im = ims[idx]
            
                    # Load data for this timestep
                    data_timestep = current_ds[VAR_NAME].isel(time=time_idx)
            
                    # Extract appropriate data based on whether we're splitting dimensions
                    if meta['split_extra_dim']:
                        # Get specific member of extra dimension
                        data = data_timestep.isel({meta['extra_dims'][0]: member_idx}).values
                    elif meta['has_extra_dim']:
                        data = data_timestep.sum(dim=meta['extra_dims']).values
                    else:
                        data = data_timestep.values
            
                    # Apply scaling factor
                    data = data * meta['scaling_factor']
            
                    # Update plot
                    if use_cartopy and 'lat' in meta['dims']:
                        im.set_array(data.ravel())
                    elif not use_cartopy:
                        im.set_array(data)
                    else:
                        im.set_data(data)
            
                    # Update title with year-month
                    # Convert frame number to actual timestep accounting for FILE_INTERVAL and start offset
                    actual_timestep = (start_timestep or 0) + (frame * FILE_INTERVAL)
                    year = actual_timestep // 12
                    month = actual_timestep % 12 + 1
                    time_str = f'Year {year}, Month {month:02d}'
                    if meta['split_extra_dim']:
                        extra_dim_name = meta['extra_dims'][0]
                        ax.set_title(f'{VAR_NAME} [{extra_dim_name}={member_idx}] - {time_str}', fontsize=16, fontweight='bold')
                    else:
                        ax.set_title(f'{VAR_NAME} - {time_str}', fontsize=16, fontweight='bold')
                    artists.append(im)
        
                return artists
    
            # Create animation
            print(f"Creating animation with {n_times} frames...")
            anim = animation.FuncAnimation(
                fig, update_frame, frames=n_times,
                interval=tint,  # 400ms between frames (2.5 fps)
                blit=True,
                repeat=True
            )
    
        # Save animation as GIF or MP4 (only for non-point_bars cases)
        # point_bars case saves animations within plot_pft_animation function
        if 'anim' in locals():
            print(f"\n\nSaving animation to {OUTPUT_FILE}...")
        
            # Create output directory if it doesn't exist
            output_dir = Path(OUTPUT_FILE).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {output_dir}")
        
            if OUTPUT_FORMAT == "mp4":
                # Check if ffmpeg is available
                import shutil
                if shutil.which('ffmpeg') is None:
                    print("WARNING: ffmpeg not found. Falling back to GIF format.")
                    print("To use MP4, install ffmpeg or run: module load FFmpeg")
                    # Change to GIF format
                    OUTPUT_FILE = OUTPUT_FILE.replace('.mp4', '.gif')
                    writer = animation.PillowWriter(fps=2.5)
                    anim.save(OUTPUT_FILE, writer=writer, dpi=100)
                else:
                    writer = animation.FFMpegWriter(fps=2.5, metadata=dict(artist='FATES Analysis'), bitrate=1800)
                    anim.save(OUTPUT_FILE, writer=writer, dpi=100)
            else:
                writer = animation.PillowWriter(fps=2.5)
                anim.save(OUTPUT_FILE, writer=writer, dpi=100)
        
            print(f"\nAnimation successfully saved to: {OUTPUT_FILE}")
            print(f"Total frames: {n_times}")
        
            # Close any open dataset (only exists for map-based plots)
            if 'current_ds' in locals() and current_ds is not None:
                current_ds.close()
            plt.close(fig)
    
        print("\nDone!")

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("ALL CASES COMPLETED")
print("="*80)
