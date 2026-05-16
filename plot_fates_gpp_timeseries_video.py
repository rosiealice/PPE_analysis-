#!/usr/bin/env python
"""
Script to create a video of FATES_GPP timeseries from NorESM output files
Reads files individually from the lnd/hist directory and creates an animated visualization
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from pathlib import Path
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import shutil
import sys
import re

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
CASENAME="i2000.f45_f45_mg37.fatesnocomp.noresm3_0_beta10.CRUJRA.30j.2026-02-06"
#CASENAME="i1850.ne16pg3_tn14.fatesnocomp.noresm3_0_beta08.CPLHIST.2025-12-18"
#CASENAME="i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta09.CPLHIST.BT3_LSP_MS3.25_BT1EBT_bias30j_LU2000.2026-01-30"


DATA_DIR = Path(f"{DATA_PATH}/{CASENAME}/lnd/hist/")

jessie_run=1
if(jessie_run==1):
    DATA_PATH= "/datalake/NS9188K/share/jessien/feb26/"
    CASENAME="noresm-fates_f45_2000s_ent_agbonly.2026-02-03"
    CASENAME="noresm-fates_ne16_transient_1900_borealseeds.2025-11-19"
    CASENAME="noresm-fates_f45_2000s_feb20c.2026-02-20"
    CASENAME="ne16_transient_1900.2026-02-06"

    DATA_DIR = Path(f"{DATA_PATH}/{CASENAME}/run/")


SAME_COLOR_SCALE = True  # Set to True to use same color scale for all variables
cmap_choose = 'YlGn'
vminv = 1; vmaxv=99
LOG_COLOR_SCALE = False

OUTPUT_FORMAT = "gif"  # "gif" or "mp4"

# Loop over multiple cases
for case in [1]:  # Specify which cases to run, e.g., [0, 1, 2] or [2]
    print(f"\n{'='*80}")
    print(f"PROCESSING CASE {case}")
    print(f"{'='*80}\n")

    LOG_COLOR_SCALE = False
    
    if(case==0):
# Variable names - can be a single string or a list of string s
        CASE_NAME = "understorey_mortality"
        VAR_NAMES = ["MORTALITY_CROWNAREA_UNDERSTORY"]  # or 
        CASE_NAME = "canopy_mortality"
        VAR_NAMES = ["MORTALITY_CROWNAREA_CANOPY"]  # or
        CASE_NAME = "MORTALITY_CFLUX_PF"
        VAR_NAMES = ["FATES_MORTALITY_CFLUX_PF"]  # or 
        CASE_NAME = "MORTALITY_PF"
        VAR_NAMES = ["FATES_MORTALITY_PF"]  # or  
        #CASE_NAME = "MORTALITY_CFLUX_PF"
        #VAR_NAMES = ["FATES_MORTALITY_CFLUX_PF"]  # or
        CASE_NAME = "NBP"
        VAR_NAMES = ["FCO2"]  # or
        #CASE_NAME = "LUCHANGE_WOODPROD_C_FLUX"
        #VAR_NAMES = ["FATES_LUCHANGE_WOODPROD_C_FLUX"]  # or

        #CASE_NAME = "btran"
        #VAR_NAMES = ["BTRAN"]  # 
        SCALING_FACTORS = {var:1 for var in VAR_NAMES}
        YEAR_RANGE = (1900, 1990) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 12 # Process every Nth timestep (1=all, 2=every other, 3=every third, etc.)
        tint=3
        UNIT_LABELS = {}  # Use units from NetCDF variable attrs by default
        LOG_COLOR_SCALE = False

    if(case==1):
        CASE_NAME = "Cfluxes"
        VAR_NAMES = ["FATES_GPP", 
        "FATES_NPP",
        "FCO2" ,
        "FATES_LUCHANGE_WOODPROD_C_FLUX",
        "FATES_FIRE_CLOSS",
        "FATES_HET_RESP"]
        # Convert from kg/m2/s to kg/m2/yr by multiplying by 3600*24*365
        SCALING_FACTORS = {var: 3600*24*365 for var in VAR_NAMES}
        SCALING_FACTORS["FATES_LUCHANGE_WOODPROD_C_FLUX"] = 1
        print(f"Applied scaling factors: {SCALING_FACTORS}")
        YEAR_RANGE = (1960, 1962) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep (1=all, 2=every other, 3=every third, etc.)
        tint=3
        UNIT_LABELS = {}  # Use units from NetCDF variable attrs by default
        OUTPUT_FORMAT = "mp4"  # "gif" or "mp4"

    if(case==2):
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
        YEAR_RANGE = (2000, 2020) # Year range to process as (start_year, end_year), e.g., (5, 10) for years 5-10. None for all years.
        FILE_INTERVAL = 1 # Process every Nth timestep (1=all, 2=every other, 3=every third, etc.)
        tint=5
        OUTPUT_FORMAT = "gif"  # "gif" or "mp4"
        UNIT_LABELS = {var: "units not specified" for var in VAR_NAMES}

    if(case==3):
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
        SAME_COLOR_SCALE = False  # Set to True to use same color scale for all variables
        vmaxv=97

    if(case==4):
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


    # Convert to list if single string provided
    if isinstance(VAR_NAMES, str):
        VAR_NAMES = [VAR_NAMES]



    OUTPUT_PATH = "/datalake/NS9560K/www/diagnostics/noresm/rosief/"+CASENAME+"/gifs"
    requested_output_format = OUTPUT_FORMAT.lower()
    if requested_output_format == "mp4" and shutil.which('ffmpeg') is None:
        effective_output_format = "gif"
        print("ffmpeg not found; using GIF output.")
    else:
        effective_output_format = requested_output_format

    file_ext = "gif" if effective_output_format == "gif" else "mp4"
    year_range_str = f"_y{YEAR_RANGE[0]}-{YEAR_RANGE[1]}" if YEAR_RANGE is not None else ""
    OUTPUT_FILE = OUTPUT_PATH+ "/fates_" + CASE_NAME + year_range_str + "_timeseries." + file_ext
    # Unit labels for each variable (applied when scaling is used)
    # If not specified, original units from the file will be used

    print(f"Reading data from: {DATA_DIR}")
    print(f"Variables to plot: {VAR_NAMES}")
    print(f"Same color scale for all: {SAME_COLOR_SCALE}")
    print(f"Log color scale: {LOG_COLOR_SCALE}")
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
        # Filter by calendar year parsed from filename to avoid assumptions about run start year.
        file_time_map = []

        print("Building timestep list from filenames...")

        def _extract_year_from_name(file_path):
            file_name = Path(file_path).name
            # Match years after h0a/rh0a, e.g. *.h0a.1901-01.nc or *.rh0a.1994-01-01-00000.nc
            m = re.search(r"\.(?:r?h0a)\.(\d{4})-\d{2}", file_name)
            if m:
                return int(m.group(1))
            return None

        # First apply YEAR_RANGE filter (inclusive) on files.
        candidate_files = []
        for file_path in all_files:
            file_year = _extract_year_from_name(file_path)
            if YEAR_RANGE is not None and file_year is not None:
                if not (YEAR_RANGE[0] <= file_year <= YEAR_RANGE[1]):
                    continue
            candidate_files.append(file_path)

        if YEAR_RANGE is not None:
            expected_output = len(candidate_files) // FILE_INTERVAL if FILE_INTERVAL > 0 else 0
            print(
                f"Found {len(candidate_files)} files in years {YEAR_RANGE[0]}-{YEAR_RANGE[1]}, "
                f"using every {FILE_INTERVAL} timestep -> ~{expected_output} output frames"
            )

        # Then sample every Nth timestep from the filtered files.
        for timestep_counter, file_path in enumerate(candidate_files):
            if (timestep_counter+1) % FILE_INTERVAL != 0:
                continue

            # Open file only to get the time value
            with xr.open_dataset(file_path, decode_times=False) as ds:
                file_time_map.append((file_path, 0, ds.time.values[0]))

            processed = timestep_counter + 1
            if processed % 50 == 0 or processed == 1:
                print(f"\rProcessed {processed} timesteps, selected {len(file_time_map)}...", end='', flush=True)

        print(f"\n\nProcessed {len(candidate_files)} timesteps, selected {len(file_time_map)} for animation")

        if not file_time_map:
            print("\nERROR: No timesteps selected after YEAR_RANGE/FILE_INTERVAL filtering.")
            print(f"YEAR_RANGE={YEAR_RANGE}, FILE_INTERVAL={FILE_INTERVAL}")
            print("Try expanding YEAR_RANGE or lowering FILE_INTERVAL.")
            exit(1)

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
            if LOG_COLOR_SCALE:
                positive_values = all_values[np.isfinite(all_values) & (all_values > 0)]
                if positive_values.size == 0:
                    raise ValueError("Log color scale requires at least one positive value")
                global_vmin = float(np.nanpercentile(positive_values, vminv))
                global_vmax = float(np.nanpercentile(positive_values, vmaxv))
            else:
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
                    if LOG_COLOR_SCALE:
                        positive_values = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
                        if positive_values.size == 0:
                            raise ValueError(f"Log color scale requires positive values for {VAR_NAME}")
                        vmin = float(np.nanpercentile(positive_values, vminv))
                        vmax = float(np.nanpercentile(positive_values, vmaxv))
                    else:
                        vmin = float(np.nanpercentile(all_vals, 0))
                        vmax = float(np.nanpercentile(all_vals, 100))
                    var_metadata[VAR_NAME]['vmin'] = vmin
                    var_metadata[VAR_NAME]['vmax'] = vmax
                    print(f"  {VAR_NAME}: {vmin:.4f} to {vmax:.4f}")
    
        lat = var_metadata['lat']
        lon = var_metadata['lon']
        use_cartopy = var_metadata['use_cartopy']
    
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
                if LOG_COLOR_SCALE:
                    first_frame = np.where(first_frame > 0, first_frame, np.nan)
            
                # Setup map features - add coastlines and country borders
                ax.coastlines(linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.2, edgecolor='gray')
                if use_cartopy and 'lat' in meta['dims']:
                    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
            
                # Initialize plot
                norm = colors.LogNorm(vmin=vmin, vmax=vmax) if LOG_COLOR_SCALE else None
                if use_cartopy and 'lat' in meta['dims']:
                    im = ax.pcolormesh(lon, lat, first_frame, 
                                      transform=ccrs.PlateCarree(),
                                      cmap=cmap_choose, vmin=None if LOG_COLOR_SCALE else vmin,
                                      vmax=None if LOG_COLOR_SCALE else vmax,
                                      norm=norm)
                else:
                    # Unstructured grid - use scatter plot with map background
                    im = ax.scatter(lon, lat, c=first_frame,
                                   cmap=cmap_choose,
                                   vmin=None if LOG_COLOR_SCALE else vmin,
                                   vmax=None if LOG_COLOR_SCALE else vmax,
                                   norm=norm, s=7,
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
                if LOG_COLOR_SCALE:
                    data = np.where(data > 0, data, np.nan)
            
                # Update plot
                if use_cartopy and 'lat' in meta['dims']:
                    im.set_array(data.ravel())
                elif not use_cartopy:
                    im.set_array(data)
                else:
                    im.set_data(data)
            
                # Update title with year-month parsed from current filename.
                file_name = Path(current_file).name
                m = re.search(r"\.(?:r?h0a)\.(\d{4})-(\d{2})", file_name)
                if m:
                    year = int(m.group(1))
                    month = int(m.group(2))
                    time_str = f'Year {year}, Month {month:02d}'
                else:
                    time_str = 'Time unknown'
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
    
        # Save animation as GIF or MP4
        print(f"\n\nSaving animation to {OUTPUT_FILE}...")
    
        # Create output directory if it doesn't exist
        output_dir = Path(OUTPUT_FILE).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
        if effective_output_format == "mp4":
            writer = animation.FFMpegWriter(fps=2.5, metadata=dict(artist='FATES Analysis'), bitrate=1800)
            anim.save(OUTPUT_FILE, writer=writer, dpi=100)
        else:
            writer = animation.PillowWriter(fps=2.5)
            anim.save(OUTPUT_FILE, writer=writer, dpi=100)
    
        print(f"\nAnimation successfully saved to: {OUTPUT_FILE}")
        print(f"Total frames: {n_times}")
    
        # Close any open dataset
        if current_ds is not None:
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
