#!/usr/bin/env python
"""
Script to create a video of FATES_GPP timeseries from NorESM output files
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

# Configuration
DATA_DIR = Path("/datalake/NS9560K/rosief/i1850.ne16pg3_tn14.fatesnocomp.noresm3_0_beta08.CPLHIST.2025-12-18/lnd/hist/")
cmap_choose = 'YlGn'

# Variable names - can be a single string or a list of strings
VAR_NAMES = ["FATES_MORTALITY_HYDRAULIC_CFLUX_PF"]  # or 

VAR_NAMES = ["FATES_GPP", 
"FATES_NPP",
 "FATES_NEP",
 "FATES_FIRE_CLOSS",
  "FATES_HET_RESP", 
  "FATES_LITTER_IN"]

#VAR_NAMES = ["FATES_MORTALITY_FIRE_CFLUX_PF", 
# "FATES_MORTALITY_HYDRAULIC_CFLUX_PF",
#"FATES_MORTALITY_CSTARV_CFLUX_PF",
#"FATES_MORTALITY_BACKGROUND_CFLUX_PF",
#"FATES_MORTALITY_SENESCENCE_CFLUX_PF",
#"FATES_HARVEST_WOODPROD_C_FLUX"]
# Scaling factors for each variable (applied to convert units)
# Default: 3600*24*365 converts from per-second to per-year
SCALING_FACTORS = {var: 3600*24*365 for var in VAR_NAMES}

VAR_NAMES = ["FATES_MORTALITY_BACKGROUND_CFLUX_PF"] 
VAR_NAMES = ["FATES_LAI_CANOPY_SZ"] 
SCALING_FACTORS = {var: 1 for var in VAR_NAMES}

# Convert to list if single string provided
if isinstance(VAR_NAMES, str):
    VAR_NAMES = [VAR_NAMES]



OUTPUT_FILE = "/datalake/NS9560K/www/diagnostics/noresm/rosief/gifs/fates_" + "_".join(VAR_NAMES) + "_timeseries.gif"
N_YEARS = 3 # Number of years to process (None for all years)
SAME_COLOR_SCALE = True  # Set to True to use same color scale for all variables
# Unit labels for each variable (applied when scaling is used)
# If not specified, original units from the file will be used
UNIT_LABELS = {var: "KgC/m2/yr" for var in VAR_NAMES}


print(f"Reading data from: {DATA_DIR}")
print(f"Variables to plot: {VAR_NAMES}")
print(f"Same color scale for all: {SAME_COLOR_SCALE}")
print(f"Output file: {OUTPUT_FILE}")
if N_YEARS is not None:
    print(f"Processing first {N_YEARS} years only")

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
    # Only scan as many files as needed for N_YEARS
    file_time_map = []
    
    # Determine how many timesteps we need
    max_timesteps = None
    if N_YEARS is not None:
        max_timesteps = N_YEARS * 12
        print(f"Need {max_timesteps} timesteps for {N_YEARS} years")
    
    print("Scanning files for timesteps...")
    files_scanned = 0
    for i, file_path in enumerate(all_files):
        # Stop if we have enough timesteps
        if max_timesteps is not None and len(file_time_map) >= max_timesteps:
            break
        
        files_scanned += 1
        if files_scanned % 10 == 0 or files_scanned == 1:
            print(f"\rScanning file {files_scanned}...", end='', flush=True)
        
        with xr.open_dataset(file_path, decode_times=False) as ds:
            n_times_in_file = len(ds.time)
            for t in range(n_times_in_file):
                file_time_map.append((file_path, t, ds.time.values[t]))
                # Stop if we have enough timesteps
                if max_timesteps is not None and len(file_time_map) >= max_timesteps:
                    break
    
    print(f"\n\nScanned {files_scanned} files, found {len(file_time_map)} timesteps")
    
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
        global_vmin = float(np.nanpercentile(all_values, 2))
        global_vmax = float(np.nanpercentile(all_values, 98))
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
                vmin = float(np.nanpercentile(all_vals, 2))
                vmax = float(np.nanpercentile(all_vals, 98))
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
    n_cols = min(n_subplots, 4) if n_subplots > 2 else min(n_subplots, 2)
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
            timestep = 0
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
            
            # Update title with year-month (relative to start)
            year = frame // 12 + 1
            month = frame % 12 + 1
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
        interval=200,  # 200ms between frames (5 fps)
        blit=True,
        repeat=True
    )
    
    # Save animation as GIF
    print(f"\n\nSaving animation to {OUTPUT_FILE}...")
    writer = animation.PillowWriter(fps=5)
    
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
