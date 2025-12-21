#!/usr/bin/env python
"""
Script to analyze ensemble model outputs from years 05-10
Extracts FATES_LAI, FATES_GPP, and FATES_NPP variables and plots them against ensemble members
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuration
ens_n = 2

BASE_PATH = "/datalake/NS9560K/rosief"
OUTPUT_ROOT = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/"

if(ens_n==1):
    PARAM_PATH = "/nird/home/rosief/gt/Run_NorESM_script/PPE_CRU_1850_soilc_scripts/PPE_param_files/lhs_params"
    paramfilename='params_lhs_20251211_1740_0'
    ensdir='ens_v1_soilc'
    PARAMETERS = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
              'fates_allom_d2bl1', 'fates_leaf_slatop']
    ENSEMBLE_MEMBERS = range(1, 17)  # n01 to n13
    output_dirname='ppe_2000_v1_soilc'
    YEARS = range(29, 30)  # Years 09 to 10
    casename = 'noresm_beta07_crujra_ppe_v1_'
    YEARS = range(29, 30)  # Years 09 to 10

elif(ens_n==2):
    PARAM_PATH = "/nird/home/rosief/gt/Run_NorESM_script/PPE_CRU_2000_c4g_scripts/PPE_param_files/lhs_params"
    paramfilename='params_lhs_20251221_0039_'
    ensdir='ens_v1_c4g'
    PARAMETERS = ['fates_leaf_vcmax25top','fates_leaf_slatop','fates_turnover_leaf_canopy','fates_landuse_grazing_rate','fates_fire_cg_strikes',]
    ENSEMBLE_MEMBERS = range(1, 21)  # n01 to n13
    output_dirname='ppe_2000_v1_c4g'
    YEARS = range(39, 40)  # Years 09 to 10
    casename = 'noresm_beta07_crujra_ppe_2000_v2_'

outputdir = OUTPUT_ROOT+output_dirname             
VARIABLES = ['FATES_LAI', 'FATES_GPP', 'FATES_NPP','FATES_VEGC']

# Define regions for analysis
# Note: For global regions, use 0-360 longitude range
REGIONS = {
    'north_65': {'lat_min': 65, 'lat_max': 90, 'lon_min': 0, 'lon_max': 360},
}

# BDTs region will be defined based on top grid cells with PFT 6 coverage
BDT_GRID_CELLS = None  # Will be populated after identifying top cells
# BETs region will be defined based on top grid cells with PFT 1 coverage
BET_GRID_CELLS = None  # Will be populated after identifying top cells
# ENTs region will be defined based on top grid cells with PFT 2 coverage
ENT_GRID_CELLS = None  # Will be populated after identifying top cells
# AC3G region will be defined based on top grid cells with PFT 12 coverage
AC3G_GRID_CELLS = None  # Will be populated after identifying top cells
# C3G region will be defined based on top grid cells with PFT 13 coverage
C3G_GRID_CELLS = None  # Will be populated after identifying top cells
# C4 region will be defined based on top grid cells with PFT 14 coverage
C4_GRID_CELLS = None  # Will be populated after identifying top cells

# Store results for both ensembles by region: {region: {variable: {ensemble_member: mean_value}}}
results_ppe_v1 = {region: {var: {} for var in VARIABLES} for region in REGIONS.keys()}
results_ppe_2000_v1 = {region: {var: {} for var in VARIABLES} for region in REGIONS.keys()}
# Store summer LAI results (months 7, 8, 9)
results_ppe_v1_summer_lai = {region: {} for region in REGIONS.keys()}
results_ppe_2000_v1_summer_lai = {region: {} for region in REGIONS.keys()}
# Store parameter values: {parameter_name: {ensemble_member: array_of_pft_values}}

param_values = {param: {} for param in PARAMETERS}

# Map regions to their PFT indices (0-based indexing)
REGION_PFT_MAP = {
    'north_65': None,  # Global region, no specific PFT
    'BDTs': 5,   # PFT 6 (index 5) - Broadleaf Deciduous Trees
    'BETs': 0,   # PFT 1 (index 0) - Broadleaf Evergreen Trees
    'ENTs': 1,   # PFT 2 (index 1) - Evergreen Needleleaf Trees
    'AC3G': 11,  # PFT 12 (index 11) - Arctic C3 Grass
    'C3G': 12,   # PFT 13 (index 12) - C3 Grass
    'C4': 13,    # PFT 14 (index 13) - C4 Grass
}

def is_valid_param_value(param_val):
    """Check if a parameter value is valid (not NaN)."""
    if isinstance(param_val, np.ndarray):
        return not np.all(np.isnan(param_val))
    else:
        return not np.isnan(param_val)

def get_param_value_for_region(param_name, ensemble_num, region_name):
    """Get parameter value for a specific region and ensemble member."""
    if param_name not in param_values or ensemble_num not in param_values[param_name]:
        return np.nan
    
    param_val = param_values[param_name][ensemble_num]
    
    # Handle special non-PFT parameters
    if param_name in ['fates_fire_cg_strikes', 'fates_landuse_grazing_rate']:
        return param_val
    
    # Handle PFT-specific parameters
    pft_idx = REGION_PFT_MAP.get(region_name, None)
    if isinstance(param_val, np.ndarray):
        if pft_idx is not None:
            return param_val[pft_idx]
        else:
            return param_val[0]  # Use first PFT for global regions
    else:
        return param_val

print("Starting ensemble analysis...")
print(f"Processing {len(list(ENSEMBLE_MEMBERS))} ensemble members")
print(f"Years: {min(YEARS)}-{max(YEARS)}")
print(f"Variables: {', '.join(VARIABLES)}\n")

# First, read parameter values
print("Reading parameter files...")
for n in ENSEMBLE_MEMBERS:
    param_file = f"{PARAM_PATH}/{paramfilename}{n:03d}.nc"
    try:
        ds_param = xr.open_dataset(param_file)
        print(f'\nEnsemble {n:02d} - param_file: {param_file}')
        
        # Read all parameters (store all PFT values)
        for param in PARAMETERS:
            if param in ds_param:
                # Get all PFT values as array
                param_array = ds_param[param].values
                # Flatten if multi-dimensional
                if param_array.ndim > 1:
                    param_array = param_array.flatten()
                
                # Handle special cases
                if param == 'fates_fire_cg_strikes':
                    # Single value parameter - store as scalar
                    if param_array.ndim == 0:
                        param_values[param][n] = float(param_array)
                    else:
                        param_values[param][n] = float(param_array[0]) if len(param_array) > 0 else np.nan
                    print(f"  {param}: single value = {param_values[param][n]:.6f}")
                elif param == 'fates_landuse_grazing_rate':
                    # 5-value parameter - use the 3rd value (index 2)
                    if param_array.ndim == 0:
                        # Scalar value, just use it
                        param_values[param][n] = float(param_array)
                    else:
                        param_values[param][n] = float(param_array[2]) if len(param_array) > 2 else np.nan
                    if param_array.ndim == 0:
                        print(f"  {param}: using value = {param_values[param][n]:.6f}")
                    else:
                        print(f"  {param}: using 3rd value = {param_values[param][n]:.6f} (from {len(param_array)} values)")
                else:
                    # PFT-indexed parameters
                    param_values[param][n] = param_array
                    if param_array.ndim == 0:
                        print(f"  {param}: single value = {float(param_array):.6f}")
                    else:
                        print(f"  {param}: {len(param_array)} PFT values, range [{param_array.min():.6f}, {param_array.max():.6f}]")
            else:
                print(f"  WARNING: {param} not found in file")
                if param in ['fates_fire_cg_strikes', 'fates_landuse_grazing_rate']:
                    param_values[param][n] = np.nan  # Single value
                else:
                    param_values[param][n] = np.full(14, np.nan)  # Assume 14 PFTs
        
        ds_param.close()
    except Exception as e:
        print(f"  ERROR reading {param_file}: {str(e)}")
        for param in PARAMETERS:
            if param in ['fates_fire_cg_strikes', 'fates_landuse_grazing_rate']:
                param_values[param][n] = np.nan  # Single value
            else:
                param_values[param][n] = np.full(14, np.nan)  # Store NaN array for all PFTs

print("\n" + "="*60)
print("Loading PFT grid cells from pre-generated files...\n")

# Path to PFT grid cell files (generated by extract_pft_dominated_pixels.py)
PFT_GRID_CELLS_PATH = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/pft_grid_cells/"

# PFT configuration: short_name -> full_name
PFT_NAMES = {
    'bdt': 'BDTs',
    'bet': 'BETs', 
    'ent': 'ENTs',
    'ac3g': 'AC3G',
    'c3g': 'C3G',
    'c4': 'C4'
}

# Initialize grid cell variables
BDT_GRID_CELLS = None
BET_GRID_CELLS = None
ENT_GRID_CELLS = None
AC3G_GRID_CELLS = None
C3G_GRID_CELLS = None
C4_GRID_CELLS = None

# Load each PFT's grid cells from file
for pft_short, pft_region in PFT_NAMES.items():
    pft_file = Path(PFT_GRID_CELLS_PATH) / f"{pft_short}_grid_cells.txt"
    
    if pft_file.exists():
        try:
            # Load lat/lon data
            data = np.loadtxt(pft_file, skiprows=1)
            
            if data.ndim == 2 and len(data) > 0:
                grid_cells = {
                    'lats': data[:, 0],
                    'lons': data[:, 1],
                    'values': None  # We don't store the coverage values from file
                }
                
                # Assign to appropriate variable
                if pft_short == 'bdt':
                    BDT_GRID_CELLS = grid_cells
                elif pft_short == 'bet':
                    BET_GRID_CELLS = grid_cells
                elif pft_short == 'ent':
                    ENT_GRID_CELLS = grid_cells
                elif pft_short == 'ac3g':
                    AC3G_GRID_CELLS = grid_cells
                elif pft_short == 'c3g':
                    C3G_GRID_CELLS = grid_cells
                elif pft_short == 'c4':
                    C4_GRID_CELLS = grid_cells
                
                # Add to REGIONS
                REGIONS[pft_region] = {'grid_cells': grid_cells}
                
                print(f"  Loaded {pft_region} grid cells from {pft_file}")
                print(f"    {len(grid_cells['lats'])} grid cells")
                print(f"    Lat range: {grid_cells['lats'].min():.2f} to {grid_cells['lats'].max():.2f}")
                print(f"    Lon range: {grid_cells['lons'].min():.2f} to {grid_cells['lons'].max():.2f}")
            else:
                print(f"  WARNING: Invalid data format in {pft_file}")
        except Exception as e:
            print(f"  ERROR reading {pft_file}: {str(e)}")
    else:
        print(f"  WARNING: {pft_file} not found")
        print(f"           Run extract_pft_dominated_pixels.py first to generate PFT grid cell files")

# Check if any files were found
files = []

if files:
    print(f"Reading FATES_NOCOMP_PATCHAREA_PF from {len(files)} files...")
    ds = xr.open_mfdataset(files[:1], combine='by_coords')  # Just use first file for identification
    
    if 'FATES_NOCOMP_PATCHAREA_PF' in ds:
        # Get the 6th PFT (index 5)
        pft_6 = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=5)
        
        # Take time mean
        pft_6_mean = pft_6.mean(dim='time')
        
        # Flatten and find top 20 grid cells
        pft_flat = pft_6_mean.values.flatten()
        lat_flat = np.repeat(ds['lat'].values, len(ds['lon']))
        lon_flat = np.tile(ds['lon'].values, len(ds['lat']))
        
        # Sort and get top 20
        valid_mask = ~np.isnan(pft_flat)
        pft_valid = pft_flat[valid_mask]
        lat_valid = lat_flat[valid_mask]
        lon_valid = lon_flat[valid_mask]
        
        top_indices = np.argsort(pft_valid)[-20:]
        
        BDT_GRID_CELLS = {
            'lats': lat_valid[top_indices],
            'lons': lon_valid[top_indices],
            'values': pft_valid[top_indices]
        }
        
        print(f"  Found top 20 BDT grid cells:")
        print(f"  Lat range: {BDT_GRID_CELLS['lats'].min():.2f} to {BDT_GRID_CELLS['lats'].max():.2f}")
        print(f"  Lon range: {BDT_GRID_CELLS['lons'].min():.2f} to {BDT_GRID_CELLS['lons'].max():.2f}")
        print(f"  PFT 6 coverage range: {BDT_GRID_CELLS['values'].min():.4f} to {BDT_GRID_CELLS['values'].max():.4f}")
        
        # Save BDT grid cells to file for use in analyze_csv_results.py
        bdt_file = Path(OUTPUT_PATH) / "bdt_grid_cells.txt"
        bdt_array = np.column_stack([BDT_GRID_CELLS['lats'], BDT_GRID_CELLS['lons']])
        np.savetxt(bdt_file, bdt_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved BDT grid cells to: {bdt_file}")
        
        # Create map showing BDT locations and full PFT 6 distribution
        print("\n  Creating map of BDT grid cells...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Left subplot: Full PFT 6 distribution
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot full PFT 6 coverage as 2D map
        pft_plot = pft_6_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                                    cmap='YlGn', add_colorbar=True,
                                    cbar_kwargs={'label': 'PFT 6 (BDT) Coverage', 'shrink': 0.6})
        ax1.set_title('Full Spatial Distribution of PFT 6 (Broadleaf Deciduous Trees)', 
                     fontsize=12, fontweight='bold')
        
        # Right subplot: Top 20 grid cells
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot the top 20 PFT 6 coverage locations
        scatter = ax2.scatter(BDT_GRID_CELLS['lons'], BDT_GRID_CELLS['lats'], 
                           c=BDT_GRID_CELLS['values'], cmap='YlGn', 
                           s=200, marker='s', edgecolors='black', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=6)
        
        plt.colorbar(scatter, ax=ax2, label='PFT 6 (BDT) Coverage', shrink=0.6)
        ax2.set_title('Top 20 Grid Cells for Broadleaf Deciduous Trees (PFT 6)', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        map_file = f"{OUTPUT_PATH}/bdt_gridcells_map.png"
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  Saved map: {map_file}")
        plt.close(fig)
        
        # Add BDTs to REGIONS with grid cell list
        REGIONS['BDTs'] = {'grid_cells': BDT_GRID_CELLS}
        
        # Process BETs (PFT 1, index 0) - Broadleaf Evergreen Trees
        print("\n" + "="*60)
        print("Processing BET (Broadleaf Evergreen Trees) grid cells...")
        pft_1 = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=0)
        
        # Take time mean
        pft_1_mean = pft_1.mean(dim='time')
        
        # Flatten and find top 20 grid cells
        pft_flat_1 = pft_1_mean.values.flatten()
        lat_flat_1 = np.repeat(ds['lat'].values, len(ds['lon']))
        lon_flat_1 = np.tile(ds['lon'].values, len(ds['lat']))
        
        # Sort and get top 20
        valid_mask_1 = ~np.isnan(pft_flat_1)
        pft_valid_1 = pft_flat_1[valid_mask_1]
        lat_valid_1 = lat_flat_1[valid_mask_1]
        lon_valid_1 = lon_flat_1[valid_mask_1]
        
        top_indices_1 = np.argsort(pft_valid_1)[-20:]
        
        BET_GRID_CELLS = {
            'lats': lat_valid_1[top_indices_1],
            'lons': lon_valid_1[top_indices_1],
            'values': pft_valid_1[top_indices_1]
        }
        
        print(f"  Found top 20 BET grid cells:")
        print(f"  Lat range: {BET_GRID_CELLS['lats'].min():.2f} to {BET_GRID_CELLS['lats'].max():.2f}")
        print(f"  Lon range: {BET_GRID_CELLS['lons'].min():.2f} to {BET_GRID_CELLS['lons'].max():.2f}")
        print(f"  PFT 1 coverage range: {BET_GRID_CELLS['values'].min():.4f} to {BET_GRID_CELLS['values'].max():.4f}")
        
        # Save BET grid cells to file for use in analyze_csv_results.py
        bet_file = Path(OUTPUT_PATH) / "bet_grid_cells.txt"
        bet_array = np.column_stack([BET_GRID_CELLS['lats'], BET_GRID_CELLS['lons']])
        np.savetxt(bet_file, bet_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved BET grid cells to: {bet_file}")
        
        # Create map showing BET locations and full PFT 1 distribution
        print("\n  Creating map of BET grid cells...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Left subplot: Full PFT 1 distribution
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot full PFT 1 coverage as 2D map
        pft_plot = pft_1_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                                    cmap='Greens', add_colorbar=True,
                                    cbar_kwargs={'label': 'PFT 1 (BET) Coverage', 'shrink': 0.6})
        ax1.set_title('Full Spatial Distribution of PFT 1 (Broadleaf Evergreen Trees)', 
                     fontsize=12, fontweight='bold')
        
        # Right subplot: Top 20 grid cells
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot the top 20 PFT 1 coverage locations
        scatter = ax2.scatter(BET_GRID_CELLS['lons'], BET_GRID_CELLS['lats'], 
                           c=BET_GRID_CELLS['values'], cmap='Greens', 
                           s=200, marker='s', edgecolors='black', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=6)
        
        plt.colorbar(scatter, ax=ax2, label='PFT 1 (BET) Coverage', shrink=0.6)
        ax2.set_title('Top 20 Grid Cells for Broadleaf Evergreen Trees (PFT 1)', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        map_file = f"{OUTPUT_PATH}/bet_gridcells_map.png"
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  Saved map: {map_file}")
        plt.close(fig)
        
        # Add BETs to REGIONS with grid cell list
        REGIONS['BETs'] = {'grid_cells': BET_GRID_CELLS}
        
        # Process ENTs (PFT 2, index 1) - Evergreen Needleleaf Trees
        print("\n" + "="*60)
        print("Processing ENT (Evergreen Needleleaf Trees) grid cells...")
        pft_2 = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=1)
        
        # Take time mean
        pft_2_mean = pft_2.mean(dim='time')
        
        # Flatten and find top 20 grid cells
        pft_flat_2 = pft_2_mean.values.flatten()
        lat_flat_2 = np.repeat(ds['lat'].values, len(ds['lon']))
        lon_flat_2 = np.tile(ds['lon'].values, len(ds['lat']))
        
        # Sort and get top 20
        valid_mask_2 = ~np.isnan(pft_flat_2)
        pft_valid_2 = pft_flat_2[valid_mask_2]
        lat_valid_2 = lat_flat_2[valid_mask_2]
        lon_valid_2 = lon_flat_2[valid_mask_2]
        
        top_indices_2 = np.argsort(pft_valid_2)[-20:]
        
        ENT_GRID_CELLS = {
            'lats': lat_valid_2[top_indices_2],
            'lons': lon_valid_2[top_indices_2],
            'values': pft_valid_2[top_indices_2]
        }
        
        print(f"  Found top 20 ENT grid cells:")
        print(f"  Lat range: {ENT_GRID_CELLS['lats'].min():.2f} to {ENT_GRID_CELLS['lats'].max():.2f}")
        print(f"  Lon range: {ENT_GRID_CELLS['lons'].min():.2f} to {ENT_GRID_CELLS['lons'].max():.2f}")
        print(f"  PFT 2 coverage range: {ENT_GRID_CELLS['values'].min():.4f} to {ENT_GRID_CELLS['values'].max():.4f}")
        
        # Save ENT grid cells to file for use in analyze_csv_results.py
        ent_file = Path(OUTPUT_PATH) / "ent_grid_cells.txt"
        ent_array = np.column_stack([ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']])
        np.savetxt(ent_file, ent_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved ENT grid cells to: {ent_file}")
        
        # Create map showing ENT locations and full PFT 2 distribution
        print("\n  Creating map of ENT grid cells...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Left subplot: Full PFT 2 distribution
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot full PFT 2 coverage as 2D map
        pft_plot = pft_2_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                                    cmap='BuGn', add_colorbar=True,
                                    cbar_kwargs={'label': 'PFT 2 (ENT) Coverage', 'shrink': 0.6})
        ax1.set_title('Full Spatial Distribution of PFT 2 (Evergreen Needleleaf Trees)', 
                     fontsize=12, fontweight='bold')
        
        # Right subplot: Top 20 grid cells
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.3)
        
        # Plot the top 20 PFT 2 coverage locations
        scatter = ax2.scatter(ENT_GRID_CELLS['lons'], ENT_GRID_CELLS['lats'], 
                           c=ENT_GRID_CELLS['values'], cmap='BuGn', 
                           s=200, marker='s', edgecolors='black', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=6)
        
        plt.colorbar(scatter, ax=ax2, label='PFT 2 (ENT) Coverage', shrink=0.6)
        ax2.set_title('Top 20 Grid Cells for Evergreen Needleleaf Trees (PFT 2)', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        map_file = f"{OUTPUT_PATH}/ent_gridcells_map.png"
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  Saved map: {map_file}")
        plt.close(fig)
        
        # Add ENTs to REGIONS with grid cell list
        REGIONS['ENTs'] = {'grid_cells': ENT_GRID_CELLS}
        
        # Process AC3G (PFT 12, index 11) - Arctic C3 Grass
        print("\n" + "="*60)
        print("Processing AC3G (Arctic C3 Grass) grid cells...")
        pft_12 = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=11)
        pft_12_mean = pft_12.mean(dim='time')
        pft_flat_12 = pft_12_mean.values.flatten()
        lat_flat_12 = np.repeat(ds['lat'].values, len(ds['lon']))
        lon_flat_12 = np.tile(ds['lon'].values, len(ds['lat']))
        valid_mask_12 = ~np.isnan(pft_flat_12)
        pft_valid_12 = pft_flat_12[valid_mask_12]
        lat_valid_12 = lat_flat_12[valid_mask_12]
        lon_valid_12 = lon_flat_12[valid_mask_12]
        top_indices_12 = np.argsort(pft_valid_12)[-20:]
        
        AC3G_GRID_CELLS = {
            'lats': lat_valid_12[top_indices_12],
            'lons': lon_valid_12[top_indices_12],
            'values': pft_valid_12[top_indices_12]
        }
        
        print(f"  Found top 20 AC3G grid cells:")
        print(f"  Lat range: {AC3G_GRID_CELLS['lats'].min():.2f} to {AC3G_GRID_CELLS['lats'].max():.2f}")
        print(f"  Lon range: {AC3G_GRID_CELLS['lons'].min():.2f} to {AC3G_GRID_CELLS['lons'].max():.2f}")
        print(f"  PFT 12 coverage range: {AC3G_GRID_CELLS['values'].min():.4f} to {AC3G_GRID_CELLS['values'].max():.4f}")
        
        ac3g_file = Path(OUTPUT_PATH) / "ac3g_grid_cells.txt"
        ac3g_array = np.column_stack([AC3G_GRID_CELLS['lats'], AC3G_GRID_CELLS['lons']])
        np.savetxt(ac3g_file, ac3g_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved AC3G grid cells to: {ac3g_file}")
        
        print("\n  Creating map of AC3G grid cells...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        pft_plot = pft_12_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                                    cmap='YlOrBr', add_colorbar=True,
                                    cbar_kwargs={'label': 'PFT 12 (AC3G) Coverage', 'shrink': 0.6})
        ax1.set_title('Full Spatial Distribution of PFT 12 (Arctic C3 Grass)', 
                     fontsize=12, fontweight='bold')
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.3)
        scatter = ax2.scatter(AC3G_GRID_CELLS['lons'], AC3G_GRID_CELLS['lats'], 
                           c=AC3G_GRID_CELLS['values'], cmap='YlOrBr', 
                           s=200, marker='s', edgecolors='black', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=6)
        plt.colorbar(scatter, ax=ax2, label='PFT 12 (AC3G) Coverage', shrink=0.6)
        ax2.set_title('Top 20 Grid Cells for Arctic C3 Grass (PFT 12)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        map_file = f"{OUTPUT_PATH}/ac3g_gridcells_map.png"
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  Saved map: {map_file}")
        plt.close(fig)
        REGIONS['AC3G'] = {'grid_cells': AC3G_GRID_CELLS}
        
        # Process C3G (PFT 13, index 12) - C3 Grass
        print("\n" + "="*60)
        print("Processing C3G (C3 Grass) grid cells...")
        pft_13 = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=12)
        pft_13_mean = pft_13.mean(dim='time')
        pft_flat_13 = pft_13_mean.values.flatten()
        lat_flat_13 = np.repeat(ds['lat'].values, len(ds['lon']))
        lon_flat_13 = np.tile(ds['lon'].values, len(ds['lat']))
        valid_mask_13 = ~np.isnan(pft_flat_13)
        pft_valid_13 = pft_flat_13[valid_mask_13]
        lat_valid_13 = lat_flat_13[valid_mask_13]
        lon_valid_13 = lon_flat_13[valid_mask_13]
        top_indices_13 = np.argsort(pft_valid_13)[-20:]
        
        C3G_GRID_CELLS = {
            'lats': lat_valid_13[top_indices_13],
            'lons': lon_valid_13[top_indices_13],
            'values': pft_valid_13[top_indices_13]
        }
        
        print(f"  Found top 20 C3G grid cells:")
        print(f"  Lat range: {C3G_GRID_CELLS['lats'].min():.2f} to {C3G_GRID_CELLS['lats'].max():.2f}")
        print(f"  Lon range: {C3G_GRID_CELLS['lons'].min():.2f} to {C3G_GRID_CELLS['lons'].max():.2f}")
        print(f"  PFT 13 coverage range: {C3G_GRID_CELLS['values'].min():.4f} to {C3G_GRID_CELLS['values'].max():.4f}")
        
        c3g_file = Path(OUTPUT_PATH) / "c3g_grid_cells.txt"
        c3g_array = np.column_stack([C3G_GRID_CELLS['lats'], C3G_GRID_CELLS['lons']])
        np.savetxt(c3g_file, c3g_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved C3G grid cells to: {c3g_file}")
        
        print("\n  Creating map of C3G grid cells...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        pft_plot = pft_13_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                                    cmap='RdYlGn', add_colorbar=True,
                                    cbar_kwargs={'label': 'PFT 13 (C3G) Coverage', 'shrink': 0.6})
        ax1.set_title('Full Spatial Distribution of PFT 13 (C3 Grass)', 
                     fontsize=12, fontweight='bold')
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.3)
        scatter = ax2.scatter(C3G_GRID_CELLS['lons'], C3G_GRID_CELLS['lats'], 
                           c=C3G_GRID_CELLS['values'], cmap='RdYlGn', 
                           s=200, marker='s', edgecolors='black', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=6)
        plt.colorbar(scatter, ax=ax2, label='PFT 13 (C3G) Coverage', shrink=0.6)
        ax2.set_title('Top 20 Grid Cells for C3 Grass (PFT 13)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        map_file = f"{OUTPUT_PATH}/c3g_gridcells_map.png"
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  Saved map: {map_file}")
        plt.close(fig)
        REGIONS['C3G'] = {'grid_cells': C3G_GRID_CELLS}
        
        # Process C4 (PFT 14, index 13) - C4 Grass
        print("\n" + "="*60)
        print("Processing C4 (C4 Grass) grid cells...")
        pft_14 = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=13)
        pft_14_mean = pft_14.mean(dim='time')
        pft_flat_14 = pft_14_mean.values.flatten()
        lat_flat_14 = np.repeat(ds['lat'].values, len(ds['lon']))
        lon_flat_14 = np.tile(ds['lon'].values, len(ds['lat']))
        valid_mask_14 = ~np.isnan(pft_flat_14)
        pft_valid_14 = pft_flat_14[valid_mask_14]
        lat_valid_14 = lat_flat_14[valid_mask_14]
        lon_valid_14 = lon_flat_14[valid_mask_14]
        top_indices_14 = np.argsort(pft_valid_14)[-20:]
        
        C4_GRID_CELLS = {
            'lats': lat_valid_14[top_indices_14],
            'lons': lon_valid_14[top_indices_14],
            'values': pft_valid_14[top_indices_14]
        }
        
        print(f"  Found top 20 C4 grid cells:")
        print(f"  Lat range: {C4_GRID_CELLS['lats'].min():.2f} to {C4_GRID_CELLS['lats'].max():.2f}")
        print(f"  Lon range: {C4_GRID_CELLS['lons'].min():.2f} to {C4_GRID_CELLS['lons'].max():.2f}")
        print(f"  PFT 14 coverage range: {C4_GRID_CELLS['values'].min():.4f} to {C4_GRID_CELLS['values'].max():.4f}")
        
        c4_file = Path(OUTPUT_PATH) / "c4_grid_cells.txt"
        c4_array = np.column_stack([C4_GRID_CELLS['lats'], C4_GRID_CELLS['lons']])
        np.savetxt(c4_file, c4_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved C4 grid cells to: {c4_file}")
        
        print("\n  Creating map of C4 grid cells...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.gridlines(draw_labels=True, alpha=0.3)
        pft_plot = pft_14_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                                    cmap='Oranges', add_colorbar=True,
                                    cbar_kwargs={'label': 'PFT 14 (C4) Coverage', 'shrink': 0.6})
        ax1.set_title('Full Spatial Distribution of PFT 14 (C4 Grass)', 
                     fontsize=12, fontweight='bold')
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.gridlines(draw_labels=True, alpha=0.3)
        scatter = ax2.scatter(C4_GRID_CELLS['lons'], C4_GRID_CELLS['lats'], 
                           c=C4_GRID_CELLS['values'], cmap='Oranges', 
                           s=200, marker='s', edgecolors='black', linewidth=2,
                           transform=ccrs.PlateCarree(), zorder=6)
        plt.colorbar(scatter, ax=ax2, label='PFT 14 (C4) Coverage', shrink=0.6)
        ax2.set_title('Top 20 Grid Cells for C4 Grass (PFT 14)', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        map_file = f"{OUTPUT_PATH}/c4_gridcells_map.png"
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"  Saved map: {map_file}")
        plt.close(fig)
        REGIONS['C4'] = {'grid_cells': C4_GRID_CELLS}
        
        ds.close()
    else:
        print("  WARNING: FATES_NOCOMP_PATCHAREA_PF not found in dataset")
        BDT_GRID_CELLS = None
        BET_GRID_CELLS = None
        ENT_GRID_CELLS = None
        AC3G_GRID_CELLS = None
        C3G_GRID_CELLS = None
        C4_GRID_CELLS = None
else:
    print("  WARNING: No files found for PFT identification")
    BDT_GRID_CELLS = None
    BET_GRID_CELLS = None
    ENT_GRID_CELLS = None
    AC3G_GRID_CELLS = None
    C3G_GRID_CELLS = None
    C4_GRID_CELLS = None

# Update results dictionaries to include BDTs, BETs, and ENTs
if 'BDTs' in REGIONS:
    results_ppe_v1['BDTs'] = {var: {} for var in VARIABLES}
    results_ppe_2000_v1['BDTs'] = {var: {} for var in VARIABLES}
    results_ppe_v1_summer_lai['BDTs'] = {}
    results_ppe_2000_v1_summer_lai['BDTs'] = {}
if 'BETs' in REGIONS:
    results_ppe_v1['BETs'] = {var: {} for var in VARIABLES}
    results_ppe_2000_v1['BETs'] = {var: {} for var in VARIABLES}
    results_ppe_v1_summer_lai['BETs'] = {}
    results_ppe_2000_v1_summer_lai['BETs'] = {}
if 'ENTs' in REGIONS:
    results_ppe_v1['ENTs'] = {var: {} for var in VARIABLES}
    results_ppe_2000_v1['ENTs'] = {var: {} for var in VARIABLES}
    results_ppe_v1_summer_lai['ENTs'] = {}
    results_ppe_2000_v1_summer_lai['ENTs'] = {}
if 'AC3G' in REGIONS:
    results_ppe_v1['AC3G'] = {var: {} for var in VARIABLES}
    results_ppe_2000_v1['AC3G'] = {var: {} for var in VARIABLES}
    results_ppe_v1_summer_lai['AC3G'] = {}
    results_ppe_2000_v1_summer_lai['AC3G'] = {}
if 'C3G' in REGIONS:
    results_ppe_v1['C3G'] = {var: {} for var in VARIABLES}
    results_ppe_2000_v1['C3G'] = {var: {} for var in VARIABLES}
    results_ppe_v1_summer_lai['C3G'] = {}
    results_ppe_2000_v1_summer_lai['C3G'] = {}
if 'C4' in REGIONS:
    results_ppe_v1['C4'] = {var: {} for var in VARIABLES}
    results_ppe_2000_v1['C4'] = {var: {} for var in VARIABLES}
    results_ppe_v1_summer_lai['C4'] = {}
    results_ppe_2000_v1_summer_lai['C4'] = {}

print("\n" + "="*60)
print("Processing model outputs for PPE_V1...\n")

# Loop through each ensemble member for first ensemble
for n in ENSEMBLE_MEMBERS:
    ensemble_id = f"n{n:02d}"
    dir_path = f"{BASE_PATH}/{casename}{ensemble_id}/lnd/hist/"
    
    print(f"Processing ensemble member {ensemble_id}...")
    print(f"  Looking in: {dir_path}")
    
    # Build file pattern for years 05-10, all months
    file_patterns = []
    for year in YEARS:
        pattern = f"{dir_path}{casename}{ensemble_id}.clm2.h0a.00{year:02d}-*.nc"
        file_patterns.append(pattern)
    print(f"  File pattern: {casename}{ensemble_id}.clm2.h0a.00{min(YEARS):02d}-*.nc")
    
    # Collect all matching files
    all_files = []
    for pattern in file_patterns:
        files = sorted(glob(pattern))
        all_files.extend(files)
    
    if not all_files:
        print(f"  WARNING: No files found for {ensemble_id}")
        continue
    
    print(f"  Found {len(all_files)} files")
    
    try:
        # Open all files as a single dataset
        ds = xr.open_mfdataset(all_files, combine='by_coords')
        
        # Extract and compute mean for each variable and region
        for var in VARIABLES:
            if var in ds:
                print(f"  {var}:")
                # Compute mean for each region
                for region_name, region_bounds in REGIONS.items():
                    if 'grid_cells' in region_bounds:
                        # For BDTs and BETs, select specific grid cells
                        grid_cells = region_bounds['grid_cells']
                        values = []
                        for lat, lon in zip(grid_cells['lats'], grid_cells['lons']):
                            val = ds[var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                            values.append(float(val))
                        var_mean = np.mean(values)
                        results_ppe_v1[region_name][var][n] = float(var_mean)
                        print(f"    {region_name}: {var_mean:.4f}")
                        
                        # Calculate summer LAI (months 7, 8, 9) for FATES_LAI
                        if var == 'FATES_LAI':
                            summer_values = []
                            for lat, lon in zip(grid_cells['lats'], grid_cells['lons']):
                                ds_point = ds[var].sel(lat=lat, lon=lon, method='nearest')
                                # Select summer months (7, 8, 9)
                                ds_summer = ds_point.sel(time=ds_point.time.dt.month.isin([7, 8, 9]))
                                summer_val = ds_summer.mean(skipna=True).values
                                summer_values.append(float(summer_val))
                            summer_mean = np.mean(summer_values)
                            results_ppe_v1_summer_lai[region_name][n] = float(summer_mean)
                            print(f"    {region_name} (summer LAI): {summer_mean:.4f}")
                    else:
                        # Subset by lat and lon
                        ds_subset = ds[var].where(
                            (ds['lat'] >= region_bounds['lat_min']) &
                            (ds['lat'] <= region_bounds['lat_max']) &
                            (ds['lon'] >= region_bounds['lon_min']) &
                            (ds['lon'] <= region_bounds['lon_max']),
                            drop=False
                        )
                        # Compute temporal and spatial mean (excluding NaN)
                        var_mean = ds_subset.mean(skipna=True).values
                        results_ppe_v1[region_name][var][n] = float(var_mean)
                        print(f"    {region_name}: {var_mean:.4f}")
                        
                        # Calculate summer LAI (months 7, 8, 9) for FATES_LAI
                        if var == 'FATES_LAI':
                            # Select summer months (7, 8, 9)
                            ds_summer = ds_subset.sel(time=ds_subset.time.dt.month.isin([7, 8, 9]))
                            summer_mean = ds_summer.mean(skipna=True).values
                            results_ppe_v1_summer_lai[region_name][n] = float(summer_mean)
                            print(f"    {region_name} (summer LAI): {summer_mean:.4f}")
            else:
                print(f"  WARNING: Variable {var} not found in dataset")
                for region_name in REGIONS.keys():
                    results_ppe_v1[region_name][var][n] = np.nan
                    if var == 'FATES_LAI':
                        results_ppe_v1_summer_lai[region_name][n] = np.nan
        
        ds.close()
        
    except Exception as e:
        print(f"  ERROR processing {ensemble_id}: {str(e)}")
        for var in VARIABLES:
            for region_name in REGIONS.keys():
                results_ppe_v1[region_name][var][n] = np.nan

print("\n" + "="*60)
if ens_n != 2:
    print("Processing model outputs for PPE_2000_V1...\n")
else:
    print("Skipping PPE_2000_V1 processing (ens_n==2)...\n")

# Loop through each ensemble member for second ensemble
if ens_n != 2:
    for n in ENSEMBLE_MEMBERS:
        ensemble_id = f"n{n:02d}"
        dir_path = f"{BASE_PATH}/noresm_beta07_crujra_ppe_2000_v1_{ensemble_id}/lnd/hist/"
        
        print(f"Processing ensemble member {ensemble_id}...")
        print(f"  Looking in: {dir_path}")
        
        # Build file pattern for years 05-10, all months
        file_patterns = []
        for year in YEARS:
            pattern = f"{dir_path}noresm_beta07_crujra_ppe_2000_v1_{ensemble_id}.clm2.h0a.00{year:02d}-*.nc"
            file_patterns.append(pattern)
        print(f"  File pattern: noresm_beta07_crujra_ppe_2000_v1_{ensemble_id}.clm2.h0a.00{min(YEARS):02d}-*.nc")
        
        # Collect all matching files
        all_files = []
        for pattern in file_patterns:
            files = sorted(glob(pattern))
            all_files.extend(files)
        
        if not all_files:
            print(f"  WARNING: No files found for {ensemble_id}")
            continue
        
        print(f"  Found {len(all_files)} files")
        
        try:
            # Open all files as a single dataset
            ds = xr.open_mfdataset(all_files, combine='by_coords')
            
            # Extract and compute mean for each variable and region
            for var in VARIABLES:
                if var in ds:
                    print(f"  {var}:")
                    # Compute mean for each region
                    for region_name, region_bounds in REGIONS.items():
                        if 'grid_cells' in region_bounds:
                            # For BDTs and BETs, select specific grid cells
                            grid_cells = region_bounds['grid_cells']
                            values = []
                            for lat, lon in zip(grid_cells['lats'], grid_cells['lons']):
                                val = ds[var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                                values.append(float(val))
                            var_mean = np.mean(values)
                            results_ppe_2000_v1[region_name][var][n] = float(var_mean)
                            print(f"    {region_name}: {var_mean:.4f}")
                            
                            # Calculate summer LAI (months 7, 8, 9) for FATES_LAI
                            if var == 'FATES_LAI':
                                summer_values = []
                                for lat, lon in zip(grid_cells['lats'], grid_cells['lons']):
                                    ds_point = ds[var].sel(lat=lat, lon=lon, method='nearest')
                                    # Select summer months (7, 8, 9)
                                    ds_summer = ds_point.sel(time=ds_point.time.dt.month.isin([7, 8, 9]))
                                    summer_val = ds_summer.mean(skipna=True).values
                                    summer_values.append(float(summer_val))
                                summer_mean = np.mean(summer_values)
                                results_ppe_2000_v1_summer_lai[region_name][n] = float(summer_mean)
                                print(f"    {region_name} (summer LAI): {summer_mean:.4f}")
                        else:
                            # Subset by lat and lon
                            ds_subset = ds[var].where(
                                (ds['lat'] >= region_bounds['lat_min']) &
                                (ds['lat'] <= region_bounds['lat_max']) &
                                (ds['lon'] >= region_bounds['lon_min']) &
                                (ds['lon'] <= region_bounds['lon_max']),
                                drop=False
                            )
                            # Compute temporal and spatial mean (excluding NaN)
                            var_mean = ds_subset.mean(skipna=True).values
                            results_ppe_2000_v1[region_name][var][n] = float(var_mean)
                            print(f"    {region_name}: {var_mean:.4f}")
                            
                            # Calculate summer LAI (months 7, 8, 9) for FATES_LAI
                            if var == 'FATES_LAI':
                                # Select summer months (7, 8, 9)
                                ds_summer = ds_subset.sel(time=ds_subset.time.dt.month.isin([7, 8, 9]))
                                summer_mean = ds_summer.mean(skipna=True).values
                                results_ppe_2000_v1_summer_lai[region_name][n] = float(summer_mean)
                                print(f"    {region_name} (summer LAI): {summer_mean:.4f}")
                else:
                    print(f"  WARNING: Variable {var} not found in dataset")
                    for region_name in REGIONS.keys():
                        results_ppe_2000_v1[region_name][var][n] = np.nan
                        if var == 'FATES_LAI':
                            results_ppe_2000_v1_summer_lai[region_name][n] = np.nan
            
            ds.close()
            
        except Exception as e:
            print(f"  ERROR processing {ensemble_id}: {str(e)}")
            for var in VARIABLES:
                for region_name in REGIONS.keys():
                    results_ppe_2000_v1[region_name][var][n] = np.nan
                    if var == 'FATES_LAI':
                        results_ppe_2000_v1_summer_lai[region_name][n] = np.nan

# Create output directory
output_dir = Path(outputdir)
output_dir.mkdir(exist_ok=True)

# Create CSV subdirectory
csv_dir = output_dir / "csv_data"
csv_dir.mkdir(exist_ok=True)

# Save ensemble vs variable data for each region
for region_name in REGIONS.keys():
    # Create DataFrame with ensemble members as rows
    ensemble_nums = sorted(results_ppe_v1[region_name][VARIABLES[0]].keys())
    
    csv_data = {'ensemble_member': ensemble_nums}
    
    # Add parameter values (use PFT-specific values for PFT regions)
    pft_idx = REGION_PFT_MAP.get(region_name, None)
    print(f"\n  Region: {region_name}, PFT index: {pft_idx}")
    for param in PARAMETERS:
        param_list = []
        
        # Handle special non-PFT parameters
        if param in ['fates_fire_cg_strikes', 'fates_landuse_grazing_rate']:
            # These are not PFT-indexed, just use the value directly
            for n in ensemble_nums:
                param_val = param_values[param].get(n, np.nan)
                param_list.append(param_val)
            csv_data[param] = param_list
            print(f"    {param} values (single/scalar): {param_list[:5]}...")  # Show first 5
        elif pft_idx is not None:
            # Use PFT-specific value for this region
            for n in ensemble_nums:
                param_array = param_values[param].get(n, np.full(14, np.nan))
                if isinstance(param_array, np.ndarray):
                    param_list.append(param_array[pft_idx])
                else:
                    param_list.append(np.nan)
            csv_data[param] = param_list
            print(f"    {param} values for PFT {pft_idx+1}: {param_list[:5]}...")  # Show first 5
        else:
            # For global regions, use first PFT (or could use average)
            for n in ensemble_nums:
                param_array = param_values[param].get(n, np.full(14, np.nan))
                if isinstance(param_array, np.ndarray):
                    param_list.append(param_array[0])
                else:
                    param_list.append(np.nan)
            csv_data[param] = param_list
            print(f"    {param} values (PFT 1): {param_list[:5]}...")  # Show first 5
    
    # Add PPE_V1 variables
    for var in VARIABLES:
        csv_data[f'{var}_PPE_V1'] = [results_ppe_v1[region_name][var].get(n, np.nan) for n in ensemble_nums]
    
    # Add PPE_2000_V1 variables
    if ens_n != 2:
        for var in VARIABLES:
            csv_data[f'{var}_PPE_2000_V1'] = [results_ppe_2000_v1[region_name][var].get(n, np.nan) for n in ensemble_nums]
    
    # Add summer LAI values
    csv_data['fates_lai_summer_PPE_V1'] = [results_ppe_v1_summer_lai[region_name].get(n, np.nan) for n in ensemble_nums]
    if ens_n != 2:
        csv_data['fates_lai_summer_PPE_2000_V1'] = [results_ppe_2000_v1_summer_lai[region_name].get(n, np.nan) for n in ensemble_nums]
    
    # Add summer LAI ratio
    if ens_n != 2:
        summer_lai_ratios = []
        for n in ensemble_nums:
            v1 = results_ppe_v1_summer_lai[region_name].get(n, np.nan)
            v2 = results_ppe_2000_v1_summer_lai[region_name].get(n, np.nan)
            if not np.isnan(v1) and not np.isnan(v2) and v1 != 0:
                summer_lai_ratios.append(v2 / v1)
            else:
                summer_lai_ratios.append(np.nan)
        csv_data['fates_lai_summer_ratio'] = summer_lai_ratios
    
    # Add ratios
    if ens_n != 2:
        for var in VARIABLES:
            ratios = []
            for n in ensemble_nums:
                v1 = results_ppe_v1[region_name][var].get(n, np.nan)
                v2 = results_ppe_2000_v1[region_name][var].get(n, np.nan)
                print(var, 'v1',v1)
                print(var, 'v2',v2)
                if not np.isnan(v1) and not np.isnan(v2) and v1 != 0:
                    ratios.append(v2 / v1)
                else:
                    ratios.append(np.nan)
            csv_data[f'{var}_ratio'] = ratios
    
    # Create and save DataFrame
    df = pd.DataFrame(csv_data)
    year_range_str = f"years{min(YEARS):02d}-{max(YEARS):02d}"
    csv_file = csv_dir / f"ensemble_data_{year_range_str}_{region_name}.csv"
    df.to_csv(csv_file, index=False)
    print(f"  Saved data to: {csv_file}")

print("\n" + "="*60)
print("Creating plots...")

# Loop over each region to create separate plots
if ens_n != 2:
  for region_name in REGIONS.keys():
      print(f"\nCreating plots for region: {region_name}")
      
      # Create figure with subplots for each variable
      fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(10, 16))
      fig.suptitle(f'FATES Variables vs Ensemble Members - {region_name.replace("_", " ").title()} (Years 05-10)', 
                   fontsize=14, fontweight='bold')

      for idx, var in enumerate(VARIABLES):
          ax = axes[idx]
          
          # Extract ensemble numbers and values for both ensembles
          ensemble_nums = sorted(results_ppe_v1[region_name][var].keys())
          values_v1 = [results_ppe_v1[region_name][var][n] for n in ensemble_nums]
          values_2000 = [results_ppe_2000_v1[region_name][var].get(n, np.nan) for n in ensemble_nums]
          
          # Plot both ensembles
          ax.plot(ensemble_nums, values_v1, 'o-', linewidth=2, markersize=8, label='PPE_V1')
          ax.plot(ensemble_nums, values_2000, 's-', linewidth=2, markersize=8, label='PPE_2000_V1')
          ax.set_xlabel('Ensemble Member', fontsize=11)
          ax.set_ylabel(var, fontsize=11)
          ax.legend()
          ax.grid(True, alpha=0.3)
          ax.set_xticks(ensemble_nums)

      plt.tight_layout()

      # Save figure
      output_file = output_dir / f"ensemble_analysis_years05-10_{region_name}.png"
      fig.savefig(output_file, dpi=300, bbox_inches='tight')
      print(f"  Figure saved to: {output_file}")
      plt.close(fig)



# Create comprehensive parameter vs FATES variable plots for each region
if ens_n != 2:
    print("\n" + "="*60)
    print("Creating parameter vs FATES variable plots...")

    for region_name in REGIONS.keys():
        print(f"\nCreating parameter plots for region: {region_name}")
        
        # Create a 4x3 grid of subplots (4 parameters x 3 variables)
        fig3, axes3 = plt.subplots(len(PARAMETERS), len(VARIABLES), figsize=(20, 16))
        fig3.suptitle(f'FATES Variables vs Parameters - {region_name.replace("_", " ").title()} (Years 05-10)', 
                      fontsize=16, fontweight='bold', y=0.995)

        # Store scatter objects for colorbar
        scatter_objects = []

        # Iterate through parameters and variables
        for param_idx, param in enumerate(PARAMETERS):
            for var_idx, var in enumerate(VARIABLES):
                ax = axes3[param_idx, var_idx]
                
                # Collect data points for PPE_V1
                ensemble_nums_v1 = []
                param_vals_v1 = []
                var_vals_v1 = []
                slatop_vals_v1 = []
                
                for n in sorted(results_ppe_v1[region_name][var].keys()):
                    param_val = get_param_value_for_region(param, n, region_name)
                    slatop_val = get_param_value_for_region('fates_leaf_slatop', n, region_name)
                    
                    if (n in param_values[param] and 
                        not np.isnan(results_ppe_v1[region_name][var][n]) and 
                        not np.isnan(param_val) and
                        n in param_values['fates_leaf_slatop'] and
                        not np.isnan(slatop_val)):
                        ensemble_nums_v1.append(n)
                        param_vals_v1.append(param_val)
                        var_vals_v1.append(results_ppe_v1[region_name][var][n])
                        slatop_vals_v1.append(slatop_val)
                
                # Collect data points for PPE_2000_V1
                ensemble_nums_2000 = []
                param_vals_2000 = []
                var_vals_2000 = []
                slatop_vals_2000 = []
                
                for n in sorted(results_ppe_2000_v1[region_name][var].keys()):
                    param_val = get_param_value_for_region(param, n, region_name)
                    slatop_val = get_param_value_for_region('fates_leaf_slatop', n, region_name)
                    
                    if (n in param_values[param] and 
                        not np.isnan(results_ppe_2000_v1[region_name][var][n]) and 
                        not np.isnan(param_val) and
                        n in param_values['fates_leaf_slatop'] and
                        not np.isnan(slatop_val)):
                        ensemble_nums_2000.append(n)
                        param_vals_2000.append(param_val)
                        var_vals_2000.append(results_ppe_2000_v1[region_name][var][n])
                        slatop_vals_2000.append(slatop_val)
                
                if len(ensemble_nums_v1) > 0 or len(ensemble_nums_2000) > 0:
                    # Create scatter plots with different markers
                    if len(ensemble_nums_v1) > 0:
                        scatter = ax.scatter(param_vals_v1, var_vals_v1, s=80, alpha=0.6, 
                                           c=slatop_vals_v1, cmap='viridis', marker='o',
                                           edgecolors='black', linewidth=0.5, label='PPE_V1')
                    if len(ensemble_nums_2000) > 0:
                        scatter2 = ax.scatter(param_vals_2000, var_vals_2000, s=80, alpha=0.6, 
                                            c=slatop_vals_2000, cmap='viridis', marker='s',
                                            edgecolors='black', linewidth=0.5, label='PPE_2000_V1')
                    
                    # Store scatter object for colorbar
                    if param_idx == 0 and var_idx == 0:
                        if len(ensemble_nums_v1) > 0:
                            scatter_objects.append(scatter)
                        elif len(ensemble_nums_2000) > 0:
                            scatter_objects.append(scatter2)
                    
                    # Add legend only in top-right plot
                    if param_idx == 0 and var_idx == len(VARIABLES) - 1:
                        ax.legend(loc='best', fontsize=8)
                    
                    # Labels - x-axis = parameter values, y-axis = variable values
                    # X-axis label shows parameter name for this row
                    ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=9)
                    if var_idx == 0:
                        # Left column: y-axis label shows variable name
                        ax.set_ylabel(var, fontsize=10)
                    
                    # Title for top row shows variable name
                    if param_idx == 0:
                        ax.set_title(var, fontsize=11, fontweight='bold')
                    
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])

        # Adjust layout to make room for colorbar
        plt.tight_layout(rect=[0, 0, 0.95, 1])

        # Add single colorbar on the right side
        if scatter_objects:
            cbar_ax = fig3.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig3.colorbar(scatter_objects[0], cax=cbar_ax)
            cbar.set_label('fates_leaf_slatop', fontsize=12)

        # Save comprehensive figure
        output_file3 = output_dir / f"parameters_vs_FATES_comprehensive_{region_name}.png"
        fig3.savefig(output_file3, dpi=300, bbox_inches='tight')
        print(f"  Comprehensive parameter plot saved to: {output_file3}")
        plt.close(fig3)

# Create ratio plots: PPE_2000_V1 / PPE_V1 vs Parameters for each region
if ens_n != 2:
    print("\n" + "="*60)
    print("Creating ratio plots (PPE_2000_V1 / PPE_V1)...")

    for region_name in REGIONS.keys():
        print(f"\nCreating ratio plots for region: {region_name}")
        
        # Create a 4x3 grid of subplots (4 parameters x 3 variables) for ratios
        fig4, axes4 = plt.subplots(len(PARAMETERS), len(VARIABLES), figsize=(20, 16))
        fig4.suptitle(f'FATES Variable Ratios (PPE_2000_V1 / PPE_V1) vs Parameters - {region_name.replace("_", " ").title()} (Years 05-10)', 
                      fontsize=16, fontweight='bold', y=0.995)

        # Store scatter objects for colorbar
        scatter_objects_ratio = []

        # Iterate through parameters and variables
        for param_idx, param in enumerate(PARAMETERS):
            for var_idx, var in enumerate(VARIABLES):
                ax = axes4[param_idx, var_idx]
                
                # Collect data points
                ensemble_nums = []
                param_vals = []
                ratio_vals = []
                slatop_vals = []
                
                for n in sorted(results_ppe_v1[region_name][var].keys()):
                    param_val = get_param_value_for_region(param, n, region_name)
                    slatop_val = get_param_value_for_region('fates_leaf_slatop', n, region_name)
                    
                    if (n in param_values[param] and 
                        n in results_ppe_2000_v1[region_name][var] and
                        not np.isnan(results_ppe_v1[region_name][var][n]) and 
                        not np.isnan(results_ppe_2000_v1[region_name][var][n]) and
                        results_ppe_v1[region_name][var][n] != 0 and
                        not np.isnan(param_val) and
                        n in param_values['fates_leaf_slatop'] and
                        not np.isnan(slatop_val)):
                        ensemble_nums.append(n)
                        param_vals.append(param_val)
                        ratio = results_ppe_2000_v1[region_name][var][n] / results_ppe_v1[region_name][var][n]
                        ratio_vals.append(ratio)
                        slatop_vals.append(slatop_val)
            
                if len(ensemble_nums) > 0:
                    # Create scatter plot colored by slatop values
                    scatter = ax.scatter(param_vals, ratio_vals, s=80, alpha=0.6, 
                                       c=slatop_vals, cmap='viridis', 
                                       edgecolors='black', linewidth=0.5)
                    
                    # Store scatter object for colorbar
                    if param_idx == 0 and var_idx == 0:
                        scatter_objects_ratio.append(scatter)
                    
                    # Labels - x-axis = parameter values, y-axis = ratio values
                    # X-axis label shows parameter name for this row
                    ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=9)
                    if var_idx == 0:
                        # Left column: y-axis shows ratio for this variable
                        ax.set_ylabel(var + ' Ratio', fontsize=10)
                    
                    # Title for top row shows variable name with "Ratio"
                    if param_idx == 0:
                        ax.set_title(var + ' Ratio', fontsize=11, fontweight='bold')
                    
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])

        # Adjust layout to make room for colorbar
        plt.tight_layout(rect=[0, 0, 0.95, 1])

        # Add single colorbar on the right side
        if scatter_objects_ratio:
            cbar_ax = fig4.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = fig4.colorbar(scatter_objects_ratio[0], cax=cbar_ax)
            cbar.set_label('fates_leaf_slatop', fontsize=12)

        # Save ratio figure
        output_file4 = output_dir / f"parameters_vs_FATES_ratios_{region_name}.png"
        fig4.savefig(output_file4, dpi=300, bbox_inches='tight')
        print(f"  Ratio parameter plot saved to: {output_file4}")
        plt.close(fig4)

# Print correlation summary for each region
print("\n" + "="*60)
print("CORRELATION SUMMARY:")
print("="*60)

for region_name in REGIONS.keys():
    print(f"\n{region_name.upper().replace('_', ' ')}:")
    for param in PARAMETERS:
        print(f"\n  {param}:")
        for var in VARIABLES:
            # Collect data points
            param_vals = []
            var_vals = []
            
            for n in sorted(results_ppe_v1[region_name][var].keys()):
                param_val = get_param_value_for_region(param, n, region_name)
                
                if (n in param_values[param] and 
                    not np.isnan(results_ppe_v1[region_name][var][n]) and 
                    not np.isnan(param_val)):
                    param_vals.append(param_val)
                    var_vals.append(results_ppe_v1[region_name][var][n])
            
            if len(param_vals) > 1:
                correlation = np.corrcoef(param_vals, var_vals)[0, 1]
                print(f"    {var:15s}: r = {correlation:7.4f}")
            else:
                print(f"    {var:15s}: insufficient data")

print("\nAnalysis complete!")

# Print summary statistics for each region
for region_name in REGIONS.keys():
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS FOR {region_name.upper().replace('_', ' ')} - PPE_V1:")
    for var in VARIABLES:
        valid_values = [v for v in results_ppe_v1[region_name][var].values() if not np.isnan(v)]
        if valid_values:
            print(f"\n{var}:")
            print(f"  Mean: {np.mean(valid_values):.4f}")
            print(f"  Std:  {np.std(valid_values):.4f}")
            print(f"  Min:  {np.min(valid_values):.4f}")
            print(f"  Max:  {np.max(valid_values):.4f}")

    if ens_n != 2:
        print("\n" + "="*60)
        print(f"SUMMARY STATISTICS FOR {region_name.upper().replace('_', ' ')} - PPE_2000_V1:")
        for var in VARIABLES:
            valid_values = [v for v in results_ppe_2000_v1[region_name][var].values() if not np.isnan(v)]
            if valid_values:
                print(f"\n{var}:")
                print(f"  Mean: {np.mean(valid_values):.4f}")
                print(f"  Std:  {np.std(valid_values):.4f}")
                print(f"  Min:  {np.min(valid_values):.4f}")
                print(f"  Max:  {np.max(valid_values):.4f}")

        print("\n" + "="*60)
        print(f"SUMMARY STATISTICS FOR {region_name.upper().replace('_', ' ')} - RATIOS (PPE_2000_V1 / PPE_V1):")
        for var in VARIABLES:
            ratios = []
            for n in results_ppe_v1[region_name][var].keys():
                if (n in results_ppe_2000_v1[region_name][var] and 
                    not np.isnan(results_ppe_v1[region_name][var][n]) and 
                    not np.isnan(results_ppe_2000_v1[region_name][var][n]) and
                    results_ppe_v1[region_name][var][n] != 0):
                    ratio = results_ppe_2000_v1[region_name][var][n] / results_ppe_v1[region_name][var][n]
                    ratios.append(ratio)
            
            if ratios:
                print(f"\n{var}:")
                print(f"  Mean: {np.mean(ratios):.4f}")
                print(f"  Std:  {np.std(ratios):.4f}")
                print(f"  Min:  {np.min(ratios):.4f}")
                print(f"  Max:  {np.max(ratios):.4f}")

print(f"\nAll figures saved to: {output_dir}")

# Identify parameter space with high NPP in v1 and low LAI in 2000 ensemble
if ens_n != 2:
    print("\n" + "="*60)
    print("IDENTIFYING PARAMETER SPACE:")
    print("High NPP in PPE_V1 AND Low LAI in PPE_2000_V1")
    print("="*60)

    for region_name in REGIONS.keys():
        print(f"\n{region_name.upper().replace('_', ' ')}:")
        
        # Get all ensemble members with valid data
        ensemble_nums = sorted(results_ppe_v1[region_name]['FATES_NPP'].keys())
        
        # Calculate thresholds
        npp_v1_values = [results_ppe_v1[region_name]['FATES_NPP'][n] for n in ensemble_nums 
                         if not np.isnan(results_ppe_v1[region_name]['FATES_NPP'].get(n, np.nan))]
        lai_2000_values = [results_ppe_2000_v1[region_name]['FATES_LAI'][n] for n in ensemble_nums 
                           if not np.isnan(results_ppe_2000_v1[region_name]['FATES_LAI'].get(n, np.nan))]
        
        if len(npp_v1_values) > 0 and len(lai_2000_values) > 0:
            # Define "high" as above 75th percentile, "low" as below 25th percentile
            high_npp_threshold = np.percentile(npp_v1_values, 75)
            low_lai_threshold = np.percentile(lai_2000_values, 25)
            
            print(f"  High NPP threshold (75th percentile): {high_npp_threshold:.4f}")
            print(f"  Low LAI threshold (25th percentile): {low_lai_threshold:.4f}")
            
            # Find ensemble members meeting both criteria
            selected_members = []
            for n in ensemble_nums:
                npp_v1 = results_ppe_v1[region_name]['FATES_NPP'].get(n, np.nan)
                lai_2000 = results_ppe_2000_v1[region_name]['FATES_LAI'].get(n, np.nan)
                
                if (not np.isnan(npp_v1) and not np.isnan(lai_2000) and 
                    npp_v1 >= high_npp_threshold and lai_2000 <= low_lai_threshold):
                    selected_members.append(n)
            
            if selected_members:
                print(f"\n  Ensemble members meeting criteria: {selected_members}")
                print("\n  Parameter values for selected members:")
                print(f"  {'Member':<8} {'NPP_V1':<10} {'LAI_2000':<10}", end='')
                for param in PARAMETERS:
                    print(f" {param.replace('fates_', ''):<20}", end='')
                print()
                
                for n in selected_members:
                    npp_v1 = results_ppe_v1[region_name]['FATES_NPP'][n]
                    lai_2000 = results_ppe_2000_v1[region_name]['FATES_LAI'][n]
                    print(f"  n{n:02d}      {npp_v1:<10.4f} {lai_2000:<10.4f}", end='')
                    for param in PARAMETERS:
                        val = param_values[param].get(n, np.nan)
                        print(f" {val:<20.6f}", end='')
                    print()
                
                # Save to CSV
                csv_data = {
                    'ensemble_member': selected_members,
                    'FATES_NPP_PPE_V1': [results_ppe_v1[region_name]['FATES_NPP'][n] for n in selected_members],
                    'FATES_LAI_PPE_2000_V1': [results_ppe_2000_v1[region_name]['FATES_LAI'][n] for n in selected_members]
                }
                for param in PARAMETERS:
                    csv_data[param] = [param_values[param].get(n, np.nan) for n in selected_members]
                
                df_selected = pd.DataFrame(csv_data)
                year_range_str = f"years{min(YEARS):02d}-{max(YEARS):02d}"
                csv_file = csv_dir / f"selected_high_NPP_low_LAI_{year_range_str}_{region_name}.csv"
                df_selected.to_csv(csv_file, index=False)
                print(f"\n  Saved selected members to: {csv_file}")
            else:
                print("\n  No ensemble members meet both criteria.")
        else:
            print("  Insufficient data for analysis.")
