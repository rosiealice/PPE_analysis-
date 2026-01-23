#!/usr/bin/env python
"""
Script to identify and extract PFT-dominated grid cells from FATES model output.
This script reads FATES_NOCOMP_PATCHAREA_PF data and identifies the top 20 grid cells
for each PFT of interest, then saves the locations to text files and creates maps.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuration
BASE_PATH = "/datalake/NS9560K/rosief"
OUTPUT_PATH = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/pft_grid_cells/"
YEARS = range(29, 30)  # Years to use for identification

# PFT configuration: name -> (pft_index, colormap)
PFT_CONFIG = {
    'BDT': {'index': 5, 'pft_num': 6, 'name': 'Broadleaf Deciduous Trees', 'cmap': 'YlGn'},
    'BET': {'index': 0, 'pft_num': 1, 'name': 'Broadleaf Evergreen Trees', 'cmap': 'Greens'},
    'ENT': {'index': 1, 'pft_num': 2, 'name': 'Evergreen Needleleaf Trees', 'cmap': 'BuGn'},
    'AC3G': {'index': 11, 'pft_num': 12, 'name': 'Arctic C3 Grass', 'cmap': 'YlOrBr'},
    'C3G': {'index': 12, 'pft_num': 13, 'name': 'C3 Grass', 'cmap': 'RdYlGn'},
    'C4': {'index': 13, 'pft_num': 14, 'name': 'C4 Grass', 'cmap': 'Oranges'},
}

def extract_top_grid_cells(pft_data, lat, lon, n_cells=20):
    """
    Extract top N grid cells for a given PFT.
    
    Parameters:
    -----------
    pft_data : xarray.DataArray
        PFT coverage data
    lat : xarray.DataArray
        Latitude values
    lon : xarray.DataArray
        Longitude values
    n_cells : int
        Number of top cells to extract
        
    Returns:
    --------
    dict : Dictionary containing lats, lons, and values of top cells
    """
    # Take time mean
    pft_mean = pft_data.mean(dim='time')
    
    # Flatten and find top N grid cells
    pft_flat = pft_mean.values.flatten()
    lat_flat = np.repeat(lat.values, len(lon))
    lon_flat = np.tile(lon.values, len(lat))
    
    # Sort and get top N
    valid_mask = ~np.isnan(pft_flat)
    pft_valid = pft_flat[valid_mask]
    lat_valid = lat_flat[valid_mask]
    lon_valid = lon_flat[valid_mask]
    
    top_indices = np.argsort(pft_valid)[-n_cells:]
    
    return {
        'lats': lat_valid[top_indices],
        'lons': lon_valid[top_indices],
        'values': pft_valid[top_indices]
    }

def create_pft_map(pft_mean, grid_cells, pft_name, pft_num, cmap, output_file):
    """
    Create a two-panel map showing full PFT distribution and top grid cells.
    
    Parameters:
    -----------
    pft_mean : xarray.DataArray
        Mean PFT coverage data
    grid_cells : dict
        Dictionary with top grid cell information
    pft_name : str
        Full name of the PFT
    pft_num : int
        PFT number
    cmap : str
        Colormap name
    output_file : str
        Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Left subplot: Full PFT distribution
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.gridlines(draw_labels=True, alpha=0.3)
    
    pft_plot = pft_mean.plot(ax=ax1, transform=ccrs.PlateCarree(), 
                            cmap=cmap, add_colorbar=True,
                            cbar_kwargs={'label': f'PFT {pft_num} Coverage', 'shrink': 0.6})
    ax1.set_title(f'Full Spatial Distribution of PFT {pft_num} ({pft_name})', 
                 fontsize=12, fontweight='bold')
    
    # Right subplot: Top 20 grid cells
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax2.gridlines(draw_labels=True, alpha=0.3)
    
    scatter = ax2.scatter(grid_cells['lons'], grid_cells['lats'], 
                       c=grid_cells['values'], cmap=cmap, 
                       s=200, marker='s', edgecolors='black', linewidth=2,
                       transform=ccrs.PlateCarree(), zorder=6)
    
    plt.colorbar(scatter, ax=ax2, label=f'PFT {pft_num} Coverage', shrink=0.6)
    ax2.set_title(f'Top 20 Grid Cells for {pft_name} (PFT {pft_num})', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved map: {output_file}")
    plt.close(fig)

def main():
    """Main function to process all PFTs and extract top grid cells."""
    print("="*60)
    print("Identifying PFT-dominated grid cells from FATES output")
    print("="*60 + "\n")
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_PATH}\n")
    
    # Read one ensemble member to identify top grid cells for various PFTs
    # Using first ensemble member and first year
    n = 1
    ensemble_id = f"n{n:02d}"
    dir_path = f"{BASE_PATH}/noresm_beta07_crujra_ppe_v1_{ensemble_id}/lnd/hist/"
    year = min(YEARS)
    pattern = f"{dir_path}noresm_beta07_crujra_ppe_v1_{ensemble_id}.clm2.h0a.00{year:02d}-*.nc"
    files = sorted(glob(pattern))
    
    if not files:
        print(f"ERROR: No files found matching pattern: {pattern}")
        return
    
    print(f"Reading FATES_NOCOMP_PATCHAREA_PF from {len(files)} files...")
    ds = xr.open_mfdataset(files[:1], combine='by_coords')  # Just use first file for identification
    
    if 'FATES_NOCOMP_PATCHAREA_PF' not in ds:
        print("ERROR: FATES_NOCOMP_PATCHAREA_PF not found in dataset")
        ds.close()
        return
    
    # Process each PFT
    for pft_short_name, pft_info in PFT_CONFIG.items():
        print("\n" + "="*60)
        print(f"Processing {pft_short_name} ({pft_info['name']}) grid cells...")
        
        # Extract PFT data
        pft_data = ds['FATES_NOCOMP_PATCHAREA_PF'].isel(fates_levpft=pft_info['index'])
        
        # Extract top grid cells
        grid_cells = extract_top_grid_cells(pft_data, ds['lat'], ds['lon'], n_cells=20)
        
        print(f"  Found top 20 {pft_short_name} grid cells:")
        print(f"  Lat range: {grid_cells['lats'].min():.2f} to {grid_cells['lats'].max():.2f}")
        print(f"  Lon range: {grid_cells['lons'].min():.2f} to {grid_cells['lons'].max():.2f}")
        print(f"  PFT {pft_info['pft_num']} coverage range: {grid_cells['values'].min():.4f} to {grid_cells['values'].max():.4f}")
        
        # Save grid cells to file
        output_file = Path(OUTPUT_PATH) / f"{pft_short_name.lower()}_grid_cells.txt"
        grid_array = np.column_stack([grid_cells['lats'], grid_cells['lons']])
        np.savetxt(output_file, grid_array, fmt='%.6f', header='lat lon', comments='')
        print(f"  Saved {pft_short_name} grid cells to: {output_file}")
        
        # Create map
        print(f"\n  Creating map of {pft_short_name} grid cells...")
        pft_mean = pft_data.mean(dim='time')
        map_file = f"{OUTPUT_PATH}/{pft_short_name.lower()}_gridcells_map.png"
        create_pft_map(pft_mean, grid_cells, pft_info['name'], pft_info['pft_num'], 
                      pft_info['cmap'], map_file)
    
    ds.close()
    
    print("\n" + "="*60)
    print("PFT grid cell extraction complete!")
    print("="*60)

if __name__ == "__main__":
    main()
