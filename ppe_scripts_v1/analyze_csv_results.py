
#!/usr/bin/env python
"""
Script to analyze the CSV outputs from ensemble analysis
Reads the ensemble data and performs additional analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuration
DATA_DIR = Path("/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/")
CSV_DIR = DATA_DIR / "csv_data"  # CSV files are stored in this subdirectory
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR_NPP_LAI = OUTPUT_DIR / "npp1850_lai2000"
OUTPUT_DIR_VEGC_LAI = OUTPUT_DIR / "vegc2000_lai2000"
OUTPUT_DIR_LAI_EMULATOR = OUTPUT_DIR / "lai_emulator"
OUTPUT_DIR_CO2_FERT = OUTPUT_DIR / "co2_fert"
OUTPUT_DIR_CUE = OUTPUT_DIR / "cue"
OUTPUT_DIR_GPP = OUTPUT_DIR / "gpp"

# Create output directories if they don't exist
OUTPUT_DIR_NPP_LAI.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_VEGC_LAI.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_LAI_EMULATOR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_CO2_FERT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_CUE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_GPP.mkdir(parents=True, exist_ok=True)

# Define regions
REGIONS = ['north_65', 'BDTs', 'BETs', 'ENTs', 'AC3G', 'C3G', 'C4']

# Year range used in the analysis (from analyze_ensemble_timeseries.py)
YEAR_RANGE = "years29-29"  # Update this to match the YEARS variable in analyze_ensemble_timeseries.py

# Region definitions (same as in analyze_ensemble_timeseries.py)
REGION_BOUNDS = {
    'north_65': {'lat_min': 65, 'lat_max': 90, 'lon_min': 0, 'lon_max': 360},
}

print("="*60)
print("ANALYZING ENSEMBLE CSV DATA")
print("="*60)

# Read observational LAI data
print("\nReading observational LAI data from MODIS...")
OBS_LAI_FILE_MODIS_TIMESERIES = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/lai/MODIS/lai_0.5x0.5.nc"  # Normal MODIS LAI timeseries
OBS_LAI_FILE_MODIS_SUMMER_MEAN = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/modis_lai/modis_lai_summer_mean.nc"  # Summer mean for ENTs
OBS_LAI_FILE_MODIS_ANNUAL_MAX = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/modis_lai/modis_lai_mean_annual_maximum.nc"  # Mean annual maximum
OBS_LAI_FILE_AVHRR = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/lai/AVHRR/lai_0.5x0.5.nc"
OBS_LAI_FILE = OBS_LAI_FILE_MODIS_TIMESERIES  # Primary dataset for backward compatibility
obs_lai_regional = {}  # For MODIS data (summer LAI for ENTs)
obs_lai_regional_avhrr = {}  # For AVHRR data
obs_lai_regional_mean_ent = {}  # For mean (annual) LAI for ENTs
obs_lai_regional_annual_max_ent = {}  # For mean annual maximum LAI for ENTs
ent_lai_summer_points = []  # Individual summer LAI values for ENT points
ent_lai_mean_points = []  # Individual mean LAI values for ENT points
ent_lai_annual_max_points = []  # Individual annual max LAI values for ENT points

# Read observational GPP data
print("\nReading observational GPP data from FLUXCOM...")
OBS_GPP_FILE_FLUXCOM = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/gpp/FLUXCOM/gpp.nc"
OBS_GPP_FILE_WECANN = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/gpp/WECANN/gpp.nc"
OBS_GPP_FILE = OBS_GPP_FILE_FLUXCOM  # Primary dataset for backward compatibility
obs_gpp_regional = {}  # For FLUXCOM data
obs_gpp_regional_wecann = {}  # For WECANN data

# Load BDT grid cells if available
BDT_GRID_CELLS = None
BDT_LAT_LONS_FILE = DATA_DIR / "bdt_grid_cells.txt"
if BDT_LAT_LONS_FILE.exists():
    print("\nLoading BDT grid cell locations...")
    bdt_data = np.loadtxt(BDT_LAT_LONS_FILE, skiprows=1)
    BDT_GRID_CELLS = {
        'lats': bdt_data[:, 0],
        'lons': bdt_data[:, 1]
    }
    print(f"  Loaded {len(BDT_GRID_CELLS['lats'])} BDT grid cells")

# Load BET grid cells if available
BET_GRID_CELLS = None
BET_LAT_LONS_FILE = DATA_DIR / "bet_grid_cells.txt"
if BET_LAT_LONS_FILE.exists():
    print("\nLoading BET grid cell locations...")
    bet_data = np.loadtxt(BET_LAT_LONS_FILE, skiprows=1)
    BET_GRID_CELLS = {
        'lats': bet_data[:, 0],
        'lons': bet_data[:, 1]
    }
    print(f"  Loaded {len(BET_GRID_CELLS['lats'])} BET grid cells")

# Load ENT grid cells if available
ENT_GRID_CELLS = None
ENT_LAT_LONS_FILE = DATA_DIR / "ent_grid_cells.txt"
if ENT_LAT_LONS_FILE.exists():
    print("\nLoading ENT grid cell locations...")
    ent_data = np.loadtxt(ENT_LAT_LONS_FILE, skiprows=1)
    ENT_GRID_CELLS = {
        'lats': ent_data[:, 0],
        'lons': ent_data[:, 1]
    }
    print(f"  Loaded {len(ENT_GRID_CELLS['lats'])} ENT grid cells")

# Load AC3G grid cells if available
AC3G_GRID_CELLS = None
AC3G_LAT_LONS_FILE = DATA_DIR / "ac3g_grid_cells.txt"
if AC3G_LAT_LONS_FILE.exists():
    print("\nLoading AC3G grid cell locations...")
    ac3g_data = np.loadtxt(AC3G_LAT_LONS_FILE, skiprows=1)
    AC3G_GRID_CELLS = {
        'lats': ac3g_data[:, 0],
        'lons': ac3g_data[:, 1]
    }
    print(f"  Loaded {len(AC3G_GRID_CELLS['lats'])} AC3G grid cells")

# Load C3G grid cells if available
C3G_GRID_CELLS = None
C3G_LAT_LONS_FILE = DATA_DIR / "c3g_grid_cells.txt"
if C3G_LAT_LONS_FILE.exists():
    print("\nLoading C3G grid cell locations...")
    c3g_data = np.loadtxt(C3G_LAT_LONS_FILE, skiprows=1)
    C3G_GRID_CELLS = {
        'lats': c3g_data[:, 0],
        'lons': c3g_data[:, 1]
    }
    print(f"  Loaded {len(C3G_GRID_CELLS['lats'])} C3G grid cells")

# Load C4 grid cells if available
C4_GRID_CELLS = None
C4_LAT_LONS_FILE = DATA_DIR / "c4_grid_cells.txt"
if C4_LAT_LONS_FILE.exists():
    print("\nLoading C4 grid cell locations...")
    c4_data = np.loadtxt(C4_LAT_LONS_FILE, skiprows=1)
    C4_GRID_CELLS = {
        'lats': c4_data[:, 0],
        'lons': c4_data[:, 1]
    }
    print(f"  Loaded {len(C4_GRID_CELLS['lats'])} C4 grid cells")

try:
    # Load normal MODIS LAI timeseries for non-ENT PFTs
    ds_obs = xr.open_dataset(OBS_LAI_FILE_MODIS_TIMESERIES)
    print(f"  Loaded MODIS LAI timeseries data")
    print(f"  Available variables: {list(ds_obs.data_vars)}")
    
    # Convert longitude from -180:180 to 0:360 if needed
    if ds_obs['lon'].min() < 0:
        print(f"  Converting longitude from -180:180 to 0:360...")
        ds_obs = ds_obs.assign_coords(lon=(ds_obs['lon'] % 360))
        # Sort by longitude to maintain order
        ds_obs = ds_obs.sortby('lon')
    
    print(f"  Longitude range: {float(ds_obs['lon'].min()):.2f} to {float(ds_obs['lon'].max()):.2f}")
    
    # Determine the LAI variable name
    lai_var = None
    for var in ['lai', 'LAI', 'leaf_area_index']:
        if var in ds_obs:
            lai_var = var
            break
    
    if lai_var is None:
        print(f"  WARNING: Could not find LAI variable in dataset")
        for region in REGIONS:
            obs_lai_regional[region] = np.nan
    else:
        print(f"  Using LAI variable: {lai_var}")
        
        # Calculate regional averages
        for region, bounds in REGION_BOUNDS.items():
            # Subset by lat and lon
            ds_subset = ds_obs[lai_var].where(
                (ds_obs['lat'] >= bounds['lat_min']) &
                (ds_obs['lat'] <= bounds['lat_max']) &
                (ds_obs['lon'] >= bounds['lon_min']) &
                (ds_obs['lon'] <= bounds['lon_max']),
                drop=False
            )
            # Compute spatial and temporal mean (excluding NaN)
            obs_mean = float(ds_subset.mean(skipna=True).values)
            obs_lai_regional[region] = obs_mean
            print(f"  {region}: {obs_mean:.4f}")
        
        # Handle BDTs separately using specific grid cells
        if BDT_GRID_CELLS is not None:
            print(f"\n  Extracting MODIS LAI for BDT grid cells...")
            bdt_lai_values = []
            for lat, lon in zip(BDT_GRID_CELLS['lats'], BDT_GRID_CELLS['lons']):
                lai_val = ds_obs[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bdt_lai_values.append(float(lai_val))
            obs_lai_regional['BDTs'] = np.nanmean(bdt_lai_values)
            print(f"  BDTs: {obs_lai_regional['BDTs']:.4f}")
        
        # Handle BETs separately using specific grid cells
        if BET_GRID_CELLS is not None:
            print(f"\n  Extracting MODIS LAI for BET grid cells...")
            bet_lai_values = []
            for lat, lon in zip(BET_GRID_CELLS['lats'], BET_GRID_CELLS['lons']):
                lai_val = ds_obs[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bet_lai_values.append(float(lai_val))
            obs_lai_regional['BETs'] = np.nanmean(bet_lai_values)
            print(f"  BETs: {obs_lai_regional['BETs']:.4f}")
        
        # Handle ENTs separately using specific grid cells and summer mean file
        if ENT_GRID_CELLS is not None:
            print(f"\n  Extracting MODIS LAI for ENT grid cells from summer mean file...")
            # Load the summer mean file specifically for ENTs
            ds_ent = xr.open_dataset(OBS_LAI_FILE_MODIS_SUMMER_MEAN)
            # Convert longitude if needed
            if ds_ent['lon'].min() < 0:
                ds_ent = ds_ent.assign_coords(lon=(ds_ent['lon'] % 360))
                ds_ent = ds_ent.sortby('lon')
            
            # Find LAI variable in ENT dataset
            lai_var_ent = None
            for var in ['lai', 'LAI', 'leaf_area_index']:
                if var in ds_ent:
                    lai_var_ent = var
                    break
            
            if lai_var_ent:
                ent_lai_values = []
                for lat, lon in zip(ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']):
                    lai_val = ds_ent[lai_var_ent].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                    ent_lai_values.append(float(lai_val))
                obs_lai_regional['ENTs'] = np.nanmean(ent_lai_values)
                ent_lai_summer_points = ent_lai_values.copy()  # Store individual point values
                print(f"  ENTs (summer): {obs_lai_regional['ENTs']:.4f}")
            else:
                print(f"  WARNING: Could not find LAI variable in ENT summer mean file")
                obs_lai_regional['ENTs'] = np.nan
            
            ds_ent.close()
            
            # Also extract mean (annual) LAI for ENTs from timeseries file
            print(f"  Extracting MODIS mean LAI for ENT grid cells from timeseries file...")
            ent_lai_mean_values = []
            for lat, lon in zip(ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']):
                lai_val = ds_obs[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ent_lai_mean_values.append(float(lai_val))
            obs_lai_regional_mean_ent['ENTs'] = np.nanmean(ent_lai_mean_values)
            ent_lai_mean_points = ent_lai_mean_values.copy()  # Store individual point values
            print(f"  ENTs (mean): {obs_lai_regional_mean_ent['ENTs']:.4f}")
            
            # Also extract mean annual maximum LAI for ENTs
            if os.path.exists(OBS_LAI_FILE_MODIS_ANNUAL_MAX):
                print(f"  Extracting MODIS mean annual maximum LAI for ENT grid cells...")
                ds_annual_max = xr.open_dataset(OBS_LAI_FILE_MODIS_ANNUAL_MAX)
                lai_var_max = None
                for var in ['lai', 'LAI', 'leaf_area_index']:
                    if var in ds_annual_max:
                        lai_var_max = var
                        break
                
                if lai_var_max:
                    ent_lai_annual_max_values = []
                    for lat, lon in zip(ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']):
                        lai_val = ds_annual_max[lai_var_max].sel(lat=lat, lon=lon, method='nearest').values
                        ent_lai_annual_max_values.append(float(lai_val))
                    obs_lai_regional_annual_max_ent['ENTs'] = np.nanmean(ent_lai_annual_max_values)
                    ent_lai_annual_max_points = ent_lai_annual_max_values.copy()
                    print(f"  ENTs (annual max): {obs_lai_regional_annual_max_ent['ENTs']:.4f}")
                else:
                    print(f"  WARNING: Could not find LAI variable in annual max file")
                    obs_lai_regional_annual_max_ent['ENTs'] = np.nan
                
                ds_annual_max.close()
            else:
                print(f"  WARNING: Annual max file not found: {OBS_LAI_FILE_MODIS_ANNUAL_MAX}")
                obs_lai_regional_annual_max_ent['ENTs'] = np.nan
            
            # Create scatter plot comparing summer vs mean LAI for ENT points
            if len(ent_lai_summer_points) > 0 and len(ent_lai_mean_points) > 0:
                print(f"\n  Creating scatter plot: Summer LAI vs Mean LAI for ENT points...")
                fig_ent, ax_ent = plt.subplots(figsize=(10, 8))
                
                # Scatter plot
                ax_ent.scatter(ent_lai_mean_points, ent_lai_summer_points, s=100, alpha=0.6,
                              c='forestgreen', edgecolors='black', linewidth=1)
                
                # Add 1:1 line
                min_val = min(min(ent_lai_mean_points), min(ent_lai_summer_points))
                max_val = max(max(ent_lai_mean_points), max(ent_lai_summer_points))
                ax_ent.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                           alpha=0.7, label='1:1 line')
                
                # Add mean values as lines
                ax_ent.axvline(obs_lai_regional_mean_ent['ENTs'], color='darkblue', 
                              linestyle='-.', linewidth=2, alpha=0.7, 
                              label=f'Mean LAI avg: {obs_lai_regional_mean_ent["ENTs"]:.2f}')
                ax_ent.axhline(obs_lai_regional['ENTs'], color='darkgreen', 
                              linestyle='--', linewidth=2, alpha=0.7, 
                              label=f'Summer LAI avg: {obs_lai_regional["ENTs"]:.2f}')
                
                ax_ent.set_xlabel('Mean (Annual) LAI [m²/m²]', fontsize=12, fontweight='bold')
                ax_ent.set_ylabel('Summer LAI [m²/m²]', fontsize=12, fontweight='bold')
                ax_ent.set_title('MODIS LAI: Summer vs Mean for ENT Grid Points', 
                                fontsize=14, fontweight='bold')
                ax_ent.legend(loc='best', fontsize=10)
                ax_ent.grid(True, alpha=0.3)
                
                # Add text with statistics
                correlation = np.corrcoef(ent_lai_mean_points, ent_lai_summer_points)[0, 1]
                n_points = len(ent_lai_mean_points)
                ax_ent.text(0.05, 0.95, f'N points: {n_points}\nCorrelation: {correlation:.3f}',
                           transform=ax_ent.transAxes, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                output_file_ent = OUTPUT_DIR / "ent_summer_vs_mean_lai.png"
                fig_ent.savefig(output_file_ent, dpi=300, bbox_inches='tight')
                print(f"  Saved: {output_file_ent}")
                plt.close(fig_ent)
        
        # Handle AC3G separately using specific grid cells
        if AC3G_GRID_CELLS is not None:
            print(f"\n  Extracting MODIS LAI for AC3G grid cells...")
            ac3g_lai_values = []
            for lat, lon in zip(AC3G_GRID_CELLS['lats'], AC3G_GRID_CELLS['lons']):
                lai_val = ds_obs[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ac3g_lai_values.append(float(lai_val))
            obs_lai_regional['AC3G'] = np.nanmean(ac3g_lai_values)
            print(f"  AC3G: {obs_lai_regional['AC3G']:.4f}")
        
        # Handle C3G separately using specific grid cells
        if C3G_GRID_CELLS is not None:
            print(f"\n  Extracting MODIS LAI for C3G grid cells...")
            c3g_lai_values = []
            for lat, lon in zip(C3G_GRID_CELLS['lats'], C3G_GRID_CELLS['lons']):
                lai_val = ds_obs[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c3g_lai_values.append(float(lai_val))
            obs_lai_regional['C3G'] = np.nanmean(c3g_lai_values)
            print(f"  C3G: {obs_lai_regional['C3G']:.4f}")
        
        # Handle C4 separately using specific grid cells
        if C4_GRID_CELLS is not None:
            print(f"\n  Extracting MODIS LAI for C4 grid cells...")
            c4_lai_values = []
            for lat, lon in zip(C4_GRID_CELLS['lats'], C4_GRID_CELLS['lons']):
                lai_val = ds_obs[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c4_lai_values.append(float(lai_val))
            obs_lai_regional['C4'] = np.nanmean(c4_lai_values)
            print(f"  C4: {obs_lai_regional['C4']:.4f}")
        
        ds_obs.close()
        
except Exception as e:
    print(f"  ERROR reading observational LAI data: {str(e)}")
    for region in REGIONS:
        obs_lai_regional[region] = np.nan

# Read observational LAI data from AVHRR
print("\nReading observational LAI data from AVHRR...")
try:
    ds_obs_avhrr = xr.open_dataset(OBS_LAI_FILE_AVHRR)
    print(f"  Loaded AVHRR LAI data")
    print(f"  Available variables: {list(ds_obs_avhrr.data_vars)}")
    
    # Convert longitude from -180:180 to 0:360 if needed
    if ds_obs_avhrr['lon'].min() < 0:
        print(f"  Converting longitude from -180:180 to 0:360...")
        ds_obs_avhrr = ds_obs_avhrr.assign_coords(lon=(ds_obs_avhrr['lon'] % 360))
        # Sort by longitude to maintain order
        ds_obs_avhrr = ds_obs_avhrr.sortby('lon')
    
    print(f"  Longitude range: {float(ds_obs_avhrr['lon'].min()):.2f} to {float(ds_obs_avhrr['lon'].max()):.2f}")
    
    # Determine the LAI variable name
    lai_var = None
    for var in ['lai', 'LAI', 'leaf_area_index']:
        if var in ds_obs_avhrr:
            lai_var = var
            break
    
    if lai_var is None:
        print(f"  WARNING: Could not find LAI variable in dataset")
        for region in REGIONS:
            obs_lai_regional_avhrr[region] = np.nan
    else:
        print(f"  Using LAI variable: {lai_var}")
        
        # Calculate regional averages
        for region, bounds in REGION_BOUNDS.items():
            # Subset by lat and lon
            ds_subset = ds_obs_avhrr[lai_var].where(
                (ds_obs_avhrr['lat'] >= bounds['lat_min']) &
                (ds_obs_avhrr['lat'] <= bounds['lat_max']) &
                (ds_obs_avhrr['lon'] >= bounds['lon_min']) &
                (ds_obs_avhrr['lon'] <= bounds['lon_max']),
                drop=False
            )
            # Compute spatial and temporal mean (excluding NaN)
            obs_mean = float(ds_subset.mean(skipna=True).values)
            obs_lai_regional_avhrr[region] = obs_mean
            print(f"  {region}: {obs_mean:.4f}")
        
        # Handle BDTs separately using specific grid cells
        if BDT_GRID_CELLS is not None:
            print(f"\n  Extracting AVHRR LAI for BDT grid cells...")
            bdt_lai_values = []
            for lat, lon in zip(BDT_GRID_CELLS['lats'], BDT_GRID_CELLS['lons']):
                lai_val = ds_obs_avhrr[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bdt_lai_values.append(float(lai_val))
            obs_lai_regional_avhrr['BDTs'] = np.nanmean(bdt_lai_values)
            print(f"  BDTs: {obs_lai_regional_avhrr['BDTs']:.4f}")
        
        # Handle BETs separately using specific grid cells
        if BET_GRID_CELLS is not None:
            print(f"\n  Extracting AVHRR LAI for BET grid cells...")
            bet_lai_values = []
            for lat, lon in zip(BET_GRID_CELLS['lats'], BET_GRID_CELLS['lons']):
                lai_val = ds_obs_avhrr[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bet_lai_values.append(float(lai_val))
            obs_lai_regional_avhrr['BETs'] = np.nanmean(bet_lai_values)
            print(f"  BETs: {obs_lai_regional_avhrr['BETs']:.4f}")
        
        # Handle ENTs separately using specific grid cells
        if ENT_GRID_CELLS is not None:
            print(f"\n  Extracting AVHRR LAI for ENT grid cells (summer months 6, 7, 8)...")
            ent_lai_values = []
            for lat, lon in zip(ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']):
                # Select summer months (6, 7, 8) for ENTs
                ds_point = ds_obs_avhrr[lai_var].sel(lat=lat, lon=lon, method='nearest')
                if 'time' in ds_point.dims:
                    ds_summer = ds_point.sel(time=ds_point.time.dt.month.isin([6, 7, 8]))
                    lai_val = ds_summer.mean(skipna=True).values
                else:
                    lai_val = ds_point.mean(skipna=True).values
                ent_lai_values.append(float(lai_val))
            obs_lai_regional_avhrr['ENTs'] = np.nanmean(ent_lai_values)
            print(f"  ENTs: {obs_lai_regional_avhrr['ENTs']:.4f}")
        
        # Handle AC3G separately using specific grid cells
        if AC3G_GRID_CELLS is not None:
            print(f"\n  Extracting AVHRR LAI for AC3G grid cells...")
            ac3g_lai_values = []
            for lat, lon in zip(AC3G_GRID_CELLS['lats'], AC3G_GRID_CELLS['lons']):
                lai_val = ds_obs_avhrr[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ac3g_lai_values.append(float(lai_val))
            obs_lai_regional_avhrr['AC3G'] = np.nanmean(ac3g_lai_values)
            print(f"  AC3G: {obs_lai_regional_avhrr['AC3G']:.4f}")
        
        # Handle C3G separately using specific grid cells
        if C3G_GRID_CELLS is not None:
            print(f"\n  Extracting AVHRR LAI for C3G grid cells...")
            c3g_lai_values = []
            for lat, lon in zip(C3G_GRID_CELLS['lats'], C3G_GRID_CELLS['lons']):
                lai_val = ds_obs_avhrr[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c3g_lai_values.append(float(lai_val))
            obs_lai_regional_avhrr['C3G'] = np.nanmean(c3g_lai_values)
            print(f"  C3G: {obs_lai_regional_avhrr['C3G']:.4f}")
        
        # Handle C4 separately using specific grid cells
        if C4_GRID_CELLS is not None:
            print(f"\n  Extracting AVHRR LAI for C4 grid cells...")
            c4_lai_values = []
            for lat, lon in zip(C4_GRID_CELLS['lats'], C4_GRID_CELLS['lons']):
                lai_val = ds_obs_avhrr[lai_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c4_lai_values.append(float(lai_val))
            obs_lai_regional_avhrr['C4'] = np.nanmean(c4_lai_values)
            print(f"  C4: {obs_lai_regional_avhrr['C4']:.4f}")
        
        ds_obs_avhrr.close()
        
except Exception as e:
    print(f"  ERROR reading AVHRR LAI data: {str(e)}")
    for region in REGIONS:
        obs_lai_regional_avhrr[region] = np.nan

# Create maps of MODIS and AVHRR LAI
print("\n" + "="*60)
print("Creating LAI observation maps...")
print("="*60)

try:
    # Reload datasets for mapping
    ds_modis = xr.open_dataset(OBS_LAI_FILE_MODIS_TIMESERIES)
    ds_avhrr = xr.open_dataset(OBS_LAI_FILE_AVHRR)
    
    # Convert longitude if needed
    if ds_modis['lon'].min() < 0:
        ds_modis = ds_modis.assign_coords(lon=(ds_modis['lon'] % 360))
        ds_modis = ds_modis.sortby('lon')
    if ds_avhrr['lon'].min() < 0:
        ds_avhrr = ds_avhrr.assign_coords(lon=(ds_avhrr['lon'] % 360))
        ds_avhrr = ds_avhrr.sortby('lon')
    
    # Find LAI variable
    lai_var_modis = None
    for var in ['lai', 'LAI', 'leaf_area_index']:
        if var in ds_modis:
            lai_var_modis = var
            break
    
    lai_var_avhrr = None
    for var in ['lai', 'LAI', 'leaf_area_index']:
        if var in ds_avhrr:
            lai_var_avhrr = var
            break
    
    if lai_var_modis and lai_var_avhrr:
        # Calculate temporal mean for each dataset
        modis_lai_mean = ds_modis[lai_var_modis].mean(dim='time', skipna=True)
        avhrr_lai_mean = ds_avhrr[lai_var_avhrr].mean(dim='time', skipna=True)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        # MODIS map
        ax1 = axes[0]
        im1 = modis_lai_mean.plot(ax=ax1, transform=ccrs.PlateCarree(),
                                   cmap='YlGn', vmin=0, vmax=6,
                                   cbar_kwargs={'label': 'LAI (m²/m²)', 'shrink': 0.8})
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.set_title('MODIS LAI (Temporal Mean)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        # Add grid cell markers for ENTs if available
        if ENT_GRID_CELLS is not None:
            ax1.scatter(ENT_GRID_CELLS['lons'], ENT_GRID_CELLS['lats'],
                       c='red', s=20, marker='x', transform=ccrs.PlateCarree(),
                       label=f'ENT grid cells (n={len(ENT_GRID_CELLS["lats"])})')
            ax1.legend(loc='lower left')
        
        # AVHRR map
        ax2 = axes[1]
        im2 = avhrr_lai_mean.plot(ax=ax2, transform=ccrs.PlateCarree(),
                                   cmap='YlGn', vmin=0, vmax=6,
                                   cbar_kwargs={'label': 'LAI (m²/m²)', 'shrink': 0.8})
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.set_title('AVHRR LAI (Temporal Mean)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        # Add grid cell markers for ENTs if available
        if ENT_GRID_CELLS is not None:
            ax2.scatter(ENT_GRID_CELLS['lons'], ENT_GRID_CELLS['lats'],
                       c='red', s=20, marker='x', transform=ccrs.PlateCarree(),
                       label=f'ENT grid cells (n={len(ENT_GRID_CELLS["lats"])})')
            ax2.legend(loc='lower left')
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / "lai_observations_comparison_map.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved LAI comparison map to: {output_file}")
        plt.close(fig)
        
        # Also create a difference map
        fig_diff, ax_diff = plt.subplots(1, 1, figsize=(14, 8),
                                         subplot_kw={'projection': ccrs.PlateCarree()})
        
        lai_diff = modis_lai_mean - avhrr_lai_mean
        im_diff = lai_diff.plot(ax=ax_diff, transform=ccrs.PlateCarree(),
                               cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                               cbar_kwargs={'label': 'LAI Difference (MODIS - AVHRR) [m²/m²]', 
                                          'shrink': 0.8})
        ax_diff.coastlines()
        ax_diff.add_feature(cfeature.BORDERS, linestyle=':')
        ax_diff.set_title('LAI Difference: MODIS - AVHRR (Temporal Mean)', 
                         fontsize=14, fontweight='bold')
        ax_diff.set_xlabel('Longitude')
        ax_diff.set_ylabel('Latitude')
        ax_diff.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        if ENT_GRID_CELLS is not None:
            ax_diff.scatter(ENT_GRID_CELLS['lons'], ENT_GRID_CELLS['lats'],
                           c='magenta', s=20, marker='x', transform=ccrs.PlateCarree(),
                           label=f'ENT grid cells (n={len(ENT_GRID_CELLS["lats"])})')
            ax_diff.legend(loc='lower left')
        
        plt.tight_layout()
        output_file_diff = OUTPUT_DIR / "lai_observations_difference_map.png"
        fig_diff.savefig(output_file_diff, dpi=300, bbox_inches='tight')
        print(f"Saved LAI difference map to: {output_file_diff}")
        plt.close(fig_diff)
        
    ds_modis.close()
    ds_avhrr.close()
    
except Exception as e:
    print(f"  ERROR creating LAI maps: {str(e)}")
    import traceback
    traceback.print_exc()

# Read observational GPP data from FLUXCOM
try:
    ds_obs_gpp = xr.open_dataset(OBS_GPP_FILE)
    print(f"  Loaded FLUXCOM GPP data")
    print(f"  Available variables: {list(ds_obs_gpp.data_vars)}")
    
    # Convert longitude from -180:180 to 0:360 if needed
    if ds_obs_gpp['lon'].min() < 0:
        print(f"  Converting longitude from -180:180 to 0:360...")
        ds_obs_gpp = ds_obs_gpp.assign_coords(lon=(ds_obs_gpp['lon'] % 360))
        # Sort by longitude to maintain order
        ds_obs_gpp = ds_obs_gpp.sortby('lon')
    
    print(f"  Longitude range: {float(ds_obs_gpp['lon'].min()):.2f} to {float(ds_obs_gpp['lon'].max()):.2f}")
    
    # Determine the GPP variable name
    gpp_var = None
    for var in ['gpp', 'GPP', 'gross_primary_productivity']:
        if var in ds_obs_gpp:
            gpp_var = var
            break
    
    if gpp_var is None:
        print(f"  WARNING: Could not find GPP variable in dataset")
        for region in REGIONS:
            obs_gpp_regional[region] = np.nan
    else:
        print(f"  Using GPP variable: {gpp_var}")
        print(f"  GPP units: gC m-2 day-1 (FLUXCOM data product)")
        
        # Calculate regional averages
        for region, bounds in REGION_BOUNDS.items():
            # Subset by lat and lon
            ds_subset = ds_obs_gpp[gpp_var].where(
                (ds_obs_gpp['lat'] >= bounds['lat_min']) &
                (ds_obs_gpp['lat'] <= bounds['lat_max']) &
                (ds_obs_gpp['lon'] >= bounds['lon_min']) &
                (ds_obs_gpp['lon'] <= bounds['lon_max']),
                drop=False
            )
            # Compute spatial and temporal mean (excluding NaN)
            obs_mean = float(ds_subset.mean(skipna=True).values)
            obs_gpp_regional[region] = obs_mean
            print(f"  {region}: {obs_mean:.4f} gC m-2 day-1")
        
        # Handle BDTs separately using specific grid cells
        if BDT_GRID_CELLS is not None:
            print(f"\n  Extracting FLUXCOM GPP for BDT grid cells...")
            bdt_gpp_values = []
            for lat, lon in zip(BDT_GRID_CELLS['lats'], BDT_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bdt_gpp_values.append(float(gpp_val))
            obs_gpp_regional['BDTs'] = np.nanmean(bdt_gpp_values)
            print(f"  BDTs: {obs_gpp_regional['BDTs']:.4f} gC m-2 day-1")
        
        # Handle BETs separately using specific grid cells
        if BET_GRID_CELLS is not None:
            print(f"\n  Extracting FLUXCOM GPP for BET grid cells...")
            bet_gpp_values = []
            for lat, lon in zip(BET_GRID_CELLS['lats'], BET_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bet_gpp_values.append(float(gpp_val))
            obs_gpp_regional['BETs'] = np.nanmean(bet_gpp_values)
            print(f"  BETs: {obs_gpp_regional['BETs']:.4f} gC m-2 day-1")
        
        # Handle ENTs separately using specific grid cells
        if ENT_GRID_CELLS is not None:
            print(f"\n  Extracting FLUXCOM GPP for ENT grid cells...")
            ent_gpp_values = []
            for lat, lon in zip(ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ent_gpp_values.append(float(gpp_val))
            obs_gpp_regional['ENTs'] = np.nanmean(ent_gpp_values)
            print(f"  ENTs: {obs_gpp_regional['ENTs']:.4f} gC m-2 day-1")
        
        # Handle AC3G separately using specific grid cells
        if AC3G_GRID_CELLS is not None:
            print(f"\n  Extracting FLUXCOM GPP for AC3G grid cells...")
            ac3g_gpp_values = []
            for lat, lon in zip(AC3G_GRID_CELLS['lats'], AC3G_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ac3g_gpp_values.append(float(gpp_val))
            obs_gpp_regional['AC3G'] = np.nanmean(ac3g_gpp_values)
            print(f"  AC3G: {obs_gpp_regional['AC3G']:.4f} gC m-2 day-1")
        
        # Handle C3G separately using specific grid cells
        if C3G_GRID_CELLS is not None:
            print(f"\n  Extracting FLUXCOM GPP for C3G grid cells...")
            c3g_gpp_values = []
            for lat, lon in zip(C3G_GRID_CELLS['lats'], C3G_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c3g_gpp_values.append(float(gpp_val))
            obs_gpp_regional['C3G'] = np.nanmean(c3g_gpp_values)
            print(f"  C3G: {obs_gpp_regional['C3G']:.4f} gC m-2 day-1")
        
        # Handle C4 separately using specific grid cells
        if C4_GRID_CELLS is not None:
            print(f"\n  Extracting FLUXCOM GPP for C4 grid cells...")
            c4_gpp_values = []
            for lat, lon in zip(C4_GRID_CELLS['lats'], C4_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c4_gpp_values.append(float(gpp_val))
            obs_gpp_regional['C4'] = np.nanmean(c4_gpp_values)
            print(f"  C4: {obs_gpp_regional['C4']:.4f} gC m-2 day-1")
        
        ds_obs_gpp.close()
        
except Exception as e:
    print(f"  ERROR reading observational GPP data: {str(e)}")
    for region in REGIONS:
        obs_gpp_regional[region] = np.nan

# Read observational GPP data from WECANN
print("\nReading observational GPP data from WECANN...")
try:
    ds_obs_gpp_wecann = xr.open_dataset(OBS_GPP_FILE_WECANN)
    print(f"  Loaded WECANN GPP data")
    print(f"  Available variables: {list(ds_obs_gpp_wecann.data_vars)}")
    
    # Convert longitude from -180:180 to 0:360 if needed
    if ds_obs_gpp_wecann['lon'].min() < 0:
        print(f"  Converting longitude from -180:180 to 0:360...")
        ds_obs_gpp_wecann = ds_obs_gpp_wecann.assign_coords(lon=(ds_obs_gpp_wecann['lon'] % 360))
        # Sort by longitude to maintain order
        ds_obs_gpp_wecann = ds_obs_gpp_wecann.sortby('lon')
    
    print(f"  Longitude range: {float(ds_obs_gpp_wecann['lon'].min()):.2f} to {float(ds_obs_gpp_wecann['lon'].max()):.2f}")
    
    # Determine the GPP variable name
    gpp_var = None
    for var in ['gpp', 'GPP', 'gross_primary_productivity']:
        if var in ds_obs_gpp_wecann:
            gpp_var = var
            break
    
    if gpp_var is None:
        print(f"  WARNING: Could not find GPP variable in dataset")
        for region in REGIONS:
            obs_gpp_regional_wecann[region] = np.nan
    else:
        print(f"  Using GPP variable: {gpp_var}")
        print(f"  GPP units: gC m-2 day-1 (WECANN data product)")
        
        # Calculate regional averages
        for region, bounds in REGION_BOUNDS.items():
            # Subset by lat and lon
            ds_subset = ds_obs_gpp_wecann[gpp_var].where(
                (ds_obs_gpp_wecann['lat'] >= bounds['lat_min']) &
                (ds_obs_gpp_wecann['lat'] <= bounds['lat_max']) &
                (ds_obs_gpp_wecann['lon'] >= bounds['lon_min']) &
                (ds_obs_gpp_wecann['lon'] <= bounds['lon_max']),
                drop=False
            )
            # Compute spatial and temporal mean (excluding NaN)
            obs_mean = float(ds_subset.mean(skipna=True).values)
            obs_gpp_regional_wecann[region] = obs_mean
            print(f"  {region}: {obs_mean:.4f} gC m-2 day-1")
        
        # Handle BDTs separately using specific grid cells
        if BDT_GRID_CELLS is not None:
            print(f"\n  Extracting WECANN GPP for BDT grid cells...")
            bdt_gpp_values = []
            for lat, lon in zip(BDT_GRID_CELLS['lats'], BDT_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp_wecann[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bdt_gpp_values.append(float(gpp_val))
            obs_gpp_regional_wecann['BDTs'] = np.nanmean(bdt_gpp_values)
            print(f"  BDTs: {obs_gpp_regional_wecann['BDTs']:.4f} gC m-2 day-1")
        
        # Handle BETs separately using specific grid cells
        if BET_GRID_CELLS is not None:
            print(f"\n  Extracting WECANN GPP for BET grid cells...")
            bet_gpp_values = []
            for lat, lon in zip(BET_GRID_CELLS['lats'], BET_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp_wecann[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                bet_gpp_values.append(float(gpp_val))
            obs_gpp_regional_wecann['BETs'] = np.nanmean(bet_gpp_values)
            print(f"  BETs: {obs_gpp_regional_wecann['BETs']:.4f} gC m-2 day-1")
        
        # Handle ENTs separately using specific grid cells
        if ENT_GRID_CELLS is not None:
            print(f"\n  Extracting WECANN GPP for ENT grid cells...")
            ent_gpp_values = []
            for lat, lon in zip(ENT_GRID_CELLS['lats'], ENT_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp_wecann[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ent_gpp_values.append(float(gpp_val))
            obs_gpp_regional_wecann['ENTs'] = np.nanmean(ent_gpp_values)
            print(f"  ENTs: {obs_gpp_regional_wecann['ENTs']:.4f} gC m-2 day-1")
        
        # Handle AC3G separately using specific grid cells
        if AC3G_GRID_CELLS is not None:
            print(f"\n  Extracting WECANN GPP for AC3G grid cells...")
            ac3g_gpp_values = []
            for lat, lon in zip(AC3G_GRID_CELLS['lats'], AC3G_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp_wecann[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                ac3g_gpp_values.append(float(gpp_val))
            obs_gpp_regional_wecann['AC3G'] = np.nanmean(ac3g_gpp_values)
            print(f"  AC3G: {obs_gpp_regional_wecann['AC3G']:.4f} gC m-2 day-1")
        
        # Handle C3G separately using specific grid cells
        if C3G_GRID_CELLS is not None:
            print(f"\n  Extracting WECANN GPP for C3G grid cells...")
            c3g_gpp_values = []
            for lat, lon in zip(C3G_GRID_CELLS['lats'], C3G_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp_wecann[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c3g_gpp_values.append(float(gpp_val))
            obs_gpp_regional_wecann['C3G'] = np.nanmean(c3g_gpp_values)
            print(f"  C3G: {obs_gpp_regional_wecann['C3G']:.4f} gC m-2 day-1")
        
        # Handle C4 separately using specific grid cells
        if C4_GRID_CELLS is not None:
            print(f"\n  Extracting WECANN GPP for C4 grid cells...")
            c4_gpp_values = []
            for lat, lon in zip(C4_GRID_CELLS['lats'], C4_GRID_CELLS['lons']):
                gpp_val = ds_obs_gpp_wecann[gpp_var].sel(lat=lat, lon=lon, method='nearest').mean(skipna=True).values
                c4_gpp_values.append(float(gpp_val))
            obs_gpp_regional_wecann['C4'] = np.nanmean(c4_gpp_values)
            print(f"  C4: {obs_gpp_regional_wecann['C4']:.4f} gC m-2 day-1")
        
        ds_obs_gpp_wecann.close()
        
except Exception as e:
    print(f"  ERROR reading WECANN GPP data: {str(e)}")
    for region in REGIONS:
        obs_gpp_regional_wecann[region] = np.nan

# Unit conversion factor for FATES_GPP: kgC m-2 s-1 to gC m-2 day-1
# kgC to gC: multiply by 1000
# s-1 to day-1: multiply by 86400 (seconds per day)
GPP_CONVERSION_FACTOR = 1000.0 * 86400.0
print(f"\nGPP unit conversion factor (kgC m-2 s-1 to gC m-2 day-1): {GPP_CONVERSION_FACTOR}")

# Read data for each region
data = {}
for region in REGIONS:
    # Try new naming convention first, then fall back to old
    csv_file = CSV_DIR / f"ensemble_data_{YEAR_RANGE}_{region}.csv"
    if not csv_file.exists():
        csv_file = CSV_DIR / f"ensemble_data_{region}.csv"
    # Also try the old location in DATA_DIR for backwards compatibility
    if not csv_file.exists():
        csv_file = DATA_DIR / f"ensemble_data_{YEAR_RANGE}_{region}.csv"
    if not csv_file.exists():
        csv_file = DATA_DIR / f"ensemble_data_{region}.csv"
    
    if csv_file.exists():
        data[region] = pd.read_csv(csv_file)
        print(f"\nLoaded data for {region}: {len(data[region])} ensemble members")
        print(f"Columns: {list(data[region].columns)}")
        
        # Convert FATES_GPP variables from kgC m-2 s-1 to gC m-2 day-1
        gpp_vars_to_convert = ['FATES_GPP_PPE_V1', 'FATES_GPP_PPE_2000_V1']
        for gpp_var in gpp_vars_to_convert:
            if gpp_var in data[region].columns:
                data[region][gpp_var] = data[region][gpp_var] * GPP_CONVERSION_FACTOR
                print(f"  Converted {gpp_var} to gC m-2 day-1")
        
        # Convert FATES_NPP variables from kgC m-2 s-1 to gC m-2 day-1 (same units as GPP for CUE calculation)
        npp_vars_to_convert = ['FATES_NPP_PPE_V1', 'FATES_NPP_PPE_2000_V1']
        for npp_var in npp_vars_to_convert:
            if npp_var in data[region].columns:
                data[region][npp_var] = data[region][npp_var] * GPP_CONVERSION_FACTOR
                print(f"  Converted {npp_var} to gC m-2 day-1")
    else:
        print(f"\nWARNING: File not found: {csv_file}")

# Helper function to get the appropriate LAI column name for a region
def get_lai_column(region, ensemble_type):
    """
    Returns the appropriate LAI column  for a given region and ensemble type.
    For ENTs (PFT 2), use summer LAI (months 6, 7, 8).
    For other regions, use regular LAI.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
        ensemble_type: Either 'PPE_V1' or 'PPE_2000_V1'
    
    Returns:
        Column name string (e.g., 'fates_lai_summer_PPE_V1' or 'FATES_LAI_PPE_V1')
    """
    if region == 'ENTs':
        return f'fates_lai_summer_{ensemble_type}'
    else:
        return f'FATES_LAI_{ensemble_type}'

# Helper function to get LAI label for plots
def get_lai_label(region):
    """
    Returns the appropriate LAI label for plot axes and titles.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
    
    Returns:
        Label string (e.g., 'LAI(2000)' or 'Summer LAI(2000)')
    """
    if region == 'ENTs':
        return 'Summer LAI'
    else:
        return 'LAI'

# Read selected members (high NPP, low LAI)
selected_data = {}
for region in REGIONS:
    # Try new naming convention first, then fall back to old
    csv_file = CSV_DIR / f"selected_high_NPP_low_LAI_{YEAR_RANGE}_{region}.csv"
    if not csv_file.exists():
        csv_file = CSV_DIR / f"selected_high_NPP_low_LAI_{region}.csv"
    # Also try the old location in DATA_DIR for backwards compatibility
    if not csv_file.exists():
        csv_file = DATA_DIR / f"selected_high_NPP_low_LAI_{YEAR_RANGE}_{region}.csv"
    if not csv_file.exists():
        csv_file = DATA_DIR / f"selected_high_NPP_low_LAI_{region}.csv"
    
    if csv_file.exists():
        selected_data[region] = pd.read_csv(csv_file)
        print(f"\nLoaded selected members for {region}: {len(selected_data[region])} members")
    else:
        print(f"\nNo selected members file for {region}")

# Create NPP vs LAI plots first
print("\n" + "="*60)
print("Creating NPP (PPE_V1) vs LAI (PPE_2000_V1) plots...")
print("="*60)

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    if 'FATES_NPP_PPE_V1' not in df.columns or lai_col not in df.columns:
        print(f"\nWARNING: Missing required columns for {region}")
        print(f"  Looking for: FATES_NPP_PPE_V1 and {lai_col}")
        continue
    
    print(f"\nCreating plot for {region.replace('_', ' ').title()}")
    if region == 'ENTs':
        print(f"  Using summer LAI (months 6, 7, 8) for ENTs region")
    
    # Create scatter plot: NPP_V1 vs LAI_2000
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all members with ensemble numbers as labels
    scatter = ax.scatter(df['FATES_NPP_PPE_V1'], df[lai_col],
                        s=100, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add ensemble member labels
    for idx, row in df.iterrows():
        ax.annotate(f"n{int(row['ensemble_member']):02d}", 
                   (row['FATES_NPP_PPE_V1'], row[lai_col]),
                   fontsize=8, ha='center', va='bottom')
    
    # Add threshold lines
    npp_threshold = df['FATES_NPP_PPE_V1'].quantile(0.75)
    lai_threshold = df[lai_col].quantile(0.25)
    ax.axvline(npp_threshold, color='red', linestyle='--', alpha=0.5, 
              label=f'NPP 75th %ile = {npp_threshold:.4f}')
    ax.axhline(lai_threshold, color='green', linestyle='--', alpha=0.5, 
              label=f'{lai_label} 25th %ile = {lai_threshold:.4f}')
    
    # Highlight the target region (high NPP, low LAI)
    ax.axvspan(npp_threshold, df['FATES_NPP_PPE_V1'].max() * 1.1, 
              ymin=0, ymax=(lai_threshold - df[lai_col].min()) / 
              (df[lai_col].max() - df[lai_col].min()),
              alpha=0.1, color='yellow', label='Target region')
    
    ax.set_xlabel('NPP(1850)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{lai_label}(2000)', fontsize=12, fontweight='bold')
    ax.set_title(f'NPP(1850) vs {lai_label}(2000) - {region.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_file = OUTPUT_DIR_NPP_LAI / f"npp_v1_vs_lai_2000_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Create plots of NPP(1850) and LAI(2000) vs parameters (on same axes)
print("\n" + "="*60)
print("Creating NPP(1850) and LAI(2000) vs parameter plots...")
print("="*60)

params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    if 'FATES_NPP_PPE_V1' not in df.columns or lai_col not in df.columns:
        print(f"\nWARNING: Missing required columns for {region}")
        continue
    
    print(f"\nCreating plots for {region.replace('_', ' ').title()}")
    
    # Create 2x2 subplot for 4 parameters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'NPP(1850) and {lai_label}(2000) vs Parameters - {region.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if param not in df.columns:
            ax.text(0.5, 0.5, f'{param}\nNot found', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Create twin axis for two different scales
        ax2 = ax.twinx()
        
        # Plot NPP(1850) on left axis (blue)
        scatter1 = ax.scatter(df[param], df['FATES_NPP_PPE_V1'], s=100, alpha=0.6, 
                             c='blue', edgecolors='black', linewidth=0.5, label='NPP(1850)')
        
        # Plot LAI(2000) on right axis (red)
        scatter2 = ax2.scatter(df[param], df[lai_col], s=100, alpha=0.6, 
                              c='red', marker='s', edgecolors='black', linewidth=0.5, label=f'{lai_label}(2000)')
        
        # Add observed LAI as horizontal line on right axis
        if region in obs_lai_regional and not np.isnan(obs_lai_regional[region]):
            if region == 'ENTs':
                ax2.axhline(obs_lai_regional[region], color='darkgreen', linestyle='--', 
                           linewidth=2, alpha=0.8, label=f'MODIS Summer {lai_label}: {obs_lai_regional[region]:.2f}')
            else:
                ax2.axhline(obs_lai_regional[region], color='darkgreen', linestyle='--', 
                           linewidth=2, alpha=0.8, label=f'MODIS {lai_label}: {obs_lai_regional[region]:.2f}')
        if region == 'ENTs' and region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
            ax2.axhline(obs_lai_regional_mean_ent[region], color='darkblue', linestyle='-.', 
                       linewidth=2, alpha=0.8, label=f'MODIS Mean {lai_label}: {obs_lai_regional_mean_ent[region]:.2f}')
        if region in obs_lai_regional_avhrr and not np.isnan(obs_lai_regional_avhrr[region]):
            ax2.axhline(obs_lai_regional_avhrr[region], color='limegreen', linestyle=':', 
                       linewidth=2, alpha=0.8, label=f'AVHRR {lai_label}: {obs_lai_regional_avhrr[region]:.2f}')
        
        # Add ensemble member labels (for NPP only to avoid clutter)
        for _, row_data in df.iterrows():
            ax.annotate(f"n{int(row_data['ensemble_member']):02d}", 
                       (row_data[param], row_data['FATES_NPP_PPE_V1']),
                       fontsize=7, ha='center', va='bottom', color='blue')
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_ylabel('NPP(1850)', fontsize=11, fontweight='bold', color='blue')
        ax2.set_ylabel(f'{lai_label}(2000)', fontsize=11, fontweight='bold', color='red')
        
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combine legends - include MODIS and AVHRR LAI if available
        legend_handles = [scatter1, scatter2]
        legend_labels = ['NPP(1850)', f'{lai_label}(2000)']
        from matplotlib.lines import Line2D
        if region in obs_lai_regional and not np.isnan(obs_lai_regional[region]):
            modis_line = Line2D([0], [0], color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
            legend_handles.append(modis_line)
            if region == 'ENTs':
                legend_labels.append(f'MODIS Summer {lai_label}: {obs_lai_regional[region]:.2f}')
            else:
                legend_labels.append(f'MODIS {lai_label}: {obs_lai_regional[region]:.2f}')
        if region == 'ENTs' and region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
            modis_mean_line = Line2D([0], [0], color='darkblue', linestyle='-.', linewidth=2, alpha=0.8)
            legend_handles.append(modis_mean_line)
            legend_labels.append(f'MODIS Mean {lai_label}: {obs_lai_regional_mean_ent[region]:.2f}')
        if region in obs_lai_regional_avhrr and not np.isnan(obs_lai_regional_avhrr[region]):
            avhrr_line = Line2D([0], [0], color='limegreen', linestyle=':', linewidth=2, alpha=0.8)
            legend_handles.append(avhrr_line)
            legend_labels.append(f'AVHRR {lai_label}: {obs_lai_regional_avhrr[region]:.2f}')
        ax.legend(legend_handles, legend_labels, loc='best', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_NPP_LAI / f"npp_and_lai_vs_parameters_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Create plots of GPP(2000) vs parameters with observational data and emulator
print("\n" + "="*60)
print("Creating GPP(2000) vs parameter plots with FLUXCOM observations and emulator...")
print("="*60)

params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    if 'FATES_GPP_PPE_2000_V1' not in df.columns:
        print(f"\nWARNING: Missing FATES_GPP_PPE_2000_V1 column for {region}")
        continue
    
    print(f"\nCreating GPP plots for {region.replace('_', ' ').title()}")
    
    # Clean data - remove rows with NaN in parameters or GPP
    df_clean = df.dropna(subset=params + ['FATES_GPP_PPE_2000_V1'])
    
    if len(df_clean) < 3:
        print(f"  WARNING: Not enough clean data points ({len(df_clean)}) for emulator. Skipping.")
        continue
    
    # Prepare training data for emulator
    X = df_clean[params].values
    y = df_clean['FATES_GPP_PPE_2000_V1'].values
    
    # Train Linear Regression emulator
    print(f"  Training emulator on {len(df_clean)} ensemble members...")
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Evaluate emulator skill
    y_pred_train = lr.predict(X)
    r2 = 1 - np.sum((y - y_pred_train)**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean((y - y_pred_train)**2))
    print(f"  Emulator R²: {r2:.4f}")
    print(f"  Emulator RMSE: {rmse:.4f}")
    
    # Generate Latin Hypercube samples for emulator predictions
    n_samples = 1000
    sampler = qmc.LatinHypercube(d=len(params))
    sample = sampler.random(n=n_samples)
    
    # Scale samples to parameter ranges
    param_mins = X.min(axis=0)
    param_maxs = X.max(axis=0)
    lhs_samples = qmc.scale(sample, param_mins, param_maxs)
    
    # Predict GPP for LHS samples
    y_lhs_pred = lr.predict(lhs_samples)
    
    print(f"  GPP range: {y.min():.4f} to {y.max():.4f}")
    print(f"  LHS predicted GPP range: {y_lhs_pred.min():.4f} to {y_lhs_pred.max():.4f}")
    
    # Store parameter samples for plotting
    param_samples = {}
    param_predictions = {}
    
    for i, param in enumerate(params):
        param_samples[param] = lhs_samples[:, i]
        param_predictions[param] = y_lhs_pred
    
    # Create 2x2 subplot for 4 parameters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'GPP(2000) vs Parameters - {region.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if param not in df.columns:
            ax.text(0.5, 0.5, f'{param}\nNot found', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Plot emulator predictions as small grey dots in background
        ax.scatter(param_samples[param], param_predictions[param], s=10, alpha=0.3, 
                  c='grey', edgecolors='none', zorder=1, label='Emulator predictions')
        
        # Plot GPP(2000) (blue points) on top
        scatter = ax.scatter(df[param], df['FATES_GPP_PPE_2000_V1'], s=100, alpha=0.6, 
                           c='blue', edgecolors='black', linewidth=0.5, zorder=2, label='Model GPP(2000)')
        
        # Add observed GPP as horizontal line
        if region in obs_gpp_regional and not np.isnan(obs_gpp_regional[region]):
            ax.axhline(obs_gpp_regional[region], color='darkred', linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=3, label=f'FLUXCOM GPP: {obs_gpp_regional[region]:.2f}')
        if region in obs_gpp_regional_wecann and not np.isnan(obs_gpp_regional_wecann[region]):
            ax.axhline(obs_gpp_regional_wecann[region], color='coral', linestyle=':', 
                      linewidth=2, alpha=0.8, zorder=3, label=f'WECANN GPP: {obs_gpp_regional_wecann[region]:.2f}')
        
        # Add ensemble member labels
        for _, row_data in df.iterrows():
            ax.annotate(f"n{int(row_data['ensemble_member']):02d}", 
                       (row_data[param], row_data['FATES_GPP_PPE_2000_V1']),
                       fontsize=7, ha='center', va='bottom', color='blue')
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_ylabel('GPP(2000) [gC m⁻² day⁻¹]', fontsize=11, fontweight='bold', color='blue')
        
        # Create legend
        legend_handles = [scatter]
        legend_labels = ['Model GPP(2000)']
        from matplotlib.lines import Line2D
        if region in obs_gpp_regional and not np.isnan(obs_gpp_regional[region]):
            fluxcom_line = Line2D([0], [0], color='darkred', linestyle='--', linewidth=2, alpha=0.8)
            legend_handles.append(fluxcom_line)
            legend_labels.append(f'FLUXCOM GPP: {obs_gpp_regional[region]:.2f}')
        if region in obs_gpp_regional_wecann and not np.isnan(obs_gpp_regional_wecann[region]):
            wecann_line = Line2D([0], [0], color='coral', linestyle=':', linewidth=2, alpha=0.8)
            legend_handles.append(wecann_line)
            legend_labels.append(f'WECANN GPP: {obs_gpp_regional_wecann[region]:.2f}')
        ax.legend(legend_handles, legend_labels, loc='best', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_GPP / f"gpp_vs_parameters_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Create plots of NPP/LAI ratio vs parameters
print("\n" + "="*60)
print("Creating NPP/LAI ratio vs parameter plots...")
print("="*60)

params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    if 'FATES_NPP_PPE_V1' not in df.columns or lai_col not in df.columns:
        print(f"\nWARNING: Missing required columns for {region}")
        continue
    
    # Calculate ratio
    df['NPP_LAI_ratio'] = df['FATES_NPP_PPE_V1'] / df[lai_col]
    
    print(f"\nCreating ratio plots for {region.replace('_', ' ').title()}")
    if region == 'ENTs':
        print(f"  Using summer LAI for NPP/LAI ratio calculation")
    
    # Check data availability
    print(f"\n  DATA CHECK:")
    print(f"  Total rows in CSV: {len(df)}")
    print(f"  Ensemble members: {sorted(df['ensemble_member'].values)}")
    
    # Check for NaN values
    print(f"\n  Checking for missing values:")
    for param in params:
        n_missing = df[param].isna().sum()
        print(f"    {param}: {n_missing} missing")
    print(f"    FATES_NPP_PPE_V1: {df['FATES_NPP_PPE_V1'].isna().sum()} missing")
    print(f"    {lai_col}: {df[lai_col].isna().sum()} missing")
    
    # Remove any rows with NaN
    df_clean = df.dropna(subset=params + ['FATES_NPP_PPE_V1', lai_col])
    print(f"\n  After removing NaN: {len(df_clean)} rows")
    
    if len(df_clean) < 3:
        print(f"\n  WARNING: Only {len(df_clean)} training points! Need at least 3 for meaningful emulation.")
        print(f"  Skipping emulator for this region.")
        continue
    
    # Train Linear Regression model
    print(f"\n  Training Linear Regression model...")
    print(f"  Number of training points: {len(df_clean)}")
    
    X = df_clean[params].values
    y = df_clean['NPP_LAI_ratio'].values
    
    print(f"\n  Parameter ranges:")
    for i, param in enumerate(params):
        print(f"    {param}: [{X[:, i].min():.6f}, {X[:, i].max():.6f}], std={X[:, i].std():.6f}")
    
    print(f"\n  Target (ratio) statistics:")
    print(f"    Range: {y.min():.4f} to {y.max():.4f}")
    print(f"    Mean: {y.mean():.4f}, Std: {y.std():.4f}")
    
    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Check training skill
    y_pred_train = lr.predict(X)
    train_score = lr.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred_train)**2))
    print(f"\n  Linear Regression Results:")
    print(f"    R² score: {train_score:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    Intercept: {lr.intercept_:.4f}")
    print(f"\n  Coefficients:")
    for i, param in enumerate(params):
        print(f"    {param}: {lr.coef_[i]:.6f}")
    
    # Create skill plot
    fig_skill, ax_skill = plt.subplots(figsize=(8, 8))
    ax_skill.scatter(y, y_pred_train, s=100, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add 1:1 line
    min_val = min(y.min(), y_pred_train.min())
    max_val = max(y.max(), y_pred_train.max())
    ax_skill.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    
    # Add ensemble labels
    for i, (obs, pred) in enumerate(zip(y, y_pred_train)):
        ax_skill.annotate(f"n{int(df_clean.iloc[i]['ensemble_member']):02d}", 
                         (obs, pred), fontsize=8, ha='center', va='bottom')
    
    ax_skill.text(0.05, 0.95, f'R² = {train_score:.3f}\nRMSE = {rmse:.3f}', 
                 transform=ax_skill.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_skill.set_xlabel('Observed NPP(1850)/LAI(2000)', fontsize=12, fontweight='bold')
    ax_skill.set_ylabel('Predicted NPP(1850)/LAI(2000)', fontsize=12, fontweight='bold')
    ax_skill.set_title(f'Emulator Skill - {region.replace("_", " ").title()}', 
                      fontsize=14, fontweight='bold')
    ax_skill.legend()
    ax_skill.grid(True, alpha=0.3)
    ax_skill.set_aspect('equal')
    
    output_file_skill = OUTPUT_DIR_NPP_LAI / f"emulator_skill_{region}.png"
    fig_skill.savefig(output_file_skill, dpi=300, bbox_inches='tight')
    print(f"  Saved emulator skill plot: {output_file_skill}")
    plt.close(fig_skill)
    
    # Generate Latin Hypercube samples
    print(f"  Generating Latin Hypercube samples...")
    n_samples = 2000
    
    # Get parameter bounds
    param_bounds = np.array([[df_clean[param].min(), df_clean[param].max()] for param in params])
    
    # Generate LHS samples in [0,1]^d
    sampler = qmc.LatinHypercube(d=len(params), seed=42)
    lhs_samples_unit = sampler.random(n=n_samples)
    
    # Scale to parameter bounds
    lhs_samples = qmc.scale(lhs_samples_unit, param_bounds[:, 0], param_bounds[:, 1])
    
    # Predict for LHS samples
    y_lhs_pred = lr.predict(lhs_samples)
    
    print(f"\n  LHS predicted ratio range: {y_lhs_pred.min():.4f} to {y_lhs_pred.max():.4f}")
    
    # Store samples for each parameter
    param_samples = {}
    param_predictions = {}
    
    for i, param in enumerate(params):
        param_samples[param] = lhs_samples[:, i]
        param_predictions[param] = y_lhs_pred
    
    # Verify emulator logic
    print(f"\n  EMULATOR SUMMARY:")
    print(f"    Using Linear Regression")
    print(f"    Model: ratio = {lr.intercept_:.4f} + " + " + ".join([f"{lr.coef_[i]:.4f}*{params[i].replace('fates_', '')}" for i in range(len(params))]))
    print(f"    Training on {len(df_clean)} actual ensemble members")
    print(f"    Generated {n_samples} Latin Hypercube samples")
    print(f"    Grey dots show predicted ratio for LHS samples")
    
    if len(df_clean) < 5:
        print(f"\n  NOTE: With only {len(df_clean)} training points, emulator skill may be limited.")
        print(f"        Consider running more ensemble members for better emulation.")
    
    # Create 2D parameter space plots: d2bl1 vs l2fr colored by NPP, LAI, and ratio
    print(f"\n  Creating 2D parameter space visualization...")
    
    d2bl1_idx = params.index('fates_allom_d2bl1')
    l2fr_idx = params.index('fates_allom_l2fr')
    
    # Create figure with 3 subplots
    fig_2d, axes_2d = plt.subplots(1, 3, figsize=(22, 6))
    
    # Get NPP and LAI values from the dataframe
    npp_values = df_clean['FATES_NPP_PPE_V1'].values
    lai_values = df_clean[lai_col].values
    
    # Train emulators for NPP and LAI
    lr_npp = LinearRegression()
    lr_npp.fit(X, npp_values)
    npp_lhs_pred = lr_npp.predict(lhs_samples)
    
    lr_lai = LinearRegression()
    lr_lai.fit(X, lai_values)
    lai_lhs_pred = lr_lai.predict(lhs_samples)
    
    # Plot 1: NPP(1850)
    scatter_npp = axes_2d[0].scatter(lhs_samples[:, d2bl1_idx], lhs_samples[:, l2fr_idx],
                                     c=npp_lhs_pred, s=20, alpha=0.4, cmap='YlOrRd', 
                                     edgecolors='none')
    axes_2d[0].scatter(X[:, d2bl1_idx], X[:, l2fr_idx],
                       c=npp_values, s=150, alpha=0.9, cmap='YlOrRd',
                       edgecolors='black', linewidth=2, marker='o')
    for i in range(len(X)):
        axes_2d[0].annotate(f"n{int(df_clean.iloc[i]['ensemble_member']):02d}",
                           (X[i, d2bl1_idx], X[i, l2fr_idx]),
                           fontsize=8, ha='center', va='bottom', fontweight='bold')
    cbar_npp = plt.colorbar(scatter_npp, ax=axes_2d[0])
    cbar_npp.set_label('NPP(1850) [gC/m²/yr]', fontsize=11, fontweight='bold')
    axes_2d[0].set_xlabel('fates_allom_d2bl1', fontsize=11, fontweight='bold')
    axes_2d[0].set_ylabel('fates_allom_l2fr', fontsize=11, fontweight='bold')
    axes_2d[0].set_title('NPP(1850)', fontsize=12, fontweight='bold')
    axes_2d[0].grid(True, alpha=0.3)
    
    # Plot 2: LAI(2000)
    scatter_lai = axes_2d[1].scatter(lhs_samples[:, d2bl1_idx], lhs_samples[:, l2fr_idx],
                                     c=lai_lhs_pred, s=20, alpha=0.4, cmap='YlGn', 
                                     edgecolors='none')
    axes_2d[1].scatter(X[:, d2bl1_idx], X[:, l2fr_idx],
                       c=lai_values, s=150, alpha=0.9, cmap='YlGn',
                       edgecolors='black', linewidth=2, marker='o')
    for i in range(len(X)):
        axes_2d[1].annotate(f"n{int(df_clean.iloc[i]['ensemble_member']):02d}",
                           (X[i, d2bl1_idx], X[i, l2fr_idx]),
                           fontsize=8, ha='center', va='bottom', fontweight='bold')
    cbar_lai = plt.colorbar(scatter_lai, ax=axes_2d[1])
    cbar_lai.set_label(f'{lai_label}(2000) [m²/m²]', fontsize=11, fontweight='bold')
    axes_2d[1].set_xlabel('fates_allom_d2bl1', fontsize=11, fontweight='bold')
    axes_2d[1].set_ylabel('fates_allom_l2fr', fontsize=11, fontweight='bold')
    axes_2d[1].set_title(f'{lai_label}(2000)', fontsize=12, fontweight='bold')
    axes_2d[1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio NPP(1850)/LAI(2000)
    scatter_ratio = axes_2d[2].scatter(lhs_samples[:, d2bl1_idx], lhs_samples[:, l2fr_idx],
                                       c=y_lhs_pred, s=20, alpha=0.4, cmap='viridis', 
                                       edgecolors='none')
    axes_2d[2].scatter(X[:, d2bl1_idx], X[:, l2fr_idx],
                       c=y, s=150, alpha=0.9, cmap='viridis',
                       edgecolors='black', linewidth=2, marker='o')
    for i in range(len(X)):
        axes_2d[2].annotate(f"n{int(df_clean.iloc[i]['ensemble_member']):02d}",
                           (X[i, d2bl1_idx], X[i, l2fr_idx]),
                           fontsize=8, ha='center', va='bottom', fontweight='bold')
    cbar_ratio = plt.colorbar(scatter_ratio, ax=axes_2d[2])
    cbar_ratio.set_label(f'NPP(1850)/{lai_label}(2000) Ratio', fontsize=11, fontweight='bold')
    axes_2d[2].set_xlabel('fates_allom_d2bl1', fontsize=11, fontweight='bold')
    axes_2d[2].set_ylabel('fates_allom_l2fr', fontsize=11, fontweight='bold')
    axes_2d[2].set_title(f'NPP(1850)/{lai_label}(2000)', fontsize=12, fontweight='bold')
    axes_2d[2].grid(True, alpha=0.3)
    
    # Add overall title
    fig_2d.suptitle(f'Parameter Space: d2bl1 vs l2fr - {region.replace("_", " ").title()}',
                   fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    output_file_2d = OUTPUT_DIR_NPP_LAI / f"parameter_space_d2bl1_l2fr_{region}.png"
    fig_2d.savefig(output_file_2d, dpi=300, bbox_inches='tight')
    print(f"  Saved 2D parameter space plot: {output_file_2d}")
    plt.close(fig_2d)
    
    # Create 2x2 subplot for 4 parameters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'NPP(1850)/LAI(2000) Ratio vs Parameters - {region.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if param not in df.columns:
            ax.text(0.5, 0.5, f'{param}\nNot found', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Plot GP emulator samples as small grey dots in background (marginal effect)
        ax.scatter(param_samples[param], param_predictions[param], s=10, alpha=0.3, c='grey', edgecolors='none', zorder=1)
        
        # Scatter plot of actual data on top
        ax.scatter(df[param], df['NPP_LAI_ratio'], s=150, alpha=0.8, 
                  c='purple', edgecolors='black', linewidth=1, zorder=2)
        
        # Add ensemble member labels
        for _, row_data in df.iterrows():
            ax.annotate(f"n{int(row_data['ensemble_member']):02d}", 
                       (row_data[param], row_data['NPP_LAI_ratio']),
                       fontsize=9, ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_ylabel('NPP(1850)/LAI(2000) Ratio', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_NPP_LAI / f"npp_lai_ratio_vs_parameters_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Create VEGC(2000) vs LAI(2000) scatter plots
print("\n" + "="*60)
print("Creating VEGC(2000) vs LAI(2000) plots...")
print("="*60)

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    if 'FATES_VEGC_PPE_2000_V1' not in df.columns or lai_col not in df.columns:
        print(f"\nWARNING: Missing required columns for {region}")
        continue
    
    print(f"\nCreating plot for {region.replace('_', ' ').title()}")
    
    # Create scatter plot: VEGC_2000 vs LAI_2000
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all members with ensemble numbers as labels
    scatter = ax.scatter(df['FATES_VEGC_PPE_2000_V1'], df[lai_col],
                        s=100, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
    
    # Add ensemble member labels
    for idx, row in df.iterrows():
        ax.annotate(f"n{int(row['ensemble_member']):02d}", 
                   (row['FATES_VEGC_PPE_2000_V1'], row[lai_col]),
                   fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('VEGC(2000) [kgC/m²]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{lai_label}(2000) [m²/m²]', fontsize=12, fontweight='bold')
    ax.set_title(f'VEGC(2000) vs {lai_label}(2000) ({YEAR_RANGE}) - {region.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_file = OUTPUT_DIR_VEGC_LAI / f"vegc_vs_lai_2000_{YEAR_RANGE}_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Create plots of VEGC(2000) and LAI(2000) vs parameters (on same axes)
print("\n" + "="*60)
print("Creating VEGC(2000) and LAI(2000) vs parameters plots...")
print("="*60)

params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    if 'FATES_VEGC_PPE_2000_V1' not in df.columns or lai_col not in df.columns:
        print(f"\nWARNING: Missing required columns for {region}")
        continue
    
    print(f"\nCreating dual-axis plots for {region.replace('_', ' ').title()}")
    
    # Create 2x2 subplot for 4 parameters with dual y-axes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'VEGC(2000) and {lai_label}(2000) vs Parameters ({YEAR_RANGE}) - {region.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax1 = axes[row, col]
        
        if param not in df.columns:
            ax1.text(0.5, 0.5, f'{param}\nNot found', ha='center', va='center', 
                   transform=ax1.transAxes)
            continue
        
        # Plot VEGC on left y-axis
        color_vegc = 'tab:orange'
        ax1.scatter(df[param], df['FATES_VEGC_PPE_2000_V1'], 
                   color=color_vegc, s=80, alpha=0.7, marker='o', label='VEGC(2000)')
        ax1.set_xlabel(param.replace('fates_', ''), fontsize=11, fontweight='bold')
        ax1.set_ylabel('VEGC(2000) [kgC/m²]', color=color_vegc, fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_vegc)
        
        # Create second y-axis for LAI
        ax2 = ax1.twinx()
        color_lai = 'tab:green'
        ax2.scatter(df[param], df[lai_col], 
                   color=color_lai, s=80, alpha=0.7, marker='s', label=f'{lai_label}(2000)')
        ax2.set_ylabel(f'{lai_label}(2000) [m²/m²]', color=color_lai, fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_lai)
        
        # Add observed LAI as horizontal line
        if region in obs_lai_regional and not np.isnan(obs_lai_regional[region]):
            ax2.axhline(obs_lai_regional[region], color='darkgreen', linestyle='--', 
                       linewidth=2, alpha=0.8, label=f'MODIS {lai_label}: {obs_lai_regional[region]:.2f}')
        if region in obs_lai_regional_avhrr and not np.isnan(obs_lai_regional_avhrr[region]):
            ax2.axhline(obs_lai_regional_avhrr[region], color='limegreen', linestyle=':', 
                       linewidth=2, alpha=0.8, label=f'AVHRR {lai_label}: {obs_lai_regional_avhrr[region]:.2f}')
        
        # Add ensemble member labels
        for idx_member, row in df.iterrows():
            # Use VEGC y-values for label positioning
            ax1.annotate(f"n{int(row['ensemble_member']):02d}",
                        (row[param], row['FATES_VEGC_PPE_2000_V1']),
                        xytext=(0, 8), textcoords='offset points',
                        fontsize=9, ha='center', va='bottom', fontweight='bold',
                        color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                                 edgecolor='none', alpha=0.7))
        
        ax1.grid(True, alpha=0.3)
        
        # Combined legend (using ax2 to get all elements including the horizontal line)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_VEGC_LAI / f"vegc_and_lai_vs_parameters_{YEAR_RANGE}_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Create plots of VEGC(2000)/LAI(2000) ratio vs parameters
print("\n" + "="*60)
print(f"Creating VEGC(2000)/LAI(2000) ratio vs parameter plots ({YEAR_RANGE})...")
print("="*60)

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    # Check if VEGC columns exist for 2000 ensemble
    if 'FATES_VEGC_PPE_2000_V1' not in df.columns or lai_col not in df.columns:
        print(f"\nWARNING: Missing VEGC or LAI columns for 2000 ensemble in {region}")
        continue
    
    # Calculate ratio for 2000 ensemble
    df['VEGC_LAI_ratio_2000'] = df['FATES_VEGC_PPE_2000_V1'] / df[lai_col]
    
    print(f"\nCreating VEGC(2000)/{lai_label}(2000) ratio plots for {region.replace('_', ' ').title()}")
    
    # Remove any rows with NaN
    df_clean_vegc = df.dropna(subset=params + ['FATES_VEGC_PPE_2000_V1', lai_col])
    print(f"  After removing NaN: {len(df_clean_vegc)} rows")
    
    if len(df_clean_vegc) < 3:
        print(f"\n  WARNING: Only {len(df_clean_vegc)} training points! Skipping.")
        continue
    
    X_vegc = df_clean_vegc[params].values
    y_vegc_2000 = df_clean_vegc['VEGC_LAI_ratio_2000'].values
    
    # Train Linear Regression model for ratio
    lr_ratio = LinearRegression()
    lr_ratio.fit(X_vegc, y_vegc_2000)
    
    # Check training skill
    y_pred_train = lr_ratio.predict(X_vegc)
    train_score = lr_ratio.score(X_vegc, y_vegc_2000)
    rmse = np.sqrt(np.mean((y_vegc_2000 - y_pred_train)**2))
    
    print(f"\n  Linear Regression Results:")
    print(f"    R² score: {train_score:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    
    # Create emulator skill plot
    fig_skill, ax_skill = plt.subplots(figsize=(8, 8))
    
    ax_skill.scatter(y_vegc_2000, y_pred_train, s=100, alpha=0.7, c='blue', edgecolors='black')
    
    # Add ensemble member labels
    for i in range(len(y_vegc_2000)):
        ax_skill.annotate(f"n{int(df_clean_vegc.iloc[i]['ensemble_member']):02d}",
                         (y_vegc_2000[i], y_pred_train[i]),
                         fontsize=8, ha='center', va='bottom')
    
    # Add 1:1 line
    min_val = min(y_vegc_2000.min(), y_pred_train.min())
    max_val = max(y_vegc_2000.max(), y_pred_train.max())
    ax_skill.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    
    ax_skill.set_xlabel('Observed VEGC(2000)/LAI(2000)', fontsize=12, fontweight='bold')
    ax_skill.set_ylabel('Predicted VEGC(2000)/LAI(2000)', fontsize=12, fontweight='bold')
    ax_skill.set_title(f'Emulator Skill ({YEAR_RANGE}) - {region.replace("_", " ").title()}', 
                      fontsize=14, fontweight='bold')
    ax_skill.legend()
    ax_skill.grid(True, alpha=0.3)
    ax_skill.set_aspect('equal')
    
    output_file_skill = OUTPUT_DIR_VEGC_LAI / f"emulator_skill_{YEAR_RANGE}_{region}.png"
    fig_skill.savefig(output_file_skill, dpi=300, bbox_inches='tight')
    print(f"  Saved emulator skill plot: {output_file_skill}")
    plt.close(fig_skill)
    
    # Generate LHS samples
    param_bounds = np.array([[df_clean_vegc[param].min(), df_clean_vegc[param].max()] for param in params])
    sampler_vegc = qmc.LatinHypercube(d=len(params), seed=43)
    lhs_samples_unit = sampler_vegc.random(n=2000)
    lhs_samples_vegc = qmc.scale(lhs_samples_unit, param_bounds[:, 0], param_bounds[:, 1])
    
    # Predict for LHS samples
    y_vegc_lhs_pred = lr_ratio.predict(lhs_samples_vegc)
    
    print(f"\n  VEGC(2000)/LAI(2000) Ratio Statistics:")
    print(f"    Range: {y_vegc_2000.min():.4f} to {y_vegc_2000.max():.4f}")
    print(f"    LHS predicted range: {y_vegc_lhs_pred.min():.4f} to {y_vegc_lhs_pred.max():.4f}")
    
    # Train emulators for VEGC and LAI separately
    vegc_values = df_clean_vegc['FATES_VEGC_PPE_2000_V1'].values
    lai_values = df_clean_vegc[lai_col].values
    
    lr_vegc = LinearRegression()
    lr_vegc.fit(X_vegc, vegc_values)
    vegc_lhs_pred = lr_vegc.predict(lhs_samples_vegc)
    
    lr_lai = LinearRegression()
    lr_lai.fit(X_vegc, lai_values)
    lai_lhs_pred = lr_lai.predict(lhs_samples_vegc)
    
    # Create 2D parameter space plots: d2bl1 vs l2fr colored by VEGC, LAI, and ratio
    d2bl1_idx = params.index('fates_allom_d2bl1')
    l2fr_idx = params.index('fates_allom_l2fr')
    
    # Create figure with 3 subplots
    fig_2d, axes_2d = plt.subplots(1, 3, figsize=(22, 6))
    
    # Plot 1: VEGC(2000)
    scatter_vegc = axes_2d[0].scatter(lhs_samples_vegc[:, d2bl1_idx], lhs_samples_vegc[:, l2fr_idx],
                                     c=vegc_lhs_pred, s=20, alpha=0.4, cmap='Oranges', 
                                     edgecolors='none')
    axes_2d[0].scatter(X_vegc[:, d2bl1_idx], X_vegc[:, l2fr_idx],
                       c=vegc_values, s=150, alpha=0.9, cmap='Oranges',
                       edgecolors='black', linewidth=2, marker='o')
    for i in range(len(X_vegc)):
        axes_2d[0].annotate(f"n{int(df_clean_vegc.iloc[i]['ensemble_member']):02d}",
                           (X_vegc[i, d2bl1_idx], X_vegc[i, l2fr_idx]),
                           fontsize=8, ha='center', va='bottom', fontweight='bold')
    cbar_vegc = plt.colorbar(scatter_vegc, ax=axes_2d[0])
    cbar_vegc.set_label('VEGC(2000) [kgC/m²]', fontsize=11, fontweight='bold')
    axes_2d[0].set_xlabel('fates_allom_d2bl1', fontsize=11, fontweight='bold')
    axes_2d[0].set_ylabel('fates_allom_l2fr', fontsize=11, fontweight='bold')
    axes_2d[0].set_title('VEGC(2000)', fontsize=12, fontweight='bold')
    axes_2d[0].grid(True, alpha=0.3)
    
    # Plot 2: LAI(2000)
    scatter_lai = axes_2d[1].scatter(lhs_samples_vegc[:, d2bl1_idx], lhs_samples_vegc[:, l2fr_idx],
                                     c=lai_lhs_pred, s=20, alpha=0.4, cmap='YlGn', 
                                     edgecolors='none')
    axes_2d[1].scatter(X_vegc[:, d2bl1_idx], X_vegc[:, l2fr_idx],
                       c=lai_values, s=150, alpha=0.9, cmap='YlGn',
                       edgecolors='black', linewidth=2, marker='o')
    for i in range(len(X_vegc)):
        axes_2d[1].annotate(f"n{int(df_clean_vegc.iloc[i]['ensemble_member']):02d}",
                           (X_vegc[i, d2bl1_idx], X_vegc[i, l2fr_idx]),
                           fontsize=8, ha='center', va='bottom', fontweight='bold')
    cbar_lai = plt.colorbar(scatter_lai, ax=axes_2d[1])
    cbar_lai.set_label('LAI(2000) [m²/m²]', fontsize=11, fontweight='bold')
    axes_2d[1].set_xlabel('fates_allom_d2bl1', fontsize=11, fontweight='bold')
    axes_2d[1].set_ylabel('fates_allom_l2fr', fontsize=11, fontweight='bold')
    axes_2d[1].set_title('LAI(2000)', fontsize=12, fontweight='bold')
    axes_2d[1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio VEGC(2000)/LAI(2000)
    scatter_ratio = axes_2d[2].scatter(lhs_samples_vegc[:, d2bl1_idx], lhs_samples_vegc[:, l2fr_idx],
                                       c=y_vegc_lhs_pred, s=20, alpha=0.4, cmap='plasma', 
                                       edgecolors='none')
    axes_2d[2].scatter(X_vegc[:, d2bl1_idx], X_vegc[:, l2fr_idx],
                       c=y_vegc_2000, s=150, alpha=0.9, cmap='plasma',
                       edgecolors='black', linewidth=2, marker='o')
    for i in range(len(X_vegc)):
        axes_2d[2].annotate(f"n{int(df_clean_vegc.iloc[i]['ensemble_member']):02d}",
                           (X_vegc[i, d2bl1_idx], X_vegc[i, l2fr_idx]),
                           fontsize=8, ha='center', va='bottom', fontweight='bold')
    cbar_ratio = plt.colorbar(scatter_ratio, ax=axes_2d[2])
    cbar_ratio.set_label('VEGC(2000)/LAI(2000) Ratio', fontsize=11, fontweight='bold')
    axes_2d[2].set_xlabel('fates_allom_d2bl1', fontsize=11, fontweight='bold')
    axes_2d[2].set_ylabel('fates_allom_l2fr', fontsize=11, fontweight='bold')
    axes_2d[2].set_title('VEGC(2000)/LAI(2000)', fontsize=12, fontweight='bold')
    axes_2d[2].grid(True, alpha=0.3)
    
    # Add overall title
    fig_2d.suptitle(f'Parameter Space: d2bl1 vs l2fr ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                   fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    output_file_2d = OUTPUT_DIR_VEGC_LAI / f"parameter_space_d2bl1_l2fr_{YEAR_RANGE}_{region}.png"
    fig_2d.savefig(output_file_2d, dpi=300, bbox_inches='tight')
    print(f"  Saved 2D parameter space plot: {output_file_2d}")
    plt.close(fig_2d)
    
    # Create 2x2 subplot for 4 parameters (ratio vs parameters)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'VEGC(2000)/LAI(2000) Ratio vs Parameters ({YEAR_RANGE}) - {region.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if param not in df.columns:
            ax.text(0.5, 0.5, f'{param}\nNot found', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Overlay LHS predictions colored by predicted VEGC
        param_idx = params.index(param)
        scatter = ax.scatter(lhs_samples_vegc[:, param_idx], y_vegc_lhs_pred, s=20, alpha=0.4, 
                            c=vegc_lhs_pred, cmap='RdBu_r',
                            edgecolors='none', label='LHS predictions')
        
        # Plot actual data colored by VEGC
        ax.scatter(df_clean_vegc[param], y_vegc_2000, s=150, alpha=0.9, 
                  c=vegc_values, cmap='RdBu_r',
                  edgecolors='black', linewidth=2, label='Ensemble members')
        
        # Add ensemble member labels
        for i in range(len(df_clean_vegc)):
            ax.annotate(f"n{int(df_clean_vegc.iloc[i]['ensemble_member']):02d}",
                       (df_clean_vegc.iloc[i][param], y_vegc_2000[i]),
                       xytext=(0, 8), textcoords='offset points',
                       fontsize=10, ha='center', va='bottom', fontweight='bold',
                       color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                                edgecolor='none', alpha=0.7))
        
        ax.set_xlabel(param.replace('fates_', ''), fontsize=11, fontweight='bold')
        ax.set_ylabel('VEGC(2000)/LAI(2000) Ratio', fontsize=11, fontweight='bold')
        ax.set_title(param.replace('fates_', ''), fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for VEGC
        if idx == 1:  # Add colorbar to upper right subplot
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('VEGC(2000) [kgC/m²]', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_VEGC_LAI / f"vegc_lai_ratio_vs_parameters_{YEAR_RANGE}_{region}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)

# Analyze each region
for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    print("\n" + "="*60)
    print(f"ANALYSIS FOR {region.upper().replace('_', ' ')}")
    print("="*60)
    
    # Summary statistics
    print("\nKey Statistics:")
    variables = ['FATES_LAI_PPE_V1', 'FATES_GPP_PPE_V1', 'FATES_NPP_PPE_V1',
                 'FATES_LAI_PPE_2000_V1', 'FATES_GPP_PPE_2000_V1', 'FATES_NPP_PPE_2000_V1']
    
    for var in variables:
        if var in df.columns:
            print(f"\n{var}:")
            print(f"  Mean: {df[var].mean():.4f}")
            print(f"  Std:  {df[var].std():.4f}")
            print(f"  Min:  {df[var].min():.4f}")
            print(f"  Max:  {df[var].max():.4f}")
    
    # Correlation between parameters and NPP_V1
    print("\n" + "-"*60)
    print("Correlations between parameters and NPP (PPE_V1):")
    params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
              'fates_allom_d2bl1', 'fates_leaf_slatop']
    
    for param in params:
        if param in df.columns and 'FATES_NPP_PPE_V1' in df.columns:
            corr = df[param].corr(df['FATES_NPP_PPE_V1'])
            print(f"  {param:<45s}: r = {corr:7.4f}")
    
    # Correlation between parameters and LAI_2000
    print("\n" + "-"*60)
    print("Correlations between parameters and LAI (PPE_2000_V1):")
    
    for param in params:
        if param in df.columns and 'FATES_LAI_PPE_2000_V1' in df.columns:
            corr = df[param].corr(df['FATES_LAI_PPE_2000_V1'])
            print(f"  {param:<45s}: r = {corr:7.4f}")
    
    # Identify interesting parameter combinations
    print("\n" + "-"*60)
    print("Members with high NPP_V1 (>75th percentile):")
    if 'FATES_NPP_PPE_V1' in df.columns:
        threshold = df['FATES_NPP_PPE_V1'].quantile(0.75)
        high_npp = df[df['FATES_NPP_PPE_V1'] >= threshold]
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Members: {list(high_npp['ensemble_member'].values)}")
        
        if len(high_npp) > 0:
            print("\n  Average parameter values for high NPP members:")
            for param in params:
                if param in df.columns:
                    avg = high_npp[param].mean()
                    overall_avg = df[param].mean()
                    print(f"    {param:<45s}: {avg:.6f} (overall: {overall_avg:.6f})")
    
    print("\n" + "-"*60)
    print("Members with low LAI_2000 (<25th percentile):")
    if 'FATES_LAI_PPE_2000_V1' in df.columns:
        threshold = df['FATES_LAI_PPE_2000_V1'].quantile(0.25)
        low_lai = df[df['FATES_LAI_PPE_2000_V1'] <= threshold]
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Members: {list(low_lai['ensemble_member'].values)}")
        
        if len(low_lai) > 0:
            print("\n  Average parameter values for low LAI members:")
            for param in params:
                if param in df.columns:
                    avg = low_lai[param].mean()
                    overall_avg = df[param].mean()
                    print(f"    {param:<45s}: {avg:.6f} (overall: {overall_avg:.6f})")

# Create comparison plots with selected members highlighted
if any(region in selected_data and len(selected_data[region]) > 0 for region in REGIONS):
    print("\n" + "="*60)
    print("Creating plots with selected members highlighted...")

    for region in REGIONS:
        if region not in data or region not in selected_data:
            continue
        
        if len(selected_data[region]) == 0:
            continue
        
        df_all = data[region]
        df_selected = selected_data[region]
        
        # Create scatter plot: NPP_V1 vs LAI_2000 with selected members highlighted
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all members
        ax.scatter(df_all['FATES_NPP_PPE_V1'], df_all['FATES_LAI_PPE_2000_V1'],
                   s=100, alpha=0.5, c='gray', label='All members', edgecolors='black', linewidth=0.5)
        
        # Highlight selected members
        ax.scatter(df_selected['FATES_NPP_PPE_V1'], df_selected['FATES_LAI_PPE_2000_V1'],
                   s=200, alpha=0.9, c='red', label='Selected (High NPP, Low LAI)', 
                   edgecolors='black', linewidth=2, marker='*')
        
        # Add labels for selected members
        for idx, row in df_selected.iterrows():
            ax.annotate(f"n{int(row['ensemble_member']):02d}", 
                       (row['FATES_NPP_PPE_V1'], row['FATES_LAI_PPE_2000_V1']),
                       fontsize=9, ha='center', va='bottom', fontweight='bold', color='red')
        
        # Add threshold lines
        npp_threshold = df_all['FATES_NPP_PPE_V1'].quantile(0.75)
        lai_threshold = df_all['FATES_LAI_PPE_2000_V1'].quantile(0.25)
        ax.axvline(npp_threshold, color='blue', linestyle='--', alpha=0.5, label=f'NPP 75th %ile')
        ax.axhline(lai_threshold, color='green', linestyle='--', alpha=0.5, label=f'LAI 25th %ile')
        
        ax.set_xlabel('NPP(1850)', fontsize=12)
        ax.set_ylabel('LAI(2000)', fontsize=12)
        ax.set_title(f'NPP(1850) vs LAI(2000) with Selected Members - {region.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        output_file = OUTPUT_DIR / f"npp_vs_lai_with_selected_{region}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved plot: {output_file}")
        plt.close(fig)

# Create LAI emulator and parameter space plots
print("\n" + "="*60)
print("Creating LAI(2000) Emulator and Parameter Space Plots...")
print("="*60)

print(f"\nDEBUG: data dictionary keys: {list(data.keys())}")
print(f"DEBUG: REGIONS list: {REGIONS}")

params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    print(f"\n  Checking region: {region}")
    if region not in data:
        print(f"    Region {region} not in data dictionary! Skipping.")
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, 'PPE_2000_V1')
    lai_label = get_lai_label(region)
    
    if lai_col not in df.columns:
        print(f"    WARNING: Missing LAI column ({lai_col}) for {region}")
        continue
    
    print(f"\n{'='*50}")
    print(f"Region: {region.replace('_', ' ').title()}")
    if region == 'ENTs':
        print(f"Using summer LAI (months 6, 7, 8)")
    print(f"{'='*50}")
    
    # Remove any rows with NaN
    df_clean = df.dropna(subset=params + [lai_col])
    print(f"  Training points: {len(df_clean)}")
    
    if len(df_clean) < 3:
        print(f"  WARNING: Only {len(df_clean)} training points! Skipping.")
        continue
    
    # Train Linear Regression model for LAI
    print(f"\n  Training LAI emulator...")
    
    X = df_clean[params].values
    y_lai = df_clean[lai_col].values
    
    lr_lai = LinearRegression()
    lr_lai.fit(X, y_lai)
    
    # Check training skill
    y_lai_pred_train = lr_lai.predict(X)
    train_score = lr_lai.score(X, y_lai)
    rmse = np.sqrt(np.mean((y_lai - y_lai_pred_train)**2))
    
    print(f"  R² score: {train_score:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  LAI range: {y_lai.min():.4f} to {y_lai.max():.4f}")
    
    # Get observational LAI for this region
    obs_lai = obs_lai_regional.get(region, np.nan)
    obs_lai_avhrr = obs_lai_regional_avhrr.get(region, np.nan)
    if not np.isnan(obs_lai):
        print(f"  MODIS LAI: {obs_lai:.4f}")
    if not np.isnan(obs_lai_avhrr):
        print(f"  AVHRR LAI: {obs_lai_avhrr:.4f}")
    
    # Create LAI emulator skill plot
    fig_skill, ax_skill = plt.subplots(figsize=(8, 8))
    ax_skill.scatter(y_lai, y_lai_pred_train, s=100, alpha=0.6, c='green', 
                    edgecolors='black', linewidth=0.5)
    
    # Add 1:1 line
    min_val = min(y_lai.min(), y_lai_pred_train.min())
    max_val = max(y_lai.max(), y_lai_pred_train.max())
    ax_skill.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    
    # Add observational LAI as horizontal and vertical lines
    if not np.isnan(obs_lai):
        if region == 'ENTs':
            ax_skill.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2, 
                            alpha=0.7, label=f'MODIS Summer {lai_label}: {obs_lai:.2f}')
            ax_skill.axvline(obs_lai, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
        else:
            ax_skill.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2, 
                            alpha=0.7, label=f'MODIS {lai_label}: {obs_lai:.2f}')
            ax_skill.axvline(obs_lai, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    if region == 'ENTs' and region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
        ax_skill.axhline(obs_lai_regional_mean_ent[region], color='darkblue', linestyle='-.', linewidth=2, 
                        alpha=0.7, label=f'MODIS Mean {lai_label}: {obs_lai_regional_mean_ent[region]:.2f}')
        ax_skill.axvline(obs_lai_regional_mean_ent[region], color='darkblue', linestyle='-.', linewidth=2, alpha=0.7)
    if not np.isnan(obs_lai_avhrr):
        ax_skill.axhline(obs_lai_avhrr, color='limegreen', linestyle=':', linewidth=2, 
                        alpha=0.7, label=f'AVHRR {lai_label}: {obs_lai_avhrr:.2f}')
        ax_skill.axvline(obs_lai_avhrr, color='limegreen', linestyle=':', linewidth=2, alpha=0.7)
    
    # Add ensemble labels
    for i, (obs, pred) in enumerate(zip(y_lai, y_lai_pred_train)):
        ax_skill.annotate(f"n{int(df_clean.iloc[i]['ensemble_member']):02d}", 
                         (obs, pred), fontsize=8, ha='center', va='bottom')
    
    ax_skill.text(0.05, 0.95, f'R² = {train_score:.3f}\nRMSE = {rmse:.3f}', 
                 transform=ax_skill.transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_skill.set_xlabel(f'Observed {lai_label}(2000)', fontsize=12, fontweight='bold')
    ax_skill.set_ylabel(f'Predicted {lai_label}(2000)', fontsize=12, fontweight='bold')
    ax_skill.set_title(f'{lai_label} Emulator Skill ({YEAR_RANGE}) - {region.replace("_", " ").title()}', 
                      fontsize=14, fontweight='bold')
    ax_skill.legend()
    ax_skill.grid(True, alpha=0.3)
    ax_skill.set_aspect('equal')
    
    output_file_skill = OUTPUT_DIR_LAI_EMULATOR / f"lai_emulator_skill_{YEAR_RANGE}_{region}.png"
    fig_skill.savefig(output_file_skill, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_skill.name}")
    plt.close(fig_skill)
    
    # Create emulator response plots: LAI vs each parameter with emulator curve
    print(f"\n  Creating LAI emulator response curves...")
    
    fig_emulator, axes_emulator = plt.subplots(2, 2, figsize=(16, 12))
    fig_emulator.suptitle(f'{lai_label}(2000) Emulator Response ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                         fontsize=16, fontweight='bold')
    
    # Calculate mean values for all parameters (for holding constant)
    param_means = {param: df_clean[param].mean() for param in params}
    
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax = axes_emulator[row, col]
        
        # Plot ensemble members
        ax.scatter(df_clean[param], y_lai, s=100, alpha=0.7, c='green',
                  edgecolors='black', linewidth=1, label='Ensemble members', zorder=3)
        
        # Generate emulator prediction curve by varying this parameter
        param_range = np.linspace(df_clean[param].min(), df_clean[param].max(), 100)
        X_pred = np.zeros((100, len(params)))
        
        # Set all parameters to their mean values
        for i, p in enumerate(params):
            X_pred[:, i] = param_means[p]
        
        # Vary the current parameter
        param_idx = params.index(param)
        X_pred[:, param_idx] = param_range
        
        # Predict LAI using emulator
        lai_pred_curve = lr_lai.predict(X_pred)
        
        # Plot emulator prediction curve
        ax.plot(param_range, lai_pred_curve, 'b-', linewidth=3, alpha=0.8,
               label='Emulator prediction', zorder=2)
        
        # Add observational LAI as horizontal line
        if not np.isnan(obs_lai):
            if region == 'ENTs':
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS Summer {lai_label}: {obs_lai:.2f}', zorder=1)
            else:
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS {lai_label}: {obs_lai:.2f}', zorder=1)
        if region == 'ENTs' and region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
            ax.axhline(obs_lai_regional_mean_ent[region], color='darkblue', linestyle='-.', linewidth=2,
                      alpha=0.8, label=f'MODIS Mean {lai_label}: {obs_lai_regional_mean_ent[region]:.2f}', zorder=1)
        if region == 'ENTs' and region in obs_lai_regional_annual_max_ent and not np.isnan(obs_lai_regional_annual_max_ent[region]):
            ax.axhline(obs_lai_regional_annual_max_ent[region], color='purple', linestyle=':', linewidth=2,
                      alpha=0.8, label=f'MODIS Annual Max {lai_label}: {obs_lai_regional_annual_max_ent[region]:.2f}', zorder=1)
        if not np.isnan(obs_lai_avhrr):
            ax.axhline(obs_lai_avhrr, color='limegreen', linestyle=':', linewidth=2,
                      alpha=0.8, label=f'AVHRR {lai_label}: {obs_lai_avhrr:.2f}', zorder=1)
        
        # Add ensemble labels
        for i, row_data in enumerate(df_clean.iterrows()):
            _, row = row_data
            ax.annotate(f"n{int(row['ensemble_member']):02d}",
                       (row[param], y_lai[i]),
                       fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{lai_label}(2000) [m²/m²]', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add text showing parameter coefficient
        coef = lr_lai.coef_[param_idx]
        ax.text(0.05, 0.95, f'Coefficient: {coef:.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_file_emulator = OUTPUT_DIR_LAI_EMULATOR / f"lai_emulator_response_{YEAR_RANGE}_{region}.png"
    fig_emulator.savefig(output_file_emulator, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_emulator.name}")
    plt.close(fig_emulator)
    
    # Generate Latin Hypercube samples
    print(f"\n  Generating Latin Hypercube samples...")
    n_samples = 2000
    
    param_bounds = np.array([[df_clean[param].min(), df_clean[param].max()] for param in params])
    
    sampler = qmc.LatinHypercube(d=len(params), seed=42)
    lhs_samples_unit = sampler.random(n=n_samples)
    lhs_samples = qmc.scale(lhs_samples_unit, param_bounds[:, 0], param_bounds[:, 1])
    
    # Predict LAI for LHS samples
    y_lai_lhs_pred = lr_lai.predict(lhs_samples)
    
    print(f"  LHS predicted LAI range: {y_lai_lhs_pred.min():.4f} to {y_lai_lhs_pred.max():.4f}")
    
    # Create 2D parameter space plot: d2bl1 vs l2fr colored by LAI
    print(f"\n  Creating parameter space visualization...")
    
    d2bl1_idx = params.index('fates_allom_d2bl1')
    l2fr_idx = params.index('fates_allom_l2fr')
    
    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    
    # Plot LHS samples (emulator predictions)
    scatter_lhs = ax_2d.scatter(lhs_samples[:, d2bl1_idx], lhs_samples[:, l2fr_idx],
                                c=y_lai_lhs_pred, s=20, alpha=0.4, cmap='YlGn',
                                edgecolors='none', vmin=y_lai.min(), vmax=y_lai.max())
    
    # Plot actual ensemble members
    scatter_actual = ax_2d.scatter(X[:, d2bl1_idx], X[:, l2fr_idx],
                                   c=y_lai, s=150, alpha=0.9, cmap='YlGn',
                                   edgecolors='black', linewidth=2, marker='o',
                                   vmin=y_lai.min(), vmax=y_lai.max())
    
    # Add ensemble labels
    for i in range(len(X)):
        ax_2d.annotate(f"n{int(df_clean.iloc[i]['ensemble_member']):02d}",
                      (X[i, d2bl1_idx], X[i, l2fr_idx]),
                      fontsize=10, ha='center', va='bottom', fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor='none', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(scatter_actual, ax=ax_2d)
    cbar.set_label(f'{lai_label}(2000) [m²/m²]', fontsize=12, fontweight='bold')
    
    # Add observational LAI as a reference line on colorbar if available
    if not np.isnan(obs_lai):
        cbar.ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=3, alpha=0.8)
        cbar.ax.text(0.5, obs_lai, f' MODIS', 
                    verticalalignment='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    if not np.isnan(obs_lai_avhrr):
        cbar.ax.axhline(obs_lai_avhrr, color='limegreen', linestyle=':', linewidth=3, alpha=0.8)
        cbar.ax.text(0.5, obs_lai_avhrr, f' AVHRR', 
                    verticalalignment='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    ax_2d.set_xlabel('fates_allom_d2bl1', fontsize=12, fontweight='bold')
    ax_2d.set_ylabel('fates_allom_l2fr', fontsize=12, fontweight='bold')
    ax_2d.set_title(f'{lai_label}(2000) Parameter Space ({YEAR_RANGE}) - {region.replace("_", " ").title()}\n' + 
                    f'Emulator R² = {train_score:.3f}', 
                    fontsize=14, fontweight='bold')
    ax_2d.grid(True, alpha=0.3)
    
    # Add text box with info
    info_text = f'Grey dots: {n_samples} LHS samples\nLarge dots: {len(X)} ensemble members'
    if not np.isnan(obs_lai):
        info_text += f'\nMODIS {lai_label}: {obs_lai:.2f} m²/m²'
    if not np.isnan(obs_lai_avhrr):
        info_text += f'\nAVHRR {lai_label}: {obs_lai_avhrr:.2f} m²/m²'
    ax_2d.text(0.02, 0.98, info_text,
              transform=ax_2d.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    output_file_2d = OUTPUT_DIR_LAI_EMULATOR / f"lai_parameter_space_{YEAR_RANGE}_{region}.png"
    fig_2d.savefig(output_file_2d, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_2d.name}")
    plt.close(fig_2d)
    
    # Create LAI vs parameters plot with emulator samples
    print(f"\n  Creating LAI vs parameters with emulator samples...")
    
    # Generate 1000 LHS samples for emulator
    n_emulator_samples = 1000
    sampler_emul = qmc.LatinHypercube(d=len(params), seed=43)
    lhs_emul_unit = sampler_emul.random(n=n_emulator_samples)
    lhs_emul = qmc.scale(lhs_emul_unit, param_bounds[:, 0], param_bounds[:, 1])
    
    # Predict LAI for these samples
    lai_emul_pred = lr_lai.predict(lhs_emul)
    
    # Get slatop values for coloring
    slatop_idx = params.index('fates_leaf_slatop')
    slatop_values = lhs_emul[:, slatop_idx]
    
    fig_lai_params, axes_lai = plt.subplots(2, 2, figsize=(16, 12))
    fig_lai_params.suptitle(f'{lai_label}(2000) vs Parameters with Emulator ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                            fontsize=16, fontweight='bold')
    
    scatter_emul = None  # Initialize outside loop
    for idx, param in enumerate(params):
        row = idx // 2
        col = idx % 2
        ax = axes_lai[row, col]
        
        param_idx = params.index(param)
        
        # Plot emulator samples (1000 LHS samples) colored by slatop
        scatter_emul = ax.scatter(lhs_emul[:, param_idx], lai_emul_pred, s=15, alpha=0.6, 
                  c=slatop_values, cmap='viridis', edgecolors='none', 
                  label=f'{n_emulator_samples} emulator samples')
        
        # Plot actual ensemble members on top
        ax.scatter(df_clean[param], y_lai, s=100, alpha=0.8, c='green',
                  edgecolors='black', linewidth=1.5, label='Ensemble members', zorder=3)
        
        # Add observational LAI as horizontal line
        if not np.isnan(obs_lai):
            if region == 'ENTs':
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS Summer {lai_label}: {obs_lai:.2f}', zorder=2)
            else:
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS {lai_label}: {obs_lai:.2f}', zorder=2)
        if region == 'ENTs' and region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
            ax.axhline(obs_lai_regional_mean_ent[region], color='darkblue', linestyle='-.', linewidth=2,
                      alpha=0.8, label=f'MODIS Mean {lai_label}: {obs_lai_regional_mean_ent[region]:.2f}', zorder=2)
        if not np.isnan(obs_lai_avhrr):
            ax.axhline(obs_lai_avhrr, color='limegreen', linestyle=':', linewidth=2,
                      alpha=0.8, label=f'AVHRR {lai_label}: {obs_lai_avhrr:.2f}', zorder=2)
        
        # Add ensemble labels
        for i, row_data in enumerate(df_clean.iterrows()):
            _, row = row_data
            ax.annotate(f"n{int(row['ensemble_member']):02d}",
                       (row[param], y_lai[i]),
                       fontsize=8, ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='none', alpha=0.7))
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), 
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{lai_label}(2000) [m²/m²]', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar for slatop values (using the last scatter_emul from the loop)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if scatter_emul is not None:
        cbar_ax = fig_lai_params.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter_emul, cax=cbar_ax)
        cbar.set_label('fates_leaf_slatop', fontsize=11, fontweight='bold')
    
    output_file_lai_params = OUTPUT_DIR_LAI_EMULATOR / f"lai_vs_params_with_emulator_{YEAR_RANGE}_{region}.png"
    fig_lai_params.savefig(output_file_lai_params, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_lai_params.name}")
    plt.close(fig_lai_params)
    
    # Create second plot colored by VEGC emulator predictions
    print(f"\n  Creating LAI vs parameters colored by VEGC emulator...")
    
    # Check if VEGC data is available
    if 'FATES_VEGC_PPE_2000_V1' not in df_clean.columns:
        print(f"    WARNING: VEGC data not available for {region}, skipping VEGC-colored plot")
    else:
        # Train VEGC emulator
        y_vegc = df_clean['FATES_VEGC_PPE_2000_V1'].values
        lr_vegc = LinearRegression()
        lr_vegc.fit(X, y_vegc)
        
        # Predict VEGC for the 1000 LHS samples
        vegc_emul_pred = lr_vegc.predict(lhs_emul)
        
        print(f"    VEGC emulator R²: {lr_vegc.score(X, y_vegc):.4f}")
        print(f"    Predicted VEGC range: {vegc_emul_pred.min():.4f} to {vegc_emul_pred.max():.4f}")
        
        fig_lai_vegc, axes_lai_vegc = plt.subplots(2, 2, figsize=(16, 12))
        fig_lai_vegc.suptitle(f'{lai_label}(2000) vs Parameters (colored by VEGC emulator) ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                              fontsize=16, fontweight='bold')
        
        scatter_vegc = None  # Initialize outside loop
        for idx, param in enumerate(params):
            row = idx // 2
            col = idx % 2
            ax = axes_lai_vegc[row, col]
            
            param_idx = params.index(param)
            
            # Plot emulator samples colored by predicted VEGC
            scatter_vegc = ax.scatter(lhs_emul[:, param_idx], lai_emul_pred, s=15, alpha=0.6,
                      c=vegc_emul_pred, cmap='RdYlBu_r', edgecolors='none',
                      label=f'{n_emulator_samples} emulator samples')
            
            # Plot actual ensemble members on top, colored by actual VEGC
            ax.scatter(df_clean[param], y_lai, s=100, alpha=0.9, c=y_vegc,
                      cmap='RdYlBu_r', edgecolors='black', linewidth=1.5, 
                      label='Ensemble members', zorder=3,
                      vmin=vegc_emul_pred.min(), vmax=vegc_emul_pred.max())
            
            # Add observational LAI as horizontal line
            if not np.isnan(obs_lai):
                if region == 'ENTs':
                    ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                              alpha=0.8, label=f'MODIS Summer {lai_label}: {obs_lai:.2f}', zorder=2)
                else:
                    ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                              alpha=0.8, label=f'MODIS {lai_label}: {obs_lai:.2f}', zorder=2)
            if region == 'ENTs' and region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
                ax.axhline(obs_lai_regional_mean_ent[region], color='darkblue', linestyle='-.', linewidth=2,
                          alpha=0.8, label=f'MODIS Mean {lai_label}: {obs_lai_regional_mean_ent[region]:.2f}', zorder=2)
            if not np.isnan(obs_lai_avhrr):
                ax.axhline(obs_lai_avhrr, color='limegreen', linestyle=':', linewidth=2,
                          alpha=0.8, label=f'AVHRR {lai_label}: {obs_lai_avhrr:.2f}', zorder=2)
            
            # Add ensemble labels with more offset
            for i, row_data in enumerate(df_clean.iterrows()):
                _, row = row_data
                ax.annotate(f"n{int(row['ensemble_member']):02d}",
                           (row[param], y_lai[i]),
                           xytext=(0, 12), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='none', alpha=0.7))
            
            ax.set_xlabel(param.replace('fates_', '').replace('_', ' '),
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{lai_label}(2000) [m²/m²]', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Add colorbar for VEGC values
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        if scatter_vegc is not None:
            cbar_ax = fig_lai_vegc.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(scatter_vegc, cax=cbar_ax)
            cbar.set_label('VEGC(2000) [kgC/m²]', fontsize=11, fontweight='bold')
        
        output_file_lai_vegc = OUTPUT_DIR_LAI_EMULATOR / f"lai_vs_params_colored_by_vegc_{YEAR_RANGE}_{region}.png"
        fig_lai_vegc.savefig(output_file_lai_vegc, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file_lai_vegc.name}")
        plt.close(fig_lai_vegc)

# Create CO2 fertilization effect plots (ratio of 2000/V1 for each variable)
print("\n" + "="*60)
print("Creating CO2 Fertilization Effect Analysis (2000/V1 ratios)...")
print("="*60)

variables = ['FATES_LAI', 'FATES_GPP', 'FATES_NPP', 'FATES_VEGC']
params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    print(f"\n  Checking region: {region}")
    if region not in data:
        print(f"    Region {region} not in data dictionary! Skipping.")
        continue
    
    df = data[region]
    
    print(f"\n{'='*50}")
    print(f"Region: {region.replace('_', ' ').title()}")
    print(f"{'='*50}")
    
    for var in variables:
        var_v1 = f'{var}_PPE_V1'
        var_2000 = f'{var}_PPE_2000_V1'
        
        # Check if both columns exist
        if var_v1 not in df.columns or var_2000 not in df.columns:
            print(f"  WARNING: Missing columns for {var}, skipping.")
            continue
        
        print(f"\n  Processing {var}...")
        
        # Calculate ratio
        df[f'{var}_ratio'] = df[var_2000] / df[var_v1]
        
        # Remove rows with NaN or inf
        df_clean = df[params + ['ensemble_member', var_v1, var_2000, f'{var}_ratio']].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        print(f"    Training points: {len(df_clean)}")
        
        if len(df_clean) < 3:
            print(f"    WARNING: Only {len(df_clean)} training points! Skipping.")
            continue
        
        # Train emulator for the ratio
        X = df_clean[params].values
        y_ratio = df_clean[f'{var}_ratio'].values
        
        lr_ratio = LinearRegression()
        lr_ratio.fit(X, y_ratio)
        
        y_ratio_pred_train = lr_ratio.predict(X)
        train_score = lr_ratio.score(X, y_ratio)
        rmse = np.sqrt(np.mean((y_ratio - y_ratio_pred_train)**2))
        
        print(f"    Emulator R²: {train_score:.4f}")
        print(f"    Emulator RMSE: {rmse:.4f}")
        print(f"    Ratio range: {y_ratio.min():.4f} to {y_ratio.max():.4f}")
        
        # Train emulator for V1 baseline to color dots
        y_v1 = df_clean[var_v1].values
        lr_v1 = LinearRegression()
        lr_v1.fit(X, y_v1)
        
        # Generate LHS samples for emulator
        n_samples = 1000
        param_bounds = np.array([[df_clean[param].min(), df_clean[param].max()] for param in params])
        
        sampler = qmc.LatinHypercube(d=len(params), seed=44)
        lhs_unit = sampler.random(n=n_samples)
        lhs_samples = qmc.scale(lhs_unit, param_bounds[:, 0], param_bounds[:, 1])
        
        # Predict ratio for LHS samples
        y_ratio_lhs = lr_ratio.predict(lhs_samples)
        
        # Predict V1 baseline for LHS samples (for coloring)
        y_v1_lhs = lr_v1.predict(lhs_samples)
        
        print(f"    LHS predicted ratio range: {y_ratio_lhs.min():.4f} to {y_ratio_lhs.max():.4f}")
        print(f"    V1 baseline range: {y_v1.min():.4f} to {y_v1.max():.4f}")
        
        # Create 4-panel plot: ratio vs each parameter with emulator
        fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(16, 12))
        fig_ratio.suptitle(f'{var} CO2 Fertilization (2000/V1 ratio) ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                          fontsize=16, fontweight='bold')
        
        scatter_emul = None  # Initialize outside loop
        for idx, param in enumerate(params):
            row = idx // 2
            col = idx % 2
            ax = axes_ratio[row, col]
            
            param_idx = params.index(param)
            
            # Plot emulator samples colored by predicted V1 baseline
            scatter_emul = ax.scatter(lhs_samples[:, param_idx], y_ratio_lhs, s=15, alpha=0.6,
                      c=y_v1_lhs, cmap='plasma', edgecolors='none', 
                      vmin=y_v1.min(), vmax=y_v1.max(),
                      label=f'{n_samples} emulator samples')
            
            # Plot actual ensemble members colored by actual V1 baseline
            ax.scatter(df_clean[param], y_ratio, s=120, alpha=0.9, c=y_v1,
                      cmap='plasma', edgecolors='black', linewidth=1.5, 
                      vmin=y_v1.min(), vmax=y_v1.max(),
                      label='Ensemble members', zorder=3)
            
            # Add ratio = 1 reference line (no CO2 effect)
            ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                      label='Ratio = 1 (no effect)', zorder=2)
            
            # Add ensemble labels
            for i, row_data in enumerate(df_clean.iterrows()):
                _, row = row_data
                ax.annotate(f"n{int(row['ensemble_member']):02d}",
                           (row[param], y_ratio[i]),
                           xytext=(0, 8), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='none', alpha=0.7))
            
            ax.set_xlabel(param.replace('fates_', '').replace('_', ' '),
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{var} Ratio (2000/V1)', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Add colorbar for V1 baseline values
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        if scatter_emul is not None:
            cbar_ax = fig_ratio.add_axes([0.96, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(scatter_emul, cax=cbar_ax)
            cbar.set_label(f'{var} V1 baseline', fontsize=11, fontweight='bold')
        
        var_name_clean = var.replace('FATES_', '').lower()
        output_file_ratio = OUTPUT_DIR_CO2_FERT / f"co2_fert_{var_name_clean}_{YEAR_RANGE}_{region}.png"
        fig_ratio.savefig(output_file_ratio, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file_ratio.name}")
        plt.close(fig_ratio)

# Create Carbon Use Efficiency (CUE) plots (NPP/GPP)
print("\n" + "="*60)
print("Creating Carbon Use Efficiency (CUE = NPP/GPP) Analysis...")
print("="*60)

# Read observed CUE min/max values
CUE_OBS_FILE = DATA_DIR / "cue.csv"
obs_cue_data = {}
if CUE_OBS_FILE.exists():
    print("\nReading observed CUE data...")
    try:
        cue_df = pd.read_csv(CUE_OBS_FILE, skipinitialspace=True, on_bad_lines='skip')
        # Clean column names
        cue_df.columns = cue_df.columns.str.strip()
        # Create mapping of PFT number to min/max CUE
        for _, row in cue_df.iterrows():
            try:
                pft_num = int(row['num'])
                obs_cue_data[pft_num] = {
                    'min': float(row['cuemin']),
                    'max': float(row['cuemax']),
                    'name': str(row['pftname']).strip()
                }
            except (ValueError, KeyError) as e:
                print(f"  WARNING: Skipping malformed row: {e}")
                continue
        print(f"  Loaded observed CUE data for {len(obs_cue_data)} PFTs")
    except Exception as e:
        print(f"  ERROR reading CUE file: {e}")
else:
    print(f"  WARNING: Observed CUE file not found: {CUE_OBS_FILE}")

# Map region names to PFT numbers for observed CUE
REGION_TO_PFT = {
    'BETs': 1,   # Broadleaf Evergreen Tropical Tree
    'ENTs': 2,   # Needleleaf Evergreen Extratropical Tree
    'BDTs': 6,   # Broadleaf Colddecid Extratropical Tree
    'AC3G': 12,  # Arctic C3 Grass
    'C3G': 13,   # Cool C3 Grass
    'C4': 14,    # C4 Grass
}

params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 
          'fates_allom_d2bl1', 'fates_leaf_slatop']

for region in REGIONS:
    print(f"\n  Checking region: {region}")
    if region not in data:
        print(f"    Region {region} not in data dictionary! Skipping.")
        continue
    
    df = data[region]
    
    print(f"\n{'='*50}")
    print(f"Region: {region.replace('_', ' ').title()}")
    print(f"{'='*50}")
    
    # Calculate CUE for both V1 and 2000 ensembles
    for ensemble_name, npp_col, gpp_col in [
        ('V1', 'FATES_NPP_PPE_V1', 'FATES_GPP_PPE_V1'),
        ('2000', 'FATES_NPP_PPE_2000_V1', 'FATES_GPP_PPE_2000_V1')
    ]:
        
        if npp_col not in df.columns or gpp_col not in df.columns:
            print(f"  WARNING: Missing columns for {ensemble_name}, skipping.")
            continue
        
        print(f"\n  Processing CUE for {ensemble_name} ensemble...")
        
        # Calculate CUE
        df[f'CUE_{ensemble_name}'] = df[npp_col] / df[gpp_col]
        
        # Get corresponding LAI column for coloring
        lai_col = f'FATES_LAI_PPE_{ensemble_name}' if ensemble_name == 'V1' else 'FATES_LAI_PPE_2000_V1'
        
        # Remove rows with NaN or inf
        df_clean = df[params + ['ensemble_member', npp_col, gpp_col, f'CUE_{ensemble_name}', lai_col]].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna()
        
        print(f"    Training points: {len(df_clean)}")
        
        if len(df_clean) < 3:
            print(f"    WARNING: Only {len(df_clean)} training points! Skipping.")
            continue
        
        # Train emulator for CUE
        X = df_clean[params].values
        y_cue = df_clean[f'CUE_{ensemble_name}'].values
        y_lai = df_clean[lai_col].values
        
        lr_cue = LinearRegression()
        lr_cue.fit(X, y_cue)
        
        y_cue_pred_train = lr_cue.predict(X)
        train_score = lr_cue.score(X, y_cue)
        rmse = np.sqrt(np.mean((y_cue - y_cue_pred_train)**2))
        
        print(f"    Emulator R²: {train_score:.4f}")
        print(f"    Emulator RMSE: {rmse:.4f}")
        print(f"    CUE range: {y_cue.min():.4f} to {y_cue.max():.4f}")
        
        # Generate LHS samples for emulator
        n_samples = 1000
        param_bounds = np.array([[df_clean[param].min(), df_clean[param].max()] for param in params])
        
        sampler = qmc.LatinHypercube(d=len(params), seed=45)
        lhs_unit = sampler.random(n=n_samples)
        lhs_samples = qmc.scale(lhs_unit, param_bounds[:, 0], param_bounds[:, 1])
        
        # Predict CUE for LHS samples
        y_cue_lhs = lr_cue.predict(lhs_samples)
        
        print(f"    LHS predicted CUE range: {y_cue_lhs.min():.4f} to {y_cue_lhs.max():.4f}")
        
        # Create 4-panel plot: CUE vs each parameter with emulator
        fig_cue, axes_cue = plt.subplots(2, 2, figsize=(16, 12))
        fig_cue.suptitle(f'Carbon Use Efficiency (NPP/GPP) - {ensemble_name} ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                        fontsize=16, fontweight='bold')
        
        for idx, param in enumerate(params):
            row = idx // 2
            col = idx % 2
            ax = axes_cue[row, col]
            
            param_idx = params.index(param)
            
            # Plot emulator samples
            ax.scatter(lhs_samples[:, param_idx], y_cue_lhs, s=15, alpha=0.6,
                      c='lightcoral', edgecolors='none', label=f'{n_samples} emulator samples')
            
            # Plot actual ensemble members
            ax.scatter(df_clean[param], y_cue, s=120, alpha=0.9, c='darkred',
                      edgecolors='black', linewidth=1.5, label='Ensemble members', zorder=3)
            
            # Add ensemble labels
            for i, row_data in enumerate(df_clean.iterrows()):
                _, row = row_data
                ax.annotate(f"n{int(row['ensemble_member']):02d}",
                           (row[param], y_cue[i]),
                           xytext=(0, 10), textcoords='offset points',
                           fontsize=8, ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='none', alpha=0.7))
            
            # Add observed CUE range as grey bar if available
            if region in REGION_TO_PFT and REGION_TO_PFT[region] in obs_cue_data:
                pft_num = REGION_TO_PFT[region]
                cue_min = obs_cue_data[pft_num]['min']
                cue_max = obs_cue_data[pft_num]['max']
                ax.axhspan(cue_min, cue_max, alpha=0.2, color='grey', 
                          label=f'Observed CUE range\n({cue_min:.2f}-{cue_max:.2f})', zorder=1)
            
            ax.set_xlabel(param.replace('fates_', '').replace('_', ' '),
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(f'CUE (NPP/GPP)', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file_cue = OUTPUT_DIR_CUE / f"cue_{ensemble_name.lower()}_{YEAR_RANGE}_{region}.png"
        fig_cue.savefig(output_file_cue, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file_cue.name}")
        plt.close(fig_cue)

print("\n" + "="*60)
print("Analysis complete!")
print(f"Output files saved to: {OUTPUT_DIR}")
