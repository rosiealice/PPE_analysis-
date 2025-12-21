#!/usr/bin/env python
"""
Script to extract GPP and LAI observational data for specific grid points
This extracts data from MODIS, AVHRR, FLUXCOM, and WECANN datasets
and writes the results to CSV files.

This script should be run once (or when observations are updated) to generate
the CSV files that will be read by analyze_csv_results.py
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuration - paths to observational datasets
OBS_LAI_FILE_MODIS_TIMESERIES = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/lai/MODIS/lai_0.5x0.5.nc"
OBS_LAI_FILE_MODIS_SUMMER_MEAN = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/modis_lai/modis_lai_summer_mean.nc"
OBS_LAI_FILE_MODIS_ANNUAL_MAX = "/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/modis_lai/modis_lai_mean_annual_maximum.nc"
OBS_LAI_FILE_AVHRR = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/lai/AVHRR/lai_0.5x0.5.nc"
OBS_GPP_FILE_FLUXCOM = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/gpp/FLUXCOM/gpp.nc"
OBS_GPP_FILE_WECANN = "/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/gpp/WECANN/gpp.nc"

# Set ensemble number and data directory
ens_n = 2  # Set to 1 or 2 depending on which ensemble you want to process
DATA_DIR = Path("/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/")

if ens_n == 1:
    DATA_DIR = DATA_DIR / "ens_v1_soilc/"
elif ens_n == 2:
    DATA_DIR = DATA_DIR / "ens_v2_c4g/"

CSV_DIR = DATA_DIR / "csv_data"
GRID_CELLS_DIR = Path("/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/pft_grid_cells/")
OUTPUT_DIR = Path(__file__).parent  # Save CSVs in data_extraction directory

# Define regions
REGIONS = ['north_65', 'BDTs', 'BETs', 'ENTs', 'AC3G', 'C3G', 'C4']

# Region definitions
REGION_BOUNDS = {
    'north_65': {'lat_min': 65, 'lat_max': 90, 'lon_min': 0, 'lon_max': 360},
}

print("="*60)
print("EXTRACTING OBSERVATIONAL GPP AND LAI DATA")
print("="*60)
print(f"Ensemble: {ens_n}")
print(f"CSV directory: {CSV_DIR}")
print(f"Grid cells directory: {GRID_CELLS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Load grid cell locations for each region
def load_grid_cells(filename):
    """Load grid cell locations from file"""
    print(f"  Looking for: {filename}")
    if not filename.exists():
        print(f"    File not found!")
        return None
    print(f"    File found, loading...")
    data = np.loadtxt(filename, skiprows=1)
    print(f"    Loaded {len(data)} grid cells")
    return {'lats': data[:, 0], 'lons': data[:, 1]}

print("\nLooking for grid cell files...")
BDT_GRID_CELLS = load_grid_cells(GRID_CELLS_DIR / "bdt_grid_cells.txt")
BET_GRID_CELLS = load_grid_cells(GRID_CELLS_DIR / "bet_grid_cells.txt")
ENT_GRID_CELLS = load_grid_cells(GRID_CELLS_DIR / "ent_grid_cells.txt")
AC3G_GRID_CELLS = load_grid_cells(GRID_CELLS_DIR / "ac3g_grid_cells.txt")
C3G_GRID_CELLS = load_grid_cells(GRID_CELLS_DIR / "c3g_grid_cells.txt")
C4_GRID_CELLS = load_grid_cells(GRID_CELLS_DIR / "c4_grid_cells.txt")

print("\nSummary:")
if BDT_GRID_CELLS: print(f"  ✓ Loaded {len(BDT_GRID_CELLS['lats'])} BDT grid cells")
else: print(f"  ✗ No BDT grid cells")
if BET_GRID_CELLS: print(f"  ✓ Loaded {len(BET_GRID_CELLS['lats'])} BET grid cells")
else: print(f"  ✗ No BET grid cells")
if ENT_GRID_CELLS: print(f"  ✓ Loaded {len(ENT_GRID_CELLS['lats'])} ENT grid cells")
else: print(f"  ✗ No ENT grid cells")
if AC3G_GRID_CELLS: print(f"  ✓ Loaded {len(AC3G_GRID_CELLS['lats'])} AC3G grid cells")
else: print(f"  ✗ No AC3G grid cells")
if C3G_GRID_CELLS: print(f"  ✓ Loaded {len(C3G_GRID_CELLS['lats'])} C3G grid cells")
else: print(f"  ✗ No C3G grid cells")
if C4_GRID_CELLS: print(f"  ✓ Loaded {len(C4_GRID_CELLS['lats'])} C4 grid cells")
else: print(f"  ✗ No C4 grid cells")

# Initialize dictionaries to store results - ensure all regions start with NaN
obs_lai_modis = {region: np.nan for region in REGIONS}
obs_lai_avhrr = {region: np.nan for region in REGIONS}
obs_lai_modis_summer = {region: np.nan for region in REGIONS}
obs_lai_modis_annual_max = {region: np.nan for region in REGIONS}
obs_gpp_fluxcom = {region: np.nan for region in REGIONS}
obs_gpp_wecann = {region: np.nan for region in REGIONS}

# Store point-level data for each region and dataset
point_data = {}

# Helper function to find LAI/GPP variable in dataset
def find_variable(ds, possible_names):
    """Find a variable in dataset from list of possible names"""
    for var in possible_names:
        if var in ds:
            return var
    return None

# Helper function to convert longitude if needed
def convert_longitude(ds):
    """Convert longitude from -180:180 to 0:360 if needed"""
    if ds['lon'].min() < 0:
        print(f"  Converting longitude from -180:180 to 0:360...")
        ds = ds.assign_coords(lon=(ds['lon'] % 360))
        ds = ds.sortby('lon')
    return ds

# Helper function to extract values for specific grid cells
def extract_grid_cell_values(ds, var_name, grid_cells, summer_months=False):
    """Extract values for specific grid cells, returns values and point data"""
    if grid_cells is None:
        return [], []
    
    values = []
    point_data = []
    for lat, lon in zip(grid_cells['lats'], grid_cells['lons']):
        data_point = ds[var_name].sel(lat=lat, lon=lon, method='nearest')
        
        # If summer months requested and time dimension exists
        if summer_months and 'time' in data_point.dims:
            data_point = data_point.sel(time=data_point.time.dt.month.isin([6, 7, 8]))
        
        # Calculate mean
        val = data_point.mean(skipna=True).values
        values.append(float(val))
        point_data.append({'lat': lat, 'lon': lon, 'value': float(val)})
    
    return values, point_data

print("\n" + "="*60)
print("EXTRACTING LAI DATA FROM MODIS")
print("="*60)

try:
    ds_modis = xr.open_dataset(OBS_LAI_FILE_MODIS_TIMESERIES)
    print(f"Loaded MODIS LAI timeseries")
    
    ds_modis = convert_longitude(ds_modis)
    lai_var = find_variable(ds_modis, ['lai', 'LAI', 'leaf_area_index'])
    
    if lai_var:
        print(f"Using LAI variable: {lai_var}")
        
        # Extract for each region
        for region in REGIONS:
            if region in REGION_BOUNDS:
                bounds = REGION_BOUNDS[region]
                ds_subset = ds_modis[lai_var].where(
                    (ds_modis['lat'] >= bounds['lat_min']) &
                    (ds_modis['lat'] <= bounds['lat_max']) &
                    (ds_modis['lon'] >= bounds['lon_min']) &
                    (ds_modis['lon'] <= bounds['lon_max']),
                    drop=False
                )
                obs_lai_modis[region] = float(ds_subset.mean(skipna=True).values)
        
        # Extract for specific grid cells
        if BDT_GRID_CELLS:
            print(f"\n  Extracting MODIS LAI for {len(BDT_GRID_CELLS['lats'])} BDT grid cells...")
            values, points = extract_grid_cell_values(ds_modis, lai_var, BDT_GRID_CELLS)
            obs_lai_modis['BDTs'] = np.nanmean(values)
            point_data['BDTs_modis_lai'] = points
            print(f"  BDTs: {obs_lai_modis['BDTs']:.4f}")
        else:
            print(f"\n  No BDT grid cells loaded - skipping")
        
        if BET_GRID_CELLS:
            print(f"  Extracting MODIS LAI for {len(BET_GRID_CELLS['lats'])} BET grid cells...")
            values, points = extract_grid_cell_values(ds_modis, lai_var, BET_GRID_CELLS)
            obs_lai_modis['BETs'] = np.nanmean(values)
            point_data['BETs_modis_lai'] = points
            print(f"  BETs: {obs_lai_modis['BETs']:.4f}")
        else:
            print(f"  No BET grid cells loaded - skipping")
        
        if AC3G_GRID_CELLS:
            print(f"  Extracting MODIS LAI for {len(AC3G_GRID_CELLS['lats'])} AC3G grid cells...")
            values, points = extract_grid_cell_values(ds_modis, lai_var, AC3G_GRID_CELLS)
            obs_lai_modis['AC3G'] = np.nanmean(values)
            point_data['AC3G_modis_lai'] = points
            print(f"  AC3G: {obs_lai_modis['AC3G']:.4f}")
        else:
            print(f"  No AC3G grid cells loaded - skipping")
        
        if C3G_GRID_CELLS:
            print(f"  Extracting MODIS LAI for {len(C3G_GRID_CELLS['lats'])} C3G grid cells...")
            values, points = extract_grid_cell_values(ds_modis, lai_var, C3G_GRID_CELLS)
            obs_lai_modis['C3G'] = np.nanmean(values)
            point_data['C3G_modis_lai'] = points
            print(f"  C3G: {obs_lai_modis['C3G']:.4f}")
        else:
            print(f"  No C3G grid cells loaded - skipping")
        
        if C4_GRID_CELLS:
            print(f"  Extracting MODIS LAI for {len(C4_GRID_CELLS['lats'])} C4 grid cells...")
            values, points = extract_grid_cell_values(ds_modis, lai_var, C4_GRID_CELLS)
            obs_lai_modis['C4'] = np.nanmean(values)
            point_data['C4_modis_lai'] = points
            print(f"  C4: {obs_lai_modis['C4']:.4f}")
        else:
            print(f"  No C4 grid cells loaded - skipping")
        
        # For ENTs, extract mean (annual) LAI from timeseries
        if ENT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_modis, lai_var, ENT_GRID_CELLS)
            obs_lai_modis['ENTs'] = np.nanmean(values)
            point_data['ENTs_modis_lai_mean'] = points
            print(f"ENTs (mean): {obs_lai_modis['ENTs']:.4f}")
    
    ds_modis.close()
    
except Exception as e:
    print(f"ERROR reading MODIS LAI data: {str(e)}")
    import traceback
    traceback.print_exc()

# Extract MODIS summer mean LAI for all regions with grid cells
print("\n" + "="*60)
print("EXTRACTING MODIS SUMMER LAI FOR ALL REGIONS")
print("="*60)

try:
    if os.path.exists(OBS_LAI_FILE_MODIS_SUMMER_MEAN):
        ds_summer = xr.open_dataset(OBS_LAI_FILE_MODIS_SUMMER_MEAN)
        ds_summer = convert_longitude(ds_summer)
        
        lai_var = find_variable(ds_summer, ['lai', 'LAI', 'leaf_area_index'])
        if lai_var:
            print(f"Using LAI variable: {lai_var}")
            
            # Extract for all regions with grid cells
            for region, grid_cells in [('BDTs', BDT_GRID_CELLS), ('BETs', BET_GRID_CELLS), 
                                       ('ENTs', ENT_GRID_CELLS), ('AC3G', AC3G_GRID_CELLS),
                                       ('C3G', C3G_GRID_CELLS), ('C4', C4_GRID_CELLS)]:
                if grid_cells:
                    print(f"  Extracting summer LAI for {len(grid_cells['lats'])} {region} grid cells...")
                    values, points = extract_grid_cell_values(ds_summer, lai_var, grid_cells)
                    obs_lai_modis_summer[region] = np.nanmean(values)
                    point_data[f'{region}_modis_lai_summer'] = points
                    print(f"  {region}: {obs_lai_modis_summer[region]:.4f}")
        
        ds_summer.close()
    else:
        print(f"Summer mean file not found: {OBS_LAI_FILE_MODIS_SUMMER_MEAN}")
    
except Exception as e:
    print(f"ERROR reading MODIS summer LAI: {str(e)}")
    import traceback
    traceback.print_exc()

# Extract MODIS annual maximum LAI for all regions with grid cells
print("\n" + "="*60)
print("EXTRACTING MODIS ANNUAL MAXIMUM LAI FOR ALL REGIONS")
print("="*60)

try:
    if os.path.exists(OBS_LAI_FILE_MODIS_ANNUAL_MAX):
        ds_annual = xr.open_dataset(OBS_LAI_FILE_MODIS_ANNUAL_MAX)
        ds_annual = convert_longitude(ds_annual)
        
        lai_var = find_variable(ds_annual, ['lai', 'LAI', 'leaf_area_index'])
        if lai_var:
            print(f"Using LAI variable: {lai_var}")
            
            # Extract for all regions with grid cells
            for region, grid_cells in [('BDTs', BDT_GRID_CELLS), ('BETs', BET_GRID_CELLS), 
                                       ('ENTs', ENT_GRID_CELLS), ('AC3G', AC3G_GRID_CELLS),
                                       ('C3G', C3G_GRID_CELLS), ('C4', C4_GRID_CELLS)]:
                if grid_cells:
                    print(f"  Extracting annual max LAI for {len(grid_cells['lats'])} {region} grid cells...")
                    values, points = extract_grid_cell_values(ds_annual, lai_var, grid_cells)
                    obs_lai_modis_annual_max[region] = np.nanmean(values)
                    point_data[f'{region}_modis_lai_annual_max'] = points
                    print(f"  {region}: {obs_lai_modis_annual_max[region]:.4f}")
        
        ds_annual.close()
    else:
        print(f"Annual max file not found: {OBS_LAI_FILE_MODIS_ANNUAL_MAX}")
    
except Exception as e:
    print(f"ERROR reading MODIS annual max LAI: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("EXTRACTING LAI DATA FROM AVHRR")
print("="*60)

try:
    ds_avhrr = xr.open_dataset(OBS_LAI_FILE_AVHRR)
    print(f"Loaded AVHRR LAI data")
    
    ds_avhrr = convert_longitude(ds_avhrr)
    lai_var = find_variable(ds_avhrr, ['lai', 'LAI', 'leaf_area_index'])
    
    if lai_var:
        print(f"Using LAI variable: {lai_var}")
        
        # Extract for each region
        for region in REGIONS:
            if region in REGION_BOUNDS:
                bounds = REGION_BOUNDS[region]
                ds_subset = ds_avhrr[lai_var].where(
                    (ds_avhrr['lat'] >= bounds['lat_min']) &
                    (ds_avhrr['lat'] <= bounds['lat_max']) &
                    (ds_avhrr['lon'] >= bounds['lon_min']) &
                    (ds_avhrr['lon'] <= bounds['lon_max']),
                    drop=False
                )
                obs_lai_avhrr[region] = float(ds_subset.mean(skipna=True).values)
        
        # Extract for specific grid cells
        if BDT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_avhrr, lai_var, BDT_GRID_CELLS)
            obs_lai_avhrr['BDTs'] = np.nanmean(values)
            point_data['BDTs_avhrr_lai'] = points
            print(f"BDTs: {obs_lai_avhrr['BDTs']:.4f}")
        
        if BET_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_avhrr, lai_var, BET_GRID_CELLS)
            obs_lai_avhrr['BETs'] = np.nanmean(values)
            point_data['BETs_avhrr_lai'] = points
            print(f"BETs: {obs_lai_avhrr['BETs']:.4f}")
        
        if ENT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_avhrr, lai_var, ENT_GRID_CELLS, summer_months=True)
            obs_lai_avhrr['ENTs'] = np.nanmean(values)
            point_data['ENTs_avhrr_lai'] = points
            print(f"ENTs: {obs_lai_avhrr['ENTs']:.4f}")
        
        if AC3G_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_avhrr, lai_var, AC3G_GRID_CELLS)
            obs_lai_avhrr['AC3G'] = np.nanmean(values)
            point_data['AC3G_avhrr_lai'] = points
            print(f"AC3G: {obs_lai_avhrr['AC3G']:.4f}")
        
        if C3G_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_avhrr, lai_var, C3G_GRID_CELLS)
            obs_lai_avhrr['C3G'] = np.nanmean(values)
            point_data['C3G_avhrr_lai'] = points
            print(f"C3G: {obs_lai_avhrr['C3G']:.4f}")
        
        if C4_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_avhrr, lai_var, C4_GRID_CELLS)
            obs_lai_avhrr['C4'] = np.nanmean(values)
            point_data['C4_avhrr_lai'] = points
            print(f"C4: {obs_lai_avhrr['C4']:.4f}")
    
    ds_avhrr.close()
    
except Exception as e:
    print(f"ERROR reading AVHRR LAI data: {str(e)}")

print("\n" + "="*60)
print("EXTRACTING GPP DATA FROM FLUXCOM")
print("="*60)

try:
    ds_fluxcom = xr.open_dataset(OBS_GPP_FILE_FLUXCOM)
    print(f"Loaded FLUXCOM GPP data")
    
    ds_fluxcom = convert_longitude(ds_fluxcom)
    gpp_var = find_variable(ds_fluxcom, ['gpp', 'GPP', 'gross_primary_productivity'])
    
    if gpp_var:
        print(f"Using GPP variable: {gpp_var}")
        print(f"GPP units: gC m-2 day-1")
        
        # Extract for each region
        for region in REGIONS:
            if region in REGION_BOUNDS:
                bounds = REGION_BOUNDS[region]
                ds_subset = ds_fluxcom[gpp_var].where(
                    (ds_fluxcom['lat'] >= bounds['lat_min']) &
                    (ds_fluxcom['lat'] <= bounds['lat_max']) &
                    (ds_fluxcom['lon'] >= bounds['lon_min']) &
                    (ds_fluxcom['lon'] <= bounds['lon_max']),
                    drop=False
                )
                obs_gpp_fluxcom[region] = float(ds_subset.mean(skipna=True).values)
        
        # Extract for specific grid cells
        if BDT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_fluxcom, gpp_var, BDT_GRID_CELLS)
            obs_gpp_fluxcom['BDTs'] = np.nanmean(values)
            point_data['BDTs_fluxcom_gpp'] = points
            print(f"BDTs: {obs_gpp_fluxcom['BDTs']:.4f}")
        
        if BET_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_fluxcom, gpp_var, BET_GRID_CELLS)
            obs_gpp_fluxcom['BETs'] = np.nanmean(values)
            point_data['BETs_fluxcom_gpp'] = points
            print(f"BETs: {obs_gpp_fluxcom['BETs']:.4f}")
        
        if ENT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_fluxcom, gpp_var, ENT_GRID_CELLS)
            obs_gpp_fluxcom['ENTs'] = np.nanmean(values)
            point_data['ENTs_fluxcom_gpp'] = points
            print(f"ENTs: {obs_gpp_fluxcom['ENTs']:.4f}")
        
        if AC3G_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_fluxcom, gpp_var, AC3G_GRID_CELLS)
            obs_gpp_fluxcom['AC3G'] = np.nanmean(values)
            point_data['AC3G_fluxcom_gpp'] = points
            print(f"AC3G: {obs_gpp_fluxcom['AC3G']:.4f}")
        
        if C3G_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_fluxcom, gpp_var, C3G_GRID_CELLS)
            obs_gpp_fluxcom['C3G'] = np.nanmean(values)
            point_data['C3G_fluxcom_gpp'] = points
            print(f"C3G: {obs_gpp_fluxcom['C3G']:.4f}")
        
        if C4_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_fluxcom, gpp_var, C4_GRID_CELLS)
            obs_gpp_fluxcom['C4'] = np.nanmean(values)
            point_data['C4_fluxcom_gpp'] = points
            print(f"C4: {obs_gpp_fluxcom['C4']:.4f}")
        
        if BET_GRID_CELLS:
            values = extract_grid_cell_values(ds_fluxcom, gpp_var, BET_GRID_CELLS)
            obs_gpp_fluxcom['BETs'] = np.nanmean(values)
            print(f"BETs: {obs_gpp_fluxcom['BETs']:.4f}")
        
        if ENT_GRID_CELLS:
            values = extract_grid_cell_values(ds_fluxcom, gpp_var, ENT_GRID_CELLS)
            obs_gpp_fluxcom['ENTs'] = np.nanmean(values)
            print(f"ENTs: {obs_gpp_fluxcom['ENTs']:.4f}")
        
        if AC3G_GRID_CELLS:
            values = extract_grid_cell_values(ds_fluxcom, gpp_var, AC3G_GRID_CELLS)
            obs_gpp_fluxcom['AC3G'] = np.nanmean(values)
            print(f"AC3G: {obs_gpp_fluxcom['AC3G']:.4f}")
        
        if C3G_GRID_CELLS:
            values = extract_grid_cell_values(ds_fluxcom, gpp_var, C3G_GRID_CELLS)
            obs_gpp_fluxcom['C3G'] = np.nanmean(values)
            print(f"C3G: {obs_gpp_fluxcom['C3G']:.4f}")
        
        if C4_GRID_CELLS:
            values = extract_grid_cell_values(ds_fluxcom, gpp_var, C4_GRID_CELLS)
            obs_gpp_fluxcom['C4'] = np.nanmean(values)
            print(f"C4: {obs_gpp_fluxcom['C4']:.4f}")
    
    ds_fluxcom.close()
    
except Exception as e:
    print(f"ERROR reading FLUXCOM GPP data: {str(e)}")

print("\n" + "="*60)
print("EXTRACTING GPP DATA FROM WECANN")
print("="*60)

try:
    ds_wecann = xr.open_dataset(OBS_GPP_FILE_WECANN)
    print(f"Loaded WECANN GPP data")
    
    ds_wecann = convert_longitude(ds_wecann)
    gpp_var = find_variable(ds_wecann, ['gpp', 'GPP', 'gross_primary_productivity'])
    
    if gpp_var:
        print(f"Using GPP variable: {gpp_var}")
        print(f"GPP units: gC m-2 day-1")
        
        # Extract for each region
        for region in REGIONS:
            if region in REGION_BOUNDS:
                bounds = REGION_BOUNDS[region]
                ds_subset = ds_wecann[gpp_var].where(
                    (ds_wecann['lat'] >= bounds['lat_min']) &
                    (ds_wecann['lat'] <= bounds['lat_max']) &
                    (ds_wecann['lon'] >= bounds['lon_min']) &
                    (ds_wecann['lon'] <= bounds['lon_max']),
                    drop=False
                )
                obs_gpp_wecann[region] = float(ds_subset.mean(skipna=True).values)
        
        # Extract for specific grid cells
        if BDT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_wecann, gpp_var, BDT_GRID_CELLS)
            obs_gpp_wecann['BDTs'] = np.nanmean(values)
            point_data['BDTs_wecann_gpp'] = points
            print(f"BDTs: {obs_gpp_wecann['BDTs']:.4f}")
        
        if BET_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_wecann, gpp_var, BET_GRID_CELLS)
            obs_gpp_wecann['BETs'] = np.nanmean(values)
            point_data['BETs_wecann_gpp'] = points
            print(f"BETs: {obs_gpp_wecann['BETs']:.4f}")
        
        if ENT_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_wecann, gpp_var, ENT_GRID_CELLS)
            obs_gpp_wecann['ENTs'] = np.nanmean(values)
            point_data['ENTs_wecann_gpp'] = points
            print(f"ENTs: {obs_gpp_wecann['ENTs']:.4f}")
        
        if AC3G_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_wecann, gpp_var, AC3G_GRID_CELLS)
            obs_gpp_wecann['AC3G'] = np.nanmean(values)
            point_data['AC3G_wecann_gpp'] = points
            print(f"AC3G: {obs_gpp_wecann['AC3G']:.4f}")
        
        if C3G_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_wecann, gpp_var, C3G_GRID_CELLS)
            obs_gpp_wecann['C3G'] = np.nanmean(values)
            point_data['C3G_wecann_gpp'] = points
            print(f"C3G: {obs_gpp_wecann['C3G']:.4f}")
        
        if C4_GRID_CELLS:
            values, points = extract_grid_cell_values(ds_wecann, gpp_var, C4_GRID_CELLS)
            obs_gpp_wecann['C4'] = np.nanmean(values)
            point_data['C4_wecann_gpp'] = points
            print(f"C4: {obs_gpp_wecann['C4']:.4f}")
    
    ds_wecann.close()
    
except Exception as e:
    print(f"ERROR reading WECANN GPP data: {str(e)}")

print("\n" + "="*60)
print("WRITING RESULTS TO CSV FILES")
print("="*60)

# Create a single dataframe with all regions
df_all = pd.DataFrame({'region': REGIONS})

# Add all data columns
df_all['lai_modis'] = [obs_lai_modis.get(r, np.nan) for r in REGIONS]
df_all['lai_modis_summer'] = [obs_lai_modis_summer.get(r, np.nan) for r in REGIONS]
df_all['lai_modis_annual_max'] = [obs_lai_modis_annual_max.get(r, np.nan) for r in REGIONS]
df_all['lai_avhrr'] = [obs_lai_avhrr.get(r, np.nan) for r in REGIONS]
df_all['gpp_fluxcom'] = [obs_gpp_fluxcom.get(r, np.nan) for r in REGIONS]
df_all['gpp_wecann'] = [obs_gpp_wecann.get(r, np.nan) for r in REGIONS]

# Write to CSV
output_file = OUTPUT_DIR / "obs_gpp_lai_data.csv"
df_all.to_csv(output_file, index=False)
print(f"Saved observational data to: {output_file}")
print(f"\nColumns: {list(df_all.columns)}")
print(f"Regions: {list(df_all['region'].values)}")

# Write point-level data to separate CSV files
print("\nWriting point-level data to CSV files...")
for key, points in point_data.items():
    if len(points) > 0:
        df_points = pd.DataFrame(points)
        point_file = OUTPUT_DIR / f"obs_{key}_points.csv"
        df_points.to_csv(point_file, index=False)
        print(f"  Saved {len(points)} points to: {point_file}")

print("\n" + "="*60)
print("EXTRACTION COMPLETE")
print("="*60)
print(f"\nSummary output file: {output_file}")
print(f"Point-level files: {len(point_data)} files in {OUTPUT_DIR}")
print(f"These files should be read by analyze_csv_results.py")
