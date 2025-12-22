
#!/usr/bin/env python
"""
Script to analyze the CSV outputs from ensemble analysis
Reads the ensemble data and performs additional analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

# EMULATOR CONFIGURATION
# Choose emulator type: 'linear' or 'gp' (Gaussian Process)
EMULATOR_TYPE = 'gp'  # Change to 'gp' for Gaussian Process emulator
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_emulator():
    """Return the appropriate emulator based on EMULATOR_TYPE setting"""
    if EMULATOR_TYPE == 'gp':
        # Gaussian Process with RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0]*5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
    else:
        # Linear Regression (default)
        return LinearRegression()

# Configuration

DATA_DIR = Path("/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/")

ens_n = 3  # Set ensemble number (1, 2, or 3)
if(ens_n==1):
    DATA_DIR = DATA_DIR / "ppe_2000_v1_soilc/"
    YEAR_RANGE = "years09-09"  # Update this to match the YEARS variable in analyze_ensemble_timeseries.py
    params = ['fates_allom_l2fr', 'fates_maintresp_leaf_ryan1991_baserate', 'fates_allom_d2bl1', 'fates_leaf_slatop']

elif(ens_n==2):
    DATA_DIR = DATA_DIR / "ppe_2000_v1_c4g/"
    YEAR_RANGE = "years39-39"  # Update this to match the YEARS variable in analyze_ensemble_timeseries.py
    params = ['fates_leaf_vcmax25top', 'fates_leaf_slatop', 'fates_turnover_leaf_canopy', 'fates_landuse_grazing_rate', 'fates_fire_cg_strikes']

elif(ens_n==3):
    DATA_DIR = DATA_DIR / "ppe_2000_v3_ent/"
    YEAR_RANGE = "years29-29"  # Update this to match the YEARS variable in analyze_ensemble_timeseries.py
    params = ['fates_stoich_nitr', 'fates_leaf_vcmax25top',
                  'fates_allom_d2bl1', 'fates_leaf_slatop','fates_grperc']



CSV_DIR = DATA_DIR / "csv_data"  # CSV files are stored in this subdirectory
print('csv dir:',CSV_DIR)
print('csv DATA_DIR:',CSV_DIR)

OUTPUT_DIR = DATA_DIR
OUTPUT_DIR_LAI_EMULATOR = OUTPUT_DIR / "lai_emulator"
OUTPUT_DIR_CUE = OUTPUT_DIR / "cue"
OUTPUT_DIR_GPP = OUTPUT_DIR / "gpp"
OUTPUT_DIR_NPP = OUTPUT_DIR / "npp"

# Create output directories if they don't exist
OUTPUT_DIR_LAI_EMULATOR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_CUE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_GPP.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_NPP.mkdir(parents=True, exist_ok=True)

# Define regions
REGIONS = ['north_65', 'BDTs', 'BETs', 'ENTs', 'AC3G', 'C3G', 'C4']

# Helper function to calculate subplot grid dimensions based on number of parameters
def calculate_subplot_grid(n_params):
    """Calculate optimal grid dimensions for subplots based on number of parameters"""
    if n_params <= 4:
        return 2, 2
    elif n_params <= 6:
        return 2, 3
    elif n_params <= 9:
        return 3, 3
    else:
        return 4, 3

# Year range used in the analysis (from analyze_ble_timeseries.py)


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
obs_lai_regional = {region: np.nan for region in REGIONS}  # Initialize all regions to NaN
obs_lai_regional_avhrr = {region: np.nan for region in REGIONS}  # Initialize all regions to NaN
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

# Read observational data from CSV file
OBS_DATA_CSV = Path("/nird/home/rosief/gt/PPE_analysis/data_extraction/obs_gpp_lai_data.csv")
print(f"\nReading observational data from {OBS_DATA_CSV}...")

obs_gpp_regional = {region: np.nan for region in REGIONS}  # Initialize all regions to NaN
obs_gpp_regional_wecann = {region: np.nan for region in REGIONS}  # Initialize all regions to NaN
obs_gpp_regional_fluxcom_annual_max = {region: np.nan for region in REGIONS}  # For annual max FLUXCOM GPP
obs_gpp_regional_wecann_annual_max = {region: np.nan for region in REGIONS}  # For annual max WECANN GPP
obs_lai_regional = {region: np.nan for region in REGIONS}  # Initialize all regions to NaN
obs_lai_regional_avhrr = {region: np.nan for region in REGIONS}  # Initialize all regions to NaN
obs_lai_regional_avhrr_summer = {region: np.nan for region in REGIONS}  # For summer AVHRR LAI
obs_lai_regional_mean_ent = {}  # For mean (annual) LAI for ENTs
obs_lai_regional_annual_max_ent = {}  # For mean annual maximum LAI for ENTs (MODIS)
obs_lai_regional_annual_max_avhrr_ent = {}  # For mean annual maximum LAI for ENTs (AVHRR)

if OBS_DATA_CSV.exists():
    obs_df = pd.read_csv(OBS_DATA_CSV)
    print(f"  Loaded observational data")
    print(f"  Columns: {list(obs_df.columns)}")
    print(f"  Regions: {list(obs_df['region']) if 'region' in obs_df.columns else 'No region column'}")
    
    # Extract observations for each region
    for _, row in obs_df.iterrows():
        region = row.get('region', None)
        if region in REGIONS:
            # GPP observations - use annual max values for ENTs
            if region == 'ENTs':
                if 'gpp_fluxcom_annual_max' in obs_df.columns and not pd.isna(row['gpp_fluxcom_annual_max']):
                    obs_gpp_regional_fluxcom_annual_max[region] = float(row['gpp_fluxcom_annual_max'])
                    print(f"  {region} FLUXCOM Annual Max GPP: {obs_gpp_regional_fluxcom_annual_max[region]:.4f}")
                if 'gpp_wecann_annual_max' in obs_df.columns and not pd.isna(row['gpp_wecann_annual_max']):
                    obs_gpp_regional_wecann_annual_max[region] = float(row['gpp_wecann_annual_max'])
                    print(f"  {region} WECANN Annual Max GPP: {obs_gpp_regional_wecann_annual_max[region]:.4f}")
            else:
                if 'gpp_fluxcom' in obs_df.columns and not pd.isna(row['gpp_fluxcom']):
                    obs_gpp_regional[region] = float(row['gpp_fluxcom'])
                    print(f"  {region} FLUXCOM GPP: {obs_gpp_regional[region]:.4f}")
                if 'gpp_wecann' in obs_df.columns and not pd.isna(row['gpp_wecann']):
                    obs_gpp_regional_wecann[region] = float(row['gpp_wecann'])
                    print(f"  {region} WECANN GPP: {obs_gpp_regional_wecann[region]:.4f}")
            
            # LAI observations
            # For ENTs, use annual max LAI; for others use regular LAI
            if region == 'ENTs' and 'lai_modis_annual_max' in obs_df.columns and not pd.isna(row['lai_modis_annual_max']):
                obs_lai_regional[region] = float(row['lai_modis_annual_max'])
                print(f"  {region} MODIS Annual Max LAI: {obs_lai_regional[region]:.4f}")
            elif 'lai_modis' in obs_df.columns and not pd.isna(row['lai_modis']):
                obs_lai_regional[region] = float(row['lai_modis'])
                print(f"  {region} MODIS LAI: {obs_lai_regional[region]:.4f}")
                
            if 'lai_avhrr' in obs_df.columns and not pd.isna(row['lai_avhrr']):
                obs_lai_regional_avhrr[region] = float(row['lai_avhrr'])
                print(f"  {region} AVHRR LAI: {obs_lai_regional_avhrr[region]:.4f}")
            
            if 'lai_avhrr_summer' in obs_df.columns and not pd.isna(row['lai_avhrr_summer']):
                obs_lai_regional_avhrr_summer[region] = float(row['lai_avhrr_summer'])
                if region == 'ENTs':
                    print(f"  {region} AVHRR Summer LAI: {obs_lai_regional_avhrr_summer[region]:.4f}")
            
            # ENT-specific LAI observations
            if region == 'ENTs':
                if 'lai_modis' in obs_df.columns and not pd.isna(row['lai_modis']):
                    obs_lai_regional_mean_ent[region] = float(row['lai_modis'])
                    print(f"  {region} MODIS Mean LAI: {obs_lai_regional_mean_ent[region]:.4f}")
                if 'lai_modis_annual_max' in obs_df.columns and not pd.isna(row['lai_modis_annual_max']):
                    obs_lai_regional_annual_max_ent[region] = float(row['lai_modis_annual_max'])
                    print(f"  {region} MODIS Annual Max LAI: {obs_lai_regional_annual_max_ent[region]:.4f}")
                if 'lai_avhrr' in obs_df.columns and not pd.isna(row['lai_avhrr']):
                    obs_lai_regional_mean_ent[f'{region}_avhrr'] = float(row['lai_avhrr'])
                    print(f"  {region} AVHRR Mean LAI: {obs_lai_regional_mean_ent[f'{region}_avhrr']:.4f}")
                if 'lai_avhrr_annual_max' in obs_df.columns and not pd.isna(row['lai_avhrr_annual_max']):
                    obs_lai_regional_annual_max_avhrr_ent[region] = float(row['lai_avhrr_annual_max'])
                    print(f"  {region} AVHRR Annual Max LAI: {obs_lai_regional_annual_max_avhrr_ent[region]:.4f}")
else:
    print(f"  WARNING: Observational data CSV not found: {OBS_DATA_CSV}")
    print(f"  All observations will be set to NaN")

# Load ensemble data and extract grid cells from the CSV files themselves
print("\nLoading ensemble data and grid cell locations from CSV files...")
data = {}
BDT_GRID_CELLS = None
BET_GRID_CELLS = None
ENT_GRID_CELLS = None
AC3G_GRID_CELLS = None
C3G_GRID_CELLS = None
C4_GRID_CELLS = None

for region in REGIONS:
    csv_file = CSV_DIR / f"ensemble_data_{YEAR_RANGE}_{region}.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        data[region] = df
        print(f"  Loaded {region}: {len(df)} ensemble members")
        print(f"    Columns: {list(df.columns)}")
        
        # Extract grid cell coordinates if lat/lon columns exist
        if 'lat' in df.columns and 'lon' in df.columns:
            # Get unique lat/lon pairs (in case there are duplicates across ensemble members)
            unique_coords = df[['lat', 'lon']].drop_duplicates()
            grid_cells = {
                'lats': unique_coords['lat'].values,
                'lons': unique_coords['lon'].values
            }
            
            # Assign to appropriate variable
            if region == 'BDTs':
                BDT_GRID_CELLS = grid_cells
                print(f"    Extracted {len(grid_cells['lats'])} BDT grid cells")
            elif region == 'BETs':
                BET_GRID_CELLS = grid_cells
                print(f"    Extracted {len(grid_cells['lats'])} BET grid cells")
            elif region == 'ENTs':
                ENT_GRID_CELLS = grid_cells
                print(f"    Extracted {len(grid_cells['lats'])} ENT grid cells")
            elif region == 'AC3G':
                AC3G_GRID_CELLS = grid_cells
                print(f"    Extracted {len(grid_cells['lats'])} AC3G grid cells")
            elif region == 'C3G':
                C3G_GRID_CELLS = grid_cells
                print(f"    Extracted {len(grid_cells['lats'])} C3G grid cells")
            elif region == 'C4':
                C4_GRID_CELLS = grid_cells
                print(f"    Extracted {len(grid_cells['lats'])} C4 grid cells")
        else:
            print(f"    WARNING: No 'lat' or 'lon' columns found - cannot extract grid cells for observations")
    else:
        print(f"  WARNING: CSV file not found: {csv_file}")

# Print summary of which grid cells were extracted
print(f"\nGrid cell extraction summary:")
print(f"  BDT_GRID_CELLS: {'Extracted' if BDT_GRID_CELLS is not None else 'NOT extracted (will use REGION_BOUNDS or NaN)'}")
print(f"  BET_GRID_CELLS: {'Extracted' if BET_GRID_CELLS is not None else 'NOT extracted (will use REGION_BOUNDS or NaN)'}")
print(f"  ENT_GRID_CELLS: {'Extracted' if ENT_GRID_CELLS is not None else 'NOT extracted (will use REGION_BOUNDS or NaN)'}")
print(f"  AC3G_GRID_CELLS: {'Extracted' if AC3G_GRID_CELLS is not None else 'NOT extracted (will use REGION_BOUNDS or NaN)'}")
print(f"  C3G_GRID_CELLS: {'Extracted' if C3G_GRID_CELLS is not None else 'NOT extracted (will use REGION_BOUNDS or NaN)'}")
print(f"  C4_GRID_CELLS: {'Extracted' if C4_GRID_CELLS is not None else 'NOT extracted (will use REGION_BOUNDS or NaN)'}")

# Unit conversion factor for FATES_GPP: kgC m-2 s-1 to gC m-2 day-1
# kgC to gC: multiply by 1000
# s-1 to day-1: multiply by 86400 (seconds per day)
GPP_CONVERSION_FACTOR = 1000.0 * 86400.0
print(f"\nGPP unit conversion factor (kgC m-2 s-1 to gC m-2 day-1): {GPP_CONVERSION_FACTOR}")

# Convert GPP and NPP units in the already-loaded data
print("\nConverting GPP and NPP units to gC m-2 day-1...")
for region in REGIONS:
    if region not in data:
        continue
    
    # Convert FATES_GPP variables from kgC m-2 s-1 to gC m-2 day-1
    gpp_vars_to_convert = ['FATES_GPP_PPE_V1', 'FATES_GPP_PPE_2000_V1', 'fates_gpp_summer_PPE_V1', 'fates_gpp_summer_PPE_2000_V1', 'fates_gpp_max_PPE_V1', 'fates_gpp_max_PPE_2000_V1']
    for gpp_var in gpp_vars_to_convert:
        if gpp_var in data[region].columns:
            data[region][gpp_var] = data[region][gpp_var] * GPP_CONVERSION_FACTOR
            print(f"  Converted {gpp_var} for {region} to gC m-2 day-1")
    
    # Convert FATES_NPP variables from kgC m-2 s-1 to gC m-2 day-1 (same units as GPP for CUE calculation)
    npp_vars_to_convert = ['FATES_NPP_PPE_V1', 'FATES_NPP_PPE_2000_V1']
    for npp_var in npp_vars_to_convert:
        if npp_var in data[region].columns:
            data[region][npp_var] = data[region][npp_var] * GPP_CONVERSION_FACTOR
            print(f"  Converted {npp_var} for {region} to gC m-2 day-1")

# Helper function to get the appropriate LAI column name for a region
def get_lai_column(region, df=None):
    """
    Returns the appropriate LAI column for a given region.
    For ENTs (PFT 2), use average annual maximum LAI.
    For other regions, use regular LAI.
    Tries PPE_2000_V1 first, then falls back to PPE_V1 if not available or empty.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
        df: Optional dataframe to check for column availability
    
    Returns:
        Column name string (e.g., 'fates_lai_summer_PPE_2000_V1' or 'FATES_LAI_PPE_V1')
    """
    if region == 'ENTs':
        # Try 2000 version first, then V1 version
        col_2000 = 'fates_lai_max_PPE_2000_V1'
        col_v1 = 'fates_lai_max_PPE_V1'
        if df is not None:
            # Check if 2000 column exists and has non-NaN data
            if col_2000 in df.columns and df[col_2000].notna().any():
                return col_2000
            elif col_v1 in df.columns:
                return col_v1
        # If no df provided, try 2000 first
        return col_2000
    else:
        # Try 2000 version first, then V1 version
        col_2000 = 'FATES_LAI_PPE_2000_V1'
        col_v1 = 'FATES_LAI_PPE_V1'
        if df is not None:
            # Check if 2000 column exists and has non-NaN data
            if col_2000 in df.columns and df[col_2000].notna().any():
                return col_2000
            elif col_v1 in df.columns:
                return col_v1
        # If no df provided, try 2000 first
        return col_2000

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
        return 'Max LAI'
    else:
        return 'LAI'

# Helper function to get GPP column for a region
def get_gpp_column(region, df=None):
    """
    Returns the appropriate GPP column for a given region.
    For ENTs (PFT 2), use average annual maximum GPP.
    For other regions, use regular GPP.
    Tries PPE_2000_V1 first, then falls back to PPE_V1 if not available or empty.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
        df: Optional dataframe to check for column availability
    
    Returns:
        Column name string
    """
    if region == 'ENTs':
        # Try 2000 version first, then V1 version
        col_2000 = 'fates_gpp_max_PPE_2000_V1'
        col_v1 = 'fates_gpp_max_PPE_V1'
        if df is not None:
            # Check if 2000 column exists and has non-NaN data
            if col_2000 in df.columns and df[col_2000].notna().any():
                return col_2000
            elif col_v1 in df.columns:
                return col_v1
            else:
                return None  # Neither column exists
        return col_2000  # Default when df not provided
    else:
        # Try 2000 version first, then V1 version
        col_2000 = 'FATES_GPP_PPE_2000_V1'
        col_v1 = 'FATES_GPP_PPE_V1'
        if df is not None:
            # Check if 2000 column exists and has non-NaN data
            if col_2000 in df.columns and df[col_2000].notna().any():
                return col_2000
            elif col_v1 in df.columns:
                return col_v1
            else:
                return None  # Neither column exists
        return col_2000  # Default when df not provided

# Helper function to get GPP label for plots
def get_gpp_label(region):
    """
    Returns the appropriate GPP label for plot axes and titles.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
    
    Returns:
        Label string
    """
    if region == 'ENTs':
        return 'Max GPP'
    else:
        return 'GPP'

# Helper function to get VEGC column for a region
def get_vegc_column(region, df=None):
    """
    Returns the appropriate VEGC column for a given region.
    Tries PPE_2000_V1 first, then falls back to PPE_V1 if not available or empty.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
        df: Optional dataframe to check for column availability
    
    Returns:
        Column name string or None
    """
    col_2000 = 'FATES_VEGC_PPE_2000_V1'
    col_v1 = 'FATES_VEGC_PPE_V1'
    
    if df is not None:
        # Check if 2000 column exists and has non-NaN data
        if col_2000 in df.columns and df[col_2000].notna().any():
            return col_2000
        elif col_v1 in df.columns and df[col_v1].notna().any():
            return col_v1
        else:
            return None  # Neither column exists or has data
    return col_2000  # Default when df not provided

# Helper function to get NPP column for a region
def get_npp_column(region, df=None):
    """
    Returns the appropriate NPP column for a given region.
    Tries PPE_2000_V1 first, then falls back to PPE_V1 if not available or empty.
    
    Args:
        region: Region name (e.g., 'ENTs', 'BDTs', 'north_65')
        df: Optional dataframe to check for column availability
    
    Returns:
        Column name string or None
    """
    col_2000 = 'FATES_NPP_PPE_2000_V1'
    col_v1 = 'FATES_NPP_PPE_V1'
    
    if df is not None:
        # Check if 2000 column exists and has non-NaN data
        if col_2000 in df.columns and df[col_2000].notna().any():
            return col_2000
        elif col_v1 in df.columns and df[col_v1].notna().any():
            return col_v1
        else:
            return None  # Neither column exists or has data
    return col_2000  # Default when df not provided

# Read selected members (high NPP, low LAI)
selected_data = {}
for region in REGIONS:
    # Try new naming convention first, then fall back to old
    csv_file = CSV_DIR / f"selected_high_NPP_low_LAI_{YEAR_RANGE}_{region}.csv"
    if not csv_file.exists():
        csv_file = CSV_DIR / f"selected_high_NPP_low_LAI_{region}.csv"
    # Also try the old location in DATA_DIR for backwards compatibility
    print('CSV_DIR for selected members:', CSV_DIR)
    if csv_file.exists():
        selected_data[region] = pd.read_csv(csv_file)
        print(f"\nLoaded selected members for {region}: {len(selected_data[region])} members")
    else:
        print(f"\nNo selected members file for {region}")




# Create plots of GPP(2000) vs parameters with observational data and emulator
print("\n" + "="*60)
print("Creating GPP(2000) vs parameter plots with FLUXCOM observations and emulator...")
print("="*60)

# Initialize list to store constrained parameter choices
parameter_choices = []

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    # Use helper function to determine GPP column
    gpp_col = get_gpp_column(region, df)
    if gpp_col is None:
        print(f"\nWARNING: Missing GPP column for {region}")
        continue
    
    print(f"  Using GPP column: {gpp_col}")
    
    # Set time label based on region
    gpp_time_label = '' if region == 'ENTs' else '(2000)' if ens_n == 1 else ''
    
    print(f"\nCreating GPP plots for {region.replace('_', ' ').title()}")
    
    # Debug: Check observational data
    if region == 'ENTs':
        if region in obs_gpp_regional_fluxcom_annual_max:
            print(f"  FLUXCOM Annual Max GPP for {region}: {obs_gpp_regional_fluxcom_annual_max[region]:.4f} gC m-2 day-1")
        if region in obs_gpp_regional_wecann_annual_max:
            print(f"  WECANN Annual Max GPP for {region}: {obs_gpp_regional_wecann_annual_max[region]:.4f} gC m-2 day-1")
    else:
        if region in obs_gpp_regional:
            print(f"  FLUXCOM GPP for {region}: {obs_gpp_regional[region]:.4f} gC m-2 day-1")
        if region in obs_gpp_regional_wecann:
            print(f"  WECANN GPP for {region}: {obs_gpp_regional_wecann[region]:.4f} gC m-2 day-1")
    
    # Check if parameters have data
    params_with_data = [p for p in params if p in df.columns and df[p].notna().any()]
    
    if len(params_with_data) == 0:
        print(f"  WARNING: No parameter data available in CSV. Cannot create emulator or parameter plots.")
        print(f"  Hint: Run analyze_ensemble_timeseries.py first to generate CSVs with parameter values.")
        continue
    
    # Clean data - remove rows with NaN in parameters or GPP
    df_clean = df.dropna(subset=params_with_data + [gpp_col])
    
    if len(df_clean) < 3:
        print(f"  WARNING: Not enough clean data points ({len(df_clean)}) for emulator. Skipping.")
        continue
    
    # Use only params with data for the analysis
    params_to_use = params_with_data
    
    # Prepare training data for emulator
    X = df_clean[params_to_use].values
    y = df_clean[gpp_col].values
    
    # Get LAI data for coloring
    lai_col = get_lai_column(region, df_clean)
    if lai_col in df_clean.columns:
        y_lai = df_clean[lai_col].values
        print(f"  Using {lai_col} for coloring")
    else:
        print(f"  WARNING: LAI column not available, using GPP values for coloring")
        y_lai = y  # Fallback to GPP if LAI not available
    
    # Train emulator for GPP
    print(f"  Training GPP emulator ({EMULATOR_TYPE.upper()}) on {len(df_clean)} ensemble members...")
    lr = get_emulator()
    lr.fit(X, y)
    
    # Train LAI emulator if LAI data is available
    if lai_col in df_clean.columns:
        lr_lai = get_emulator()
        lr_lai.fit(X, y_lai)
        print(f"  Training LAI emulator ({EMULATOR_TYPE.upper()})...")
    
    # Train VEGC emulator if VEGC data is available
    vegc_col = get_vegc_column(region, df_clean)
    lr_vegc = None
    if vegc_col and vegc_col in df_clean.columns:
        y_vegc = df_clean[vegc_col].values
        lr_vegc = get_emulator()
        lr_vegc.fit(X, y_vegc)
        print(f"  Training VEGC emulator ({EMULATOR_TYPE.upper()})...")
        vegc_r2 = lr_vegc.score(X, y_vegc)
        print(f"  VEGC Emulator R²: {vegc_r2:.4f}")
    
    # Train NPP emulator if NPP data is available
    npp_col = get_npp_column(region, df_clean)
    lr_npp = None
    if npp_col and npp_col in df_clean.columns:
        y_npp = df_clean[npp_col].values
        lr_npp = get_emulator()
        lr_npp.fit(X, y_npp)
        print(f"  Training NPP emulator ({EMULATOR_TYPE.upper()})...")
        npp_r2 = lr_npp.score(X, y_npp)
        print(f"  NPP Emulator R²: {npp_r2:.4f}")
    
    # Evaluate emulator skill
    y_pred_train = lr.predict(X)
    r2 = 1 - np.sum((y - y_pred_train)**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean((y - y_pred_train)**2))
    print(f"  GPP Emulator R²: {r2:.4f}")
    print(f"  GPP Emulator RMSE: {rmse:.4f}")
    
    # Generate Latin Hypercube samples for emulator predictions
    n_samples = 3000
    sampler = qmc.LatinHypercube(d=len(params))
    sample = sampler.random(n=n_samples)
    
    # Scale samples to parameter ranges
    param_mins = X.min(axis=0)
    param_maxs = X.max(axis=0)
    lhs_samples = qmc.scale(sample, param_mins, param_maxs)
    
    # Predict GPP for LHS samples
    y_lhs_pred = lr.predict(lhs_samples)
    
    # Predict LAI for LHS samples if LAI emulator is available
    if lai_col in df_clean.columns:
        y_lai_lhs_pred = lr_lai.predict(lhs_samples)
        print(f"  LAI range: {y_lai.min():.4f} to {y_lai.max():.4f}")
        print(f"  LHS predicted LAI range: {y_lai_lhs_pred.min():.4f} to {y_lai_lhs_pred.max():.4f}")
    else:
        y_lai_lhs_pred = y_lhs_pred  # Fallback to GPP
    
    # Predict VEGC for LHS samples if VEGC emulator is available
    y_vegc_lhs_pred = None
    if lr_vegc is not None:
        y_vegc_lhs_pred = lr_vegc.predict(lhs_samples)
        print(f"  VEGC range: {y_vegc.min():.4f} to {y_vegc.max():.4f}")
        print(f"  LHS predicted VEGC range: {y_vegc_lhs_pred.min():.4f} to {y_vegc_lhs_pred.max():.4f}")
    
    # Predict NPP for LHS samples if NPP emulator is available
    y_npp_lhs_pred = None
    if lr_npp is not None:
        y_npp_lhs_pred = lr_npp.predict(lhs_samples)
        print(f"  NPP range: {y_npp.min():.4f} to {y_npp.max():.4f}")
        print(f"  LHS predicted NPP range: {y_npp_lhs_pred.min():.4f} to {y_npp_lhs_pred.max():.4f}")
    
    print(f"  GPP range: {y.min():.4f} to {y.max():.4f}")
    print(f"  LHS predicted GPP range: {y_lhs_pred.min():.4f} to {y_lhs_pred.max():.4f}")
    
    # Debug: Print observation values for this region
    if region == 'ENTs':
        if region in obs_gpp_regional_fluxcom_annual_max and not np.isnan(obs_gpp_regional_fluxcom_annual_max[region]):
            print(f"  Will plot FLUXCOM Annual Max GPP horizontal line at: {obs_gpp_regional_fluxcom_annual_max[region]:.4f}")
        if region in obs_gpp_regional_wecann_annual_max and not np.isnan(obs_gpp_regional_wecann_annual_max[region]):
            print(f"  Will plot WECANN Annual Max GPP horizontal line at: {obs_gpp_regional_wecann_annual_max[region]:.4f}")
    else:
        if region in obs_gpp_regional and not np.isnan(obs_gpp_regional[region]):
            print(f"  Will plot FLUXCOM GPP horizontal line at: {obs_gpp_regional[region]:.4f}")
        if region in obs_gpp_regional_wecann and not np.isnan(obs_gpp_regional_wecann[region]):
            print(f"  Will plot WECANN GPP horizontal line at: {obs_gpp_regional_wecann[region]:.4f}")
    
    # Identify emulator samples within LAI boundaries
    lai_within_bounds_mask = np.zeros(len(y_lai_lhs_pred), dtype=bool)
    if lai_col in df_clean.columns:
        # For ENTs, use summer LAI values
        if region == 'ENTs':
            obs_lai_modis_summer = obs_lai_regional.get(region, np.nan)  # This is already summer LAI for ENTs
            obs_lai_avhrr_summer = obs_lai_regional_avhrr_summer.get(region, np.nan)
            
            if not np.isnan(obs_lai_modis_summer) and not np.isnan(obs_lai_avhrr_summer):
                # Define boundaries as min and max of the two summer observations
                lai_min = min(obs_lai_modis_summer, obs_lai_avhrr_summer)
                lai_max = max(obs_lai_modis_summer, obs_lai_avhrr_summer)
                lai_within_bounds_mask = (y_lai_lhs_pred >= lai_min) & (y_lai_lhs_pred <= lai_max)
                n_within_bounds = np.sum(lai_within_bounds_mask)
                print(f"  LAI boundaries (using summer): [{lai_min:.4f}, {lai_max:.4f}]")
                print(f"  Emulator samples within LAI bounds: {n_within_bounds}/{len(y_lai_lhs_pred)}")
        else:
            # For other regions, use regular LAI observations
            obs_lai_modis = obs_lai_regional.get(region, np.nan)
            obs_lai_avhrr_val = obs_lai_regional_avhrr.get(region, np.nan)
            
            if not np.isnan(obs_lai_modis) and not np.isnan(obs_lai_avhrr_val):
                # Define boundaries as min and max of the two observations
                lai_min = min(obs_lai_modis, obs_lai_avhrr_val)
                lai_max = max(obs_lai_modis, obs_lai_avhrr_val)
                lai_within_bounds_mask = (y_lai_lhs_pred >= lai_min) & (y_lai_lhs_pred <= lai_max)
                n_within_bounds = np.sum(lai_within_bounds_mask)
                print(f"  LAI boundaries: [{lai_min:.4f}, {lai_max:.4f}]")
                print(f"  Emulator samples within LAI bounds: {n_within_bounds}/{len(y_lai_lhs_pred)}")
    
    # Identify emulator samples within BOTH LAI and GPP boundaries (black dots)
    gpp_lai_within_bounds_mask = np.zeros(len(y_lai_lhs_pred), dtype=bool)
    gpp_only_within_bounds_mask = np.zeros(len(y_lai_lhs_pred), dtype=bool)
    
    # Get GPP observations based on region type
    if region == 'ENTs':
        obs_gpp_fluxcom = obs_gpp_regional_fluxcom_annual_max.get(region, np.nan)
        obs_gpp_wecann_val = obs_gpp_regional_wecann_annual_max.get(region, np.nan)
    else:
        obs_gpp_fluxcom = obs_gpp_regional.get(region, np.nan)
        obs_gpp_wecann_val = obs_gpp_regional_wecann.get(region, np.nan)
    
    if not np.isnan(obs_gpp_fluxcom) and not np.isnan(obs_gpp_wecann_val):
        # Define GPP boundaries as min and max of the two observations
        gpp_min = min(obs_gpp_fluxcom, obs_gpp_wecann_val)
        gpp_max = max(obs_gpp_fluxcom, obs_gpp_wecann_val)
        gpp_within_bounds = (y_lhs_pred >= gpp_min) & (y_lhs_pred <= gpp_max)
        
        # Samples within BOTH LAI and GPP bounds
        if np.any(lai_within_bounds_mask):
            gpp_lai_within_bounds_mask = lai_within_bounds_mask & gpp_within_bounds
            n_within_both = np.sum(gpp_lai_within_bounds_mask)
            print(f"  GPP boundaries: [{gpp_min:.4f}, {gpp_max:.4f}]")
            print(f"  Emulator samples within BOTH LAI and GPP bounds: {n_within_both}/{len(y_lai_lhs_pred)}")
        
        # Samples within GPP bounds only (not LAI)
        gpp_only_within_bounds_mask = gpp_within_bounds & ~lai_within_bounds_mask
        n_gpp_only = np.sum(gpp_only_within_bounds_mask)
        print(f"  Emulator samples within GPP bounds only: {n_gpp_only}/{len(y_lai_lhs_pred)}")
    
    # Store parameter samples for plotting
    param_samples = {}
    param_predictions = {}
    
    for i, param in enumerate(params):
        param_samples[param] = lhs_samples[:, i]
        param_predictions[param] = y_lhs_pred
    
    # Create dynamic subplot grid based on number of parameters
    n_rows, n_cols = calculate_subplot_grid(len(params))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes = axes.flatten() if len(params) > 1 else [axes]
    fig.suptitle(f'GPP{gpp_time_label} vs Parameters (colored by LAI) - {region.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    scatter_emul = None  # Initialize outside loop
    for idx, param in enumerate(params):
        ax = axes[idx]
        
        if param not in df.columns:
            ax.text(0.5, 0.5, f'{param}\nNot found', ha='center', va='center', 
                   transform=ax.transAxes)
            continue
        
        # Plot emulator predictions NOT within LAI bounds colored by predicted LAI
        if np.any(~lai_within_bounds_mask):
            scatter_emul = ax.scatter(param_samples[param][~lai_within_bounds_mask], 
                      param_predictions[param][~lai_within_bounds_mask], s=15, alpha=0.6, 
                      c=y_lai_lhs_pred[~lai_within_bounds_mask], cmap='YlGn', edgecolors='none', zorder=1, 
                      vmin=y_lai.min(), vmax=y_lai.max(),
                      label='Emulator predictions')
        
        # Plot emulator predictions WITHIN LAI bounds only (not GPP) in pink
        lai_only_mask = lai_within_bounds_mask & ~gpp_lai_within_bounds_mask
        if np.any(lai_only_mask):
            ax.scatter(param_samples[param][lai_only_mask], 
                      param_predictions[param][lai_only_mask], s=15, alpha=0.7, 
                      c='hotpink', edgecolors='none', zorder=1.5,
                      label='Within LAI bounds')
        
        # Plot emulator predictions WITHIN BOTH LAI and GPP bounds in black
        if np.any(gpp_lai_within_bounds_mask):
            ax.scatter(param_samples[param][gpp_lai_within_bounds_mask], 
                      param_predictions[param][gpp_lai_within_bounds_mask], s=20, alpha=0.8, 
                      c='black', edgecolors='none', zorder=1.6,
                      label='Within LAI & GPP bounds')
        
        # Plot GPP(2000) colored by actual LAI on top
        ax.scatter(df[param], df[gpp_col], s=100, alpha=0.9, 
                   c=df[lai_col] if lai_col in df.columns else df[gpp_col], 
                   cmap='YlGn', edgecolors='black', linewidth=1.5, zorder=2, 
                   vmin=y_lai.min(), vmax=y_lai.max(),
                   label=f'Model GPP{gpp_time_label}')
        
        # Add observed GPP as horizontal line
        if region == 'ENTs':
            # Add grey shading between the two observational estimates
            if (region in obs_gpp_regional_fluxcom_annual_max and not np.isnan(obs_gpp_regional_fluxcom_annual_max[region]) and
                region in obs_gpp_regional_wecann_annual_max and not np.isnan(obs_gpp_regional_wecann_annual_max[region])):
                gpp_min = min(obs_gpp_regional_fluxcom_annual_max[region], obs_gpp_regional_wecann_annual_max[region])
                gpp_max = max(obs_gpp_regional_fluxcom_annual_max[region], obs_gpp_regional_wecann_annual_max[region])
                ax.axhspan(gpp_min, gpp_max, color='grey', alpha=0.2, zorder=0.5)
            
            if region in obs_gpp_regional_fluxcom_annual_max and not np.isnan(obs_gpp_regional_fluxcom_annual_max[region]):
                print(f"  Adding FLUXCOM horizontal line at y={obs_gpp_regional_fluxcom_annual_max[region]:.4f}")
                ax.axhline(obs_gpp_regional_fluxcom_annual_max[region], color='darkred', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=3, label=f'FLUXCOM Annual Max GPP: {obs_gpp_regional_fluxcom_annual_max[region]:.2f}')
            if region in obs_gpp_regional_wecann_annual_max and not np.isnan(obs_gpp_regional_wecann_annual_max[region]):
                print(f"  Adding WECANN horizontal line at y={obs_gpp_regional_wecann_annual_max[region]:.4f}")
                ax.axhline(obs_gpp_regional_wecann_annual_max[region], color='coral', linestyle=':', 
                          linewidth=2, alpha=0.8, zorder=3, label=f'WECANN Annual Max GPP: {obs_gpp_regional_wecann_annual_max[region]:.2f}')
        else:
            # Add grey shading between the two observational estimates
            if (region in obs_gpp_regional and not np.isnan(obs_gpp_regional[region]) and
                region in obs_gpp_regional_wecann and not np.isnan(obs_gpp_regional_wecann[region])):
                gpp_min = min(obs_gpp_regional[region], obs_gpp_regional_wecann[region])
                gpp_max = max(obs_gpp_regional[region], obs_gpp_regional_wecann[region])
                ax.axhspan(gpp_min, gpp_max, color='grey', alpha=0.2, zorder=0.5)
            
            if region in obs_gpp_regional and not np.isnan(obs_gpp_regional[region]):
                print(f"  Adding FLUXCOM horizontal line at y={obs_gpp_regional[region]:.4f}")
                ax.axhline(obs_gpp_regional[region], color='darkred', linestyle='--', 
                          linewidth=2, alpha=0.8, zorder=3, label=f'FLUXCOM GPP: {obs_gpp_regional[region]:.2f}')
            if region in obs_gpp_regional_wecann and not np.isnan(obs_gpp_regional_wecann[region]):
                print(f"  Adding WECANN horizontal line at y={obs_gpp_regional_wecann[region]:.4f}")
                ax.axhline(obs_gpp_regional_wecann[region], color='coral', linestyle=':', 
                          linewidth=2, alpha=0.8, zorder=3, label=f'WECANN GPP: {obs_gpp_regional_wecann[region]:.2f}')
        
        # Add ensemble member labels
        for _, row_data in df.iterrows():
            ax.annotate(f"n{int(row_data['ensemble_member']):02d}", 
                       (row_data[param], row_data[gpp_col]),
                       fontsize=7, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='none', alpha=0.7))
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=11, fontweight='bold')
        gpp_label = get_gpp_label(region)
        ax.set_ylabel(f'{gpp_label}{gpp_time_label} [gC m⁻² day⁻¹]', fontsize=11, fontweight='bold')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_handles = []
        legend_labels = []
        if region in obs_gpp_regional and not np.isnan(obs_gpp_regional[region]):
            fluxcom_line = Line2D([0], [0], color='darkred', linestyle='--', linewidth=2, alpha=0.8)
            legend_handles.append(fluxcom_line)
            legend_labels.append(f'FLUXCOM GPP: {obs_gpp_regional[region]:.2f}')
        if region in obs_gpp_regional_wecann and not np.isnan(obs_gpp_regional_wecann[region]):
            wecann_line = Line2D([0], [0], color='coral', linestyle=':', linewidth=2, alpha=0.8)
            legend_handles.append(wecann_line)
            legend_labels.append(f'WECANN GPP: {obs_gpp_regional_wecann[region]:.2f}')
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='best', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    # Add colorbar for LAI values
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if scatter_emul is not None:
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter_emul, cax=cbar_ax)
        cbar.set_label('LAI [m²/m²]', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_GPP / f"gpp_vs_parameters_{region}_{EMULATOR_TYPE}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_file}")
    plt.close(fig)
    
    # Create parameter space visualization (slatop vs vcmax25top) showing emulator space and constrained regions
    if 'fates_leaf_slatop' in params and 'fates_leaf_vcmax25top' in params:
        print(f"  Creating parameter space visualization (slatop vs vcmax25top)...")
        
        slatop_idx = params.index('fates_leaf_slatop')
        vcmax_idx = params.index('fates_leaf_vcmax25top')
        
        fig_param_space, ax_space = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot all emulator samples (light blue)
        ax_space.scatter(lhs_samples[:, slatop_idx], lhs_samples[:, vcmax_idx], 
                        s=15, alpha=0.3, c='lightblue', edgecolors='none', 
                        label='All emulator space', zorder=1)
        
        # Plot samples within GPP bounds only (green)
        if np.any(gpp_only_within_bounds_mask):
            ax_space.scatter(lhs_samples[gpp_only_within_bounds_mask, slatop_idx], 
                           lhs_samples[gpp_only_within_bounds_mask, vcmax_idx],
                           s=25, alpha=0.7, c='limegreen', edgecolors='none',
                           label=f'Within GPP bounds ({np.sum(gpp_only_within_bounds_mask)} samples)', zorder=2)
        
        # Plot samples within LAI bounds only (pink)
        lai_only_mask = lai_within_bounds_mask & ~gpp_lai_within_bounds_mask
        if np.any(lai_only_mask):
            ax_space.scatter(lhs_samples[lai_only_mask, slatop_idx], 
                           lhs_samples[lai_only_mask, vcmax_idx],
                           s=25, alpha=0.7, c='hotpink', edgecolors='none',
                           label=f'Within LAI bounds ({np.sum(lai_only_mask)} samples)', zorder=2)
        
        # Plot samples within BOTH LAI and GPP bounds (black)
        if np.any(gpp_lai_within_bounds_mask):
            ax_space.scatter(lhs_samples[gpp_lai_within_bounds_mask, slatop_idx], 
                           lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx],
                           s=30, alpha=0.9, c='black', edgecolors='yellow', linewidth=0.5,
                           label=f'Within LAI & GPP bounds ({np.sum(gpp_lai_within_bounds_mask)} samples)', zorder=3)
            
            # Calculate and plot mean of black points
            mean_slatop = np.mean(lhs_samples[gpp_lai_within_bounds_mask, slatop_idx])
            mean_vcmax = np.mean(lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx])
            std_slatop = np.std(lhs_samples[gpp_lai_within_bounds_mask, slatop_idx])
            std_vcmax = np.std(lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx])
            print(f"  Mean of constrained points: slatop={mean_slatop:.4f}, vcmax25top={mean_vcmax:.4f}")
            print(f"  Std of constrained points: slatop={std_slatop:.4f}, vcmax25top={std_vcmax:.4f}")
            
            # Store parameter choices for CSV output
            parameter_choices.append({
                'PFT': region,
                'vcmax25top': mean_vcmax,
                'slatop': mean_slatop,
                'vcmax25top_std': std_vcmax,
                'slatop_std': std_slatop,
                'n_constrained_points': np.sum(gpp_lai_within_bounds_mask)
            })
            
            ax_space.scatter(mean_slatop, mean_vcmax,
                           s=400, alpha=1.0, c='black', edgecolors='gold', linewidth=3,
                           marker='*', label=f'Mean of constrained points', zorder=5)
            
            # Add text annotation with parameter values
            ax_space.annotate(f'slatop={mean_slatop:.3f}\nvcmax25top={mean_vcmax:.2f}',
                            (mean_slatop, mean_vcmax),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                     edgecolor='black', alpha=0.8),
                            ha='left', va='bottom', zorder=6)
        else:
            print(f"  No points within both LAI and GPP bounds - no mean star plotted")
        
        # Plot actual ensemble members
        ax_space.scatter(df_clean['fates_leaf_slatop'], df_clean['fates_leaf_vcmax25top'],
                        s=150, alpha=0.9, c='red', edgecolors='darkred', linewidth=2,
                        marker='*', label='Ensemble members', zorder=4)
        
        # Add ensemble labels
        for _, row in df_clean.iterrows():
            ax_space.annotate(f"n{int(row['ensemble_member']):02d}",
                            (row['fates_leaf_slatop'], row['fates_leaf_vcmax25top']),
                            fontsize=9, ha='center', va='bottom', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor='darkred', alpha=0.8))
        
        ax_space.set_xlabel('fates_leaf_slatop', fontsize=13, fontweight='bold')
        ax_space.set_ylabel('fates_leaf_vcmax25top', fontsize=13, fontweight='bold')
        ax_space.set_title(f'Parameter Space: Slatop vs Vcmax25top - {region.replace("_", " ").title()}\\n' +
                          f'Constrained by LAI and GPP observations',
                          fontsize=14, fontweight='bold')
        ax_space.legend(loc='best', fontsize=10, framealpha=0.9)
        ax_space.grid(True, alpha=0.3)
        
        output_file_param = OUTPUT_DIR_GPP / f"parameter_space_slatop_vcmax_{region}_{EMULATOR_TYPE}.png"
        fig_param_space.savefig(output_file_param, dpi=300, bbox_inches='tight')
        print(f"  Saved parameter space plot: {output_file_param}")
        plt.close(fig_param_space)
    else:
        print(f"  Skipping parameter space plot (requires both slatop and vcmax25top)")
    
    # Create 3D parameter space visualization (slatop vs vcmax25top vs stoich_nitr)
    if 'fates_leaf_slatop' in params and 'fates_leaf_vcmax25top' in params and 'fates_stoich_nitr' in params:
        print(f"  Creating 3D parameter space visualization (slatop vs vcmax25top vs stoich_nitr)...")
        
        slatop_idx = params.index('fates_leaf_slatop')
        vcmax_idx = params.index('fates_leaf_vcmax25top')
        stoich_idx = params.index('fates_stoich_nitr')
        
        fig_3d = plt.figure(figsize=(12, 10))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # Plot all emulator samples (light blue)
        ax_3d.scatter(lhs_samples[:, slatop_idx], lhs_samples[:, vcmax_idx], lhs_samples[:, stoich_idx],
                     s=10, alpha=0.2, c='lightblue', edgecolors='none', 
                     label='All emulator space', zorder=1)
        
        # Plot samples within GPP bounds only (green)
        if np.any(gpp_only_within_bounds_mask):
            ax_3d.scatter(lhs_samples[gpp_only_within_bounds_mask, slatop_idx], 
                         lhs_samples[gpp_only_within_bounds_mask, vcmax_idx],
                         lhs_samples[gpp_only_within_bounds_mask, stoich_idx],
                         s=20, alpha=0.6, c='limegreen', edgecolors='none',
                         label=f'Within GPP bounds ({np.sum(gpp_only_within_bounds_mask)} samples)', zorder=2)
        
        # Plot samples within LAI bounds only (pink)
        lai_only_mask = lai_within_bounds_mask & ~gpp_lai_within_bounds_mask
        if np.any(lai_only_mask):
            ax_3d.scatter(lhs_samples[lai_only_mask, slatop_idx], 
                         lhs_samples[lai_only_mask, vcmax_idx],
                         lhs_samples[lai_only_mask, stoich_idx],
                         s=20, alpha=0.6, c='hotpink', edgecolors='none',
                         label=f'Within LAI bounds ({np.sum(lai_only_mask)} samples)', zorder=2)
        
        # Plot samples within BOTH LAI and GPP bounds (black)
        if np.any(gpp_lai_within_bounds_mask):
            ax_3d.scatter(lhs_samples[gpp_lai_within_bounds_mask, slatop_idx], 
                         lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx],
                         lhs_samples[gpp_lai_within_bounds_mask, stoich_idx],
                         s=30, alpha=0.9, c='black', edgecolors='yellow', linewidth=0.5,
                         label=f'Within LAI & GPP bounds ({np.sum(gpp_lai_within_bounds_mask)} samples)', zorder=3)
            
            # Calculate and plot mean of black points
            mean_slatop = np.mean(lhs_samples[gpp_lai_within_bounds_mask, slatop_idx])
            mean_vcmax = np.mean(lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx])
            mean_stoich = np.mean(lhs_samples[gpp_lai_within_bounds_mask, stoich_idx])
            
            ax_3d.scatter(mean_slatop, mean_vcmax, mean_stoich,
                         s=500, alpha=1.0, c='black', edgecolors='gold', linewidth=3,
                         marker='*', label=f'Mean of constrained points', zorder=5)
            
            # Add text annotation with parameter values (positioned above points)
            z_range = np.max(lhs_samples[gpp_lai_within_bounds_mask, stoich_idx]) - np.min(lhs_samples[gpp_lai_within_bounds_mask, stoich_idx])
            text_z_offset = z_range * 0.15  # Position 15% above the data range
            ax_3d.text(mean_slatop, mean_vcmax, mean_stoich + text_z_offset,
                      f'  slatop={mean_slatop:.3f}\n  vcmax={mean_vcmax:.2f}\n  stoich_nitr={mean_stoich:.3f}',
                      fontsize=9, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                               edgecolor='black', alpha=0.8), zorder=10)
        
        # Plot actual ensemble members
        if 'fates_stoich_nitr' in df_clean.columns:
            ax_3d.scatter(df_clean['fates_leaf_slatop'], df_clean['fates_leaf_vcmax25top'],
                         df_clean['fates_stoich_nitr'],
                         s=150, alpha=0.9, c='red', edgecolors='darkred', linewidth=2,
                         marker='*', label='Ensemble members', zorder=4)
            
            # Add ensemble labels
            for _, row in df_clean.iterrows():
                ax_3d.text(row['fates_leaf_slatop'], row['fates_leaf_vcmax25top'], 
                          row['fates_stoich_nitr'],
                          f"n{int(row['ensemble_member']):02d}",
                          fontsize=8, ha='center', va='bottom', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor='darkred', alpha=0.8))
        
        ax_3d.set_xlabel('fates_leaf_slatop', fontsize=11, fontweight='bold')
        ax_3d.set_ylabel('fates_leaf_vcmax25top', fontsize=11, fontweight='bold')
        ax_3d.set_zlabel('fates_stoich_nitr', fontsize=11, fontweight='bold')
        ax_3d.set_title(f'3D Parameter Space - {region.replace("_", " ").title()}\\n' +
                       f'Constrained by LAI and GPP observations',
                       fontsize=14, fontweight='bold', pad=20)
        ax_3d.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax_3d.grid(True, alpha=0.3)
        
        # Set viewing angle for better visibility
        ax_3d.view_init(elev=20, azim=45)
        
        output_file_3d = OUTPUT_DIR_GPP / f"parameter_space_3d_{region}_{EMULATOR_TYPE}.png"
        fig_3d.savefig(output_file_3d, dpi=300, bbox_inches='tight')
        print(f"  Saved 3D parameter space plot: {output_file_3d}")
        plt.close(fig_3d)
        
        # Create 3D plot of constrained points colored by d2bl1
        if 'fates_allom_d2bl1' in params and np.any(gpp_lai_within_bounds_mask):
            print(f"  Creating 3D constrained parameter space colored by d2bl1...")
            
            d2bl1_idx = params.index('fates_allom_d2bl1')
            
            # Get the constrained points
            constrained_slatop = lhs_samples[gpp_lai_within_bounds_mask, slatop_idx]
            constrained_vcmax = lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx]
            constrained_stoich = lhs_samples[gpp_lai_within_bounds_mask, stoich_idx]
            constrained_d2bl1 = lhs_samples[gpp_lai_within_bounds_mask, d2bl1_idx]
            
            fig_3d_d2bl1 = plt.figure(figsize=(12, 10))
            ax_3d_d2bl1 = fig_3d_d2bl1.add_subplot(111, projection='3d')
            
            # Plot constrained points colored by d2bl1
            scatter = ax_3d_d2bl1.scatter(constrained_slatop, constrained_vcmax, constrained_stoich,
                                         s=30, alpha=0.8, c=constrained_d2bl1, cmap='viridis', 
                                         edgecolors='black', linewidth=0.5, zorder=3)
            
            # Add colorbar for d2bl1
            cbar = plt.colorbar(scatter, ax=ax_3d_d2bl1, pad=0.1, shrink=0.8)
            cbar.set_label('fates_allom_d2bl1', fontsize=11, fontweight='bold')
            
            # Calculate and plot mean
            mean_slatop = np.mean(constrained_slatop)
            mean_vcmax = np.mean(constrained_vcmax)
            mean_stoich = np.mean(constrained_stoich)
            mean_d2bl1 = np.mean(constrained_d2bl1)
            
            ax_3d_d2bl1.scatter(mean_slatop, mean_vcmax, mean_stoich,
                               s=500, alpha=1.0, c='red', edgecolors='gold', linewidth=3,
                               marker='*', label=f'Mean', zorder=5)
            
            # Add text annotation (positioned above points)
            z_range = np.max(constrained_stoich) - np.min(constrained_stoich)
            text_z_offset = z_range * 0.15  # Position 15% above the data range
            ax_3d_d2bl1.text(mean_slatop, mean_vcmax, mean_stoich + text_z_offset,
                          f'  Mean:\n  slatop={mean_slatop:.3f}\n  vcmax={mean_vcmax:.2f}\n  stoich_nitr={mean_stoich:.3f}\n  d2bl1={mean_d2bl1:.3f}',
                          fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='black', alpha=0.9), zorder=10)
            
            # Plot actual ensemble members
            if 'fates_stoich_nitr' in df_clean.columns:
                ax_3d_d2bl1.scatter(df_clean['fates_leaf_slatop'], df_clean['fates_leaf_vcmax25top'],
                                   df_clean['fates_stoich_nitr'],
                                   s=150, alpha=0.9, c='red', edgecolors='darkred', linewidth=2,
                                   marker='*', label='Ensemble members', zorder=4)
                
                # Add ensemble labels
                for _, row in df_clean.iterrows():
                    ax_3d_d2bl1.text(row['fates_leaf_slatop'], row['fates_leaf_vcmax25top'], 
                                    row['fates_stoich_nitr'],
                                    f"n{int(row['ensemble_member']):02d}",
                                    fontsize=8, ha='center', va='bottom', fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                             edgecolor='darkred', alpha=0.8))
            
            ax_3d_d2bl1.set_xlabel('fates_leaf_slatop', fontsize=11, fontweight='bold')
            ax_3d_d2bl1.set_ylabel('fates_leaf_vcmax25top', fontsize=11, fontweight='bold')
            ax_3d_d2bl1.set_zlabel('fates_stoich_nitr', fontsize=11, fontweight='bold')
            ax_3d_d2bl1.set_title(f'3D Constrained Parameter Space (colored by d2bl1) - {region.replace("_", " ").title()}\\n' +
                                 f'Points within both LAI and GPP bounds',
                                 fontsize=14, fontweight='bold', pad=20)
            ax_3d_d2bl1.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax_3d_d2bl1.grid(True, alpha=0.3)
            
            # Set viewing angle for better visibility
            ax_3d_d2bl1.view_init(elev=20, azim=45)
            
            output_file_3d_d2bl1 = OUTPUT_DIR_GPP / f"parameter_space_3d_d2bl1_{region}_{EMULATOR_TYPE}.png"
            fig_3d_d2bl1.savefig(output_file_3d_d2bl1, dpi=300, bbox_inches='tight')
            print(f"  Saved 3D parameter space (colored by d2bl1) plot: {output_file_3d_d2bl1}")
            plt.close(fig_3d_d2bl1)
        else:
            if 'fates_allom_d2bl1' not in params:
                print(f"  Skipping 3D d2bl1-colored plot (fates_allom_d2bl1 not in parameters)")
            elif not np.any(gpp_lai_within_bounds_mask):
                print(f"  Skipping 3D d2bl1-colored plot (no constrained points)")
        
        # Create 3D plot of constrained points colored by emulated VEGC
        if y_vegc_lhs_pred is not None and np.any(gpp_lai_within_bounds_mask):
            print(f"  Creating 3D constrained parameter space colored by emulated VEGC...")
            
            # Get the constrained points
            constrained_slatop = lhs_samples[gpp_lai_within_bounds_mask, slatop_idx]
            constrained_vcmax = lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx]
            constrained_stoich = lhs_samples[gpp_lai_within_bounds_mask, stoich_idx]
            constrained_vegc = y_vegc_lhs_pred[gpp_lai_within_bounds_mask]
            
            fig_3d_vegc = plt.figure(figsize=(12, 10))
            ax_3d_vegc = fig_3d_vegc.add_subplot(111, projection='3d')
            
            # Plot constrained points colored by emulated VEGC
            scatter_vegc = ax_3d_vegc.scatter(constrained_slatop, constrained_vcmax, constrained_stoich,
                                             s=30, alpha=0.8, c=constrained_vegc, cmap='plasma', 
                                             edgecolors='black', linewidth=0.5, zorder=3)
            
            # Add colorbar for VEGC
            cbar_vegc = plt.colorbar(scatter_vegc, ax=ax_3d_vegc, pad=0.1, shrink=0.8)
            cbar_vegc.set_label('Emulated VEGC [kgC/m²]', fontsize=11, fontweight='bold')
            
            # Calculate and plot mean
            mean_slatop = np.mean(constrained_slatop)
            mean_vcmax = np.mean(constrained_vcmax)
            mean_stoich = np.mean(constrained_stoich)
            mean_vegc = np.mean(constrained_vegc)
            
            ax_3d_vegc.scatter(mean_slatop, mean_vcmax, mean_stoich,
                              s=500, alpha=1.0, c='red', edgecolors='gold', linewidth=3,
                              marker='*', label=f'Mean', zorder=5)
            
            # Add text annotation (positioned above points)
            z_range = np.max(constrained_stoich) - np.min(constrained_stoich)
            text_z_offset = z_range * 0.15  # Position 15% above the data range
            ax_3d_vegc.text(mean_slatop, mean_vcmax, mean_stoich + text_z_offset,
                          f'  Mean:\n  slatop={mean_slatop:.3f}\n  vcmax={mean_vcmax:.2f}\n  stoich_nitr={mean_stoich:.3f}\n  VEGC={mean_vegc:.2f}',
                          fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='black', alpha=0.9), zorder=10)
            
            # Plot actual ensemble members
            if 'fates_stoich_nitr' in df_clean.columns:
                ax_3d_vegc.scatter(df_clean['fates_leaf_slatop'], df_clean['fates_leaf_vcmax25top'],
                                  df_clean['fates_stoich_nitr'],
                                  s=150, alpha=0.9, c='red', edgecolors='darkred', linewidth=2,
                                  marker='*', label='Ensemble members', zorder=4)
                
                # Add ensemble labels
                for _, row in df_clean.iterrows():
                    ax_3d_vegc.text(row['fates_leaf_slatop'], row['fates_leaf_vcmax25top'], 
                                   row['fates_stoich_nitr'],
                                   f"n{int(row['ensemble_member']):02d}",
                                   fontsize=8, ha='center', va='bottom', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                            edgecolor='darkred', alpha=0.8))
            
            ax_3d_vegc.set_xlabel('fates_leaf_slatop', fontsize=11, fontweight='bold')
            ax_3d_vegc.set_ylabel('fates_leaf_vcmax25top', fontsize=11, fontweight='bold')
            ax_3d_vegc.set_zlabel('fates_stoich_nitr', fontsize=11, fontweight='bold')
            ax_3d_vegc.set_title(f'3D Constrained Parameter Space (colored by emulated VEGC) - {region.replace("_", " ").title()}\\n' +
                                f'Points within both LAI and GPP bounds',
                                fontsize=14, fontweight='bold', pad=20)
            ax_3d_vegc.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax_3d_vegc.grid(True, alpha=0.3)
            
            # Set viewing angle for better visibility
            ax_3d_vegc.view_init(elev=20, azim=45)
            
            output_file_3d_vegc = OUTPUT_DIR_GPP / f"parameter_space_3d_vegc_{region}_{EMULATOR_TYPE}.png"
            fig_3d_vegc.savefig(output_file_3d_vegc, dpi=300, bbox_inches='tight')
            print(f"  Saved 3D parameter space (colored by emulated VEGC) plot: {output_file_3d_vegc}")
            plt.close(fig_3d_vegc)
        else:
            if y_vegc_lhs_pred is None:
                print(f"  Skipping 3D VEGC-colored plot (VEGC emulator not available)")
            elif not np.any(gpp_lai_within_bounds_mask):
                print(f"  Skipping 3D VEGC-colored plot (no constrained points)")
        
        # Create 3D plot of constrained points colored by emulated NPP
        if y_npp_lhs_pred is not None and np.any(gpp_lai_within_bounds_mask):
            print(f"  Creating 3D constrained parameter space colored by emulated NPP...")
            
            # Get the constrained points
            constrained_slatop = lhs_samples[gpp_lai_within_bounds_mask, slatop_idx]
            constrained_vcmax = lhs_samples[gpp_lai_within_bounds_mask, vcmax_idx]
            constrained_stoich = lhs_samples[gpp_lai_within_bounds_mask, stoich_idx]
            constrained_npp = y_npp_lhs_pred[gpp_lai_within_bounds_mask]
            
            fig_3d_npp = plt.figure(figsize=(12, 10))
            ax_3d_npp = fig_3d_npp.add_subplot(111, projection='3d')
            
            # Plot constrained points colored by emulated NPP
            scatter_npp = ax_3d_npp.scatter(constrained_slatop, constrained_vcmax, constrained_stoich,
                                           s=30, alpha=0.8, c=constrained_npp, cmap='plasma', 
                                           edgecolors='black', linewidth=0.5, zorder=3)
            
            # Add colorbar for NPP
            cbar_npp = plt.colorbar(scatter_npp, ax=ax_3d_npp, pad=0.1, shrink=0.8)
            cbar_npp.set_label('Emulated NPP (gC m$^{-2}$ day$^{-1}$)', fontsize=11, fontweight='bold')
            
            # Calculate and plot mean
            mean_slatop = np.mean(constrained_slatop)
            mean_vcmax = np.mean(constrained_vcmax)
            mean_stoich = np.mean(constrained_stoich)
            mean_npp = np.mean(constrained_npp)
            
            ax_3d_npp.scatter(mean_slatop, mean_vcmax, mean_stoich,
                             s=500, alpha=1.0, c='red', edgecolors='gold', linewidth=3,
                             marker='*', label=f'Mean', zorder=5)
            
            # Add text annotation (positioned above points)
            z_range = np.max(constrained_stoich) - np.min(constrained_stoich)
            text_z_offset = z_range * 0.15  # Position 15% above the data range
            ax_3d_npp.text(mean_slatop, mean_vcmax, mean_stoich + text_z_offset,
                          f'  Mean:\n  slatop={mean_slatop:.3f}\n  vcmax={mean_vcmax:.2f}\n  stoich_nitr={mean_stoich:.3f}\n  NPP={mean_npp:.2f}',
                          fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='black', alpha=0.9), zorder=10)
            
            # Plot actual ensemble members
            if 'fates_stoich_nitr' in df_clean.columns:
                ax_3d_npp.scatter(df_clean['fates_leaf_slatop'], df_clean['fates_leaf_vcmax25top'],
                                 df_clean['fates_stoich_nitr'],
                                 s=150, alpha=0.9, c='red', edgecolors='darkred', linewidth=2,
                                 marker='*', label='Ensemble members', zorder=4)
                
                # Add ensemble labels
                for _, row in df_clean.iterrows():
                    ax_3d_npp.text(row['fates_leaf_slatop'], row['fates_leaf_vcmax25top'], 
                                  row['fates_stoich_nitr'],
                                  f"n{int(row['ensemble_member']):02d}",
                                  fontsize=8, ha='center', va='bottom', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                           edgecolor='darkred', alpha=0.8))
            
            ax_3d_npp.set_xlabel('fates_leaf_slatop', fontsize=11, fontweight='bold')
            ax_3d_npp.set_ylabel('fates_leaf_vcmax25top', fontsize=11, fontweight='bold')
            ax_3d_npp.set_zlabel('fates_stoich_nitr', fontsize=11, fontweight='bold')
            ax_3d_npp.set_title(f'3D Constrained Parameter Space (colored by emulated NPP) - {region.replace("_", " ").title()}\n' +
                               f'Points within both LAI and GPP bounds',
                               fontsize=14, fontweight='bold', pad=20)
            ax_3d_npp.legend(loc='upper left', fontsize=9, framealpha=0.9)
            ax_3d_npp.grid(True, alpha=0.3)
            
            # Set viewing angle for better visibility
            ax_3d_npp.view_init(elev=20, azim=45)
            
            output_file_3d_npp = OUTPUT_DIR_GPP / f"parameter_space_3d_npp_{region}_{EMULATOR_TYPE}.png"
            fig_3d_npp.savefig(output_file_3d_npp, dpi=300, bbox_inches='tight')
            print(f"  Saved 3D parameter space (colored by emulated NPP) plot: {output_file_3d_npp}")
            plt.close(fig_3d_npp)
        else:
            if y_npp_lhs_pred is None:
                print(f"  Skipping 3D NPP-colored plot (NPP emulator not available)")
            elif not np.any(gpp_lai_within_bounds_mask):
                print(f"  Skipping 3D NPP-colored plot (no constrained points)")
    else:
        print(f"  Skipping 3D parameter space plot (requires slatop, vcmax25top, and stoich_nitr)")
    
    # Create histogram plots showing parameter distributions from emulated space
    print(f"  Creating parameter distribution histograms...")
    
    n_rows, n_cols = calculate_subplot_grid(len(params))
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes_hist = axes_hist.flatten() if len(params) > 1 else [axes_hist]
    fig_hist.suptitle(f'Parameter Distributions from Emulated Space - {region.replace("_", " ").title()}', 
                      fontsize=16, fontweight='bold')
    
    for idx, param in enumerate(params):
        ax_hist = axes_hist[idx]
        
        param_idx = params.index(param)
        param_values_all = lhs_samples[:, param_idx]
        
        # Determine bins for histogram (using all samples)
        bins = np.linspace(param_values_all.min(), param_values_all.max(), 30)
        
        # Plot histogram for all emulated space (light grey, in background)
        ax_hist.hist(param_values_all, bins=bins, alpha=0.3, color='lightgrey', 
                    edgecolor='grey', linewidth=0.5, label='All emulated space', zorder=1)
        
        # Plot histogram for samples within LAI bounds only (green)
        lai_only_mask = lai_within_bounds_mask & ~gpp_lai_within_bounds_mask
        if np.any(lai_only_mask):
            param_values_lai = lhs_samples[lai_only_mask, param_idx]
            ax_hist.hist(param_values_lai, bins=bins, alpha=0.5, color='green', 
                        edgecolor='darkgreen', linewidth=1.0, 
                        label=f'Within LAI bounds ({np.sum(lai_only_mask)})', zorder=2)
        
        # Plot histogram for samples within GPP bounds only (pink)
        if np.any(gpp_only_within_bounds_mask):
            param_values_gpp = lhs_samples[gpp_only_within_bounds_mask, param_idx]
            ax_hist.hist(param_values_gpp, bins=bins, alpha=0.5, color='hotpink', 
                        edgecolor='deeppink', linewidth=1.0, 
                        label=f'Within GPP bounds ({np.sum(gpp_only_within_bounds_mask)})', zorder=3)
        
        # Plot histogram for samples within BOTH LAI and GPP bounds (black)
        if np.any(gpp_lai_within_bounds_mask):
            param_values_both = lhs_samples[gpp_lai_within_bounds_mask, param_idx]
            ax_hist.hist(param_values_both, bins=bins, alpha=0.7, color='black', 
                        edgecolor='black', linewidth=1.5, 
                        label=f'Within both bounds ({np.sum(gpp_lai_within_bounds_mask)})', zorder=4)
        
        ax_hist.set_xlabel(param.replace('fates_', '').replace('_', ' '), 
                          fontsize=11, fontweight='bold')
        ax_hist.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax_hist.legend(loc='best', fontsize=8, framealpha=0.9)
        ax_hist.grid(True, alpha=0.3, axis='y')
    
    # Hide any unused subplots
    for idx in range(len(params), len(axes_hist)):
        axes_hist[idx].axis('off')
    
    plt.tight_layout()
    output_file_hist = OUTPUT_DIR_GPP / f"parameter_distributions_{region}_{EMULATOR_TYPE}.png"
    fig_hist.savefig(output_file_hist, dpi=300, bbox_inches='tight')
    print(f"  Saved histogram plot: {output_file_hist}")
    plt.close(fig_hist)

# Save parameter choices to CSV
if parameter_choices:
    param_df = pd.DataFrame(parameter_choices)
    param_output_file = OUTPUT_DIR_GPP / f"parameter_choice_{EMULATOR_TYPE}.csv"
    param_df.to_csv(param_output_file, index=False)
    print(f"\n  Saved parameter choices to: {param_output_file}")
    print(f"  Parameter choices summary:")
    print(param_df.to_string(index=False))
else:
    print(f"\n  No constrained parameter choices found to save.")

# Create plots of NPP(2000) vs parameters with emulator (colored by LAI)
print("\n" + "="*60)
print("Creating NPP(2000) vs parameter plots with emulator (colored by LAI)...")
print("="*60)

# Determine NPP column name based on ensemble
if ens_n == 1:
    npp_col = 'FATES_NPP_PPE_2000_V1'
    npp_time_label = '(2000)'
else:
    npp_col = 'FATES_NPP_PPE_V1'
    npp_time_label = ''

for region in REGIONS:
    if region not in data:
        continue
    
    df = data[region]
    
    if npp_col not in df.columns:
        print(f"\nWARNING: Missing {npp_col} column for {region}")
        continue
    
    print(f"\nCreating NPP plots for {region.replace('_', ' ').title()}")
    
    # Clean data - remove rows with NaN in parameters or NPP
    df_clean = df.dropna(subset=params + [npp_col])
    
    if len(df_clean) < 3:
        print(f"  WARNING: Not enough clean data points ({len(df_clean)}) for emulator. Skipping.")
        continue
    
    # Prepare training data for emulator
    X = df_clean[params].values
    y = df_clean[npp_col].values
    
    # Get LAI data for coloring
    lai_col = get_lai_column(region, df_clean)
    if lai_col in df_clean.columns:
        y_lai = df_clean[lai_col].values
        print(f"  Using {lai_col} for coloring")
    else:
        print(f"  WARNING: LAI column not available, using NPP values for coloring")
        y_lai = y  # Fallback to NPP if LAI not available
    
    # Train emulator for NPP
    print(f"  Training NPP emulator ({EMULATOR_TYPE.upper()}) on {len(df_clean)} ensemble members...")
    lr_npp = get_emulator()
    lr_npp.fit(X, y)
    
    # Train LAI emulator if LAI data is available
    lr_lai = None
    if lai_col in df_clean.columns:
        print(f"  Training LAI emulator ({EMULATOR_TYPE.upper()}) for coloring...")
        lr_lai = get_emulator()
        lr_lai.fit(X, y_lai)
    
    # Calculate R² and RMSE
    y_pred = lr_npp.predict(X)
    r2 = lr_npp.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    
    print(f"  NPP Emulator R²: {r2:.4f}")
    print(f"  NPP Emulator RMSE: {rmse:.4f}")
    print(f"  NPP range: {y.min():.4f} to {y.max():.4f}")
    print(f"  LHS predicted NPP range: {y_pred.min():.4f} to {y_pred.max():.4f}")
    
    # Generate Latin Hypercube samples for emulator predictions
    n_samples = 2000 if len(params) >= 4 else 1000
    lhs_samples = {}
    for param in params:
        param_min = df_clean[param].min()
        param_max = df_clean[param].max()
        lhs_samples[param] = np.random.uniform(param_min, param_max, n_samples)
    
    X_lhs = np.column_stack([lhs_samples[param] for param in params])
    y_lhs_pred = lr_npp.predict(X_lhs)
    
    # Predict LAI for LHS samples (for coloring)
    if lr_lai is not None:
        y_lai_lhs_pred = lr_lai.predict(X_lhs)
    else:
        y_lai_lhs_pred = y_lhs_pred  # Fallback
    
    # Create plots
    n_rows, n_cols = calculate_subplot_grid(len(params))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes = axes.flatten() if len(params) > 1 else [axes]
    fig.suptitle(f'NPP{npp_time_label} vs Parameters with Emulator (Colored by LAI) - {region.replace("_", " ").title()} ({YEAR_RANGE})',
                fontsize=16, fontweight='bold')
    
    scatter_emul = None
    for idx, param in enumerate(params):
        ax = axes[idx]
        
        # Plot emulator LHS predictions (colored by LAI)
        scatter_emul = ax.scatter(lhs_samples[param], y_lhs_pred, c=y_lai_lhs_pred,
                                 s=10, alpha=0.3, cmap='viridis', vmin=y_lai.min(), vmax=y_lai.max(),
                                 label='Emulator predictions', zorder=1)
        
        # Plot ensemble members (colored by LAI)
        scatter_ens = ax.scatter(df_clean[param], y, c=y_lai,
                                s=100, alpha=0.8, cmap='viridis', edgecolors='black', linewidth=1,
                                vmin=y_lai.min(), vmax=y_lai.max(),
                                label='Ensemble members', zorder=2)
        
        # Add ensemble member labels
        for _, row_data in df_clean.iterrows():
            ax.annotate(f"n{int(row_data['ensemble_member']):02d}", 
                       (row_data[param], row_data[npp_col]),
                       fontsize=7, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='none', alpha=0.7))
        
        ax.set_xlabel(param.replace('fates_', '').replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_ylabel(f'NPP{npp_time_label} [gC m⁻² day⁻¹]', fontsize=11, fontweight='bold')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
                   markeredgecolor='black', label='Ensemble members', alpha=0.8),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, 
                   label='Emulator predictions', alpha=0.3)
        ]
        legend_labels = ['Ensemble members', 'Emulator predictions']
        
        ax.legend(legend_handles, legend_labels, loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar for LAI values
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if scatter_emul is not None:
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(scatter_emul, cax=cbar_ax)
        cbar.set_label('LAI [m²/m²]', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR_NPP / f"npp_vs_parameters_{region}_{EMULATOR_TYPE}.png"
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
    
    for param in params:
        if param in df.columns and 'FATES_NPP_PPE_V1' in df.columns:
            corr = df[param].corr(df['FATES_NPP_PPE_V1'])
            print(f"  {param:<45s}: r = {corr:7.4f}")
    


# Create LAI emulator and parameter space plots
print("\n" + "="*60)
print("Creating LAI(2000) Emulator and Parameter Space Plots...")
print("="*60)

print(f"\nDEBUG: data dictionary keys: {list(data.keys())}")
print(f"DEBUG: REGIONS list: {REGIONS}")

for region in REGIONS:
    print(f"\n  Checking region: {region}")
    if region not in data:
        print(f"    Region {region} not in data dictionary! Skipping.")
        continue
    
    df = data[region]
    
    # Get appropriate LAI column for this region
    lai_col = get_lai_column(region, df)
    lai_label = get_lai_label(region)
    
    if lai_col not in df.columns:
        print(f"    WARNING: Missing LAI column ({lai_col}) for {region}")
        print(f"    Available columns: {list(df.columns)}")
        continue
    
    print(f"\n{'='*50}")
    print(f"Region: {region.replace('_', ' ').title()}")
    if region == 'ENTs':
        print(f"Using average annual maximum LAI")
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
    
    lr_lai = get_emulator()
    lr_lai.fit(X, y_lai)
    print(f"  Using {EMULATOR_TYPE.upper()} emulator")
    
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
    if region == 'ENTs':
        if region in obs_lai_regional_mean_ent and not np.isnan(obs_lai_regional_mean_ent[region]):
            print(f"  MODIS Mean LAI (ENTs): {obs_lai_regional_mean_ent[region]:.4f}")
        if region in obs_lai_regional_annual_max_ent and not np.isnan(obs_lai_regional_annual_max_ent[region]):
            print(f"  MODIS Annual Max LAI (ENTs): {obs_lai_regional_annual_max_ent[region]:.4f}")
    
    # Create emulator response plots: LAI vs each parameter with emulator curve
    print(f"\n  Creating LAI emulator response curves...")
    
    # Debug: Show which observations will be plotted
    print(f"  LAI observations for {region}:")
    print(f"    obs_lai (MODIS): {obs_lai if not np.isnan(obs_lai) else 'NaN - will NOT be plotted'}")
    print(f"    obs_lai_avhrr (AVHRR): {obs_lai_avhrr if not np.isnan(obs_lai_avhrr) else 'NaN - will NOT be plotted'}")
    if region == 'ENTs':
        obs_lai_mean = obs_lai_regional_mean_ent.get(region, np.nan)
        obs_lai_max = obs_lai_regional_annual_max_ent.get(region, np.nan)
        print(f"    obs_lai_regional_mean_ent: {obs_lai_mean if not np.isnan(obs_lai_mean) else 'NaN - will NOT be plotted'}")
        print(f"    obs_lai_regional_annual_max_ent: {obs_lai_max if not np.isnan(obs_lai_max) else 'NaN - will NOT be plotted'}")
    
    n_rows, n_cols = calculate_subplot_grid(len(params))
    fig_emulator, axes_emulator = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes_emulator = axes_emulator.flatten() if len(params) > 1 else [axes_emulator]
    fig_emulator.suptitle(f'{lai_label}(2000) Emulator Response ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                         fontsize=16, fontweight='bold')
    
    # Calculate mean values for all parameters (for holding constant)
    param_means = {param: df_clean[param].mean() for param in params}
    
    for idx, param in enumerate(params):
        ax = axes_emulator[idx]
        
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
        
        # Debug output for first subplot only
        if idx == 0:
            print(f"  Adding horizontal lines to emulator response plots:")
        
        # Add grey shading between observational LAI estimates
        if region == 'ENTs':
            # For ENTs, use annual max values
            lai_modis = obs_lai if not np.isnan(obs_lai) else obs_lai_regional_annual_max_ent.get(region, np.nan)
            lai_avhrr = obs_lai_regional_annual_max_avhrr_ent.get(region, obs_lai_avhrr) if not np.isnan(obs_lai_avhrr) else np.nan
            if not np.isnan(lai_modis) and not np.isnan(lai_avhrr):
                lai_min = min(lai_modis, lai_avhrr)
                lai_max = max(lai_modis, lai_avhrr)
                ax.axhspan(lai_min, lai_max, color='grey', alpha=0.2, zorder=0.5)
        else:
            # For other regions, use regular values
            if not np.isnan(obs_lai) and not np.isnan(obs_lai_avhrr):
                lai_min = min(obs_lai, obs_lai_avhrr)
                lai_max = max(obs_lai, obs_lai_avhrr)
                ax.axhspan(lai_min, lai_max, color='grey', alpha=0.2, zorder=0.5)
        
        # Add observational LAI as horizontal line
        if not np.isnan(obs_lai):
            if idx == 0:
                print(f"    ✓ Adding MODIS {'Summer ' if region == 'ENTs' else ''}{lai_label} line at y={obs_lai:.4f}")
            if region == 'ENTs':
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS Annual Max LAI: {obs_lai_regional_annual_max_ent.get(region, obs_lai):.2f}', zorder=1)
            else:
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS {lai_label}: {obs_lai:.2f}', zorder=1)
        elif idx == 0:
            print(f"    ✗ Skipping MODIS line (value is NaN)")
            
        if not np.isnan(obs_lai_avhrr):
            if idx == 0:
                print(f"    ✓ Adding AVHRR {'Annual Max ' if region == 'ENTs' else ''}{lai_label} line at y={obs_lai_avhrr:.4f}")
            # For ENTs, use annual max; for others use regular value
            avhrr_val = obs_lai_regional_annual_max_avhrr_ent.get(region, obs_lai_avhrr) if region == 'ENTs' else obs_lai_avhrr
            avhrr_label_suffix = ' Annual Max' if region == 'ENTs' else ''
            ax.axhline(avhrr_val, color='limegreen', linestyle=':', linewidth=2,
                      alpha=0.8, label=f'AVHRR{avhrr_label_suffix} {lai_label}: {avhrr_val:.2f}', zorder=1)
        elif idx == 0:
            print(f"    ✗ Skipping AVHRR line (value is NaN)")
        
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
        
        # Add text showing parameter coefficient (only for linear regression)
        if EMULATOR_TYPE == 'linear':
            coef = lr_lai.coef_[param_idx]
            ax.text(0.05, 0.95, f'Coefficient: {coef:.4f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_file_emulator = OUTPUT_DIR_LAI_EMULATOR / f"lai_emulator_response_{YEAR_RANGE}_{region}_{EMULATOR_TYPE}.png"
    fig_emulator.savefig(output_file_emulator, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_emulator.name}")
    plt.close(fig_emulator)
    
    # Generate Latin Hypercube samples
    print(f"\n  Generating Latin Hypercube samples...")
    n_samples = 3000
    
    param_bounds = np.array([[df_clean[param].min(), df_clean[param].max()] for param in params])
    
    sampler = qmc.LatinHypercube(d=len(params), seed=42)
    lhs_samples_unit = sampler.random(n=n_samples)
    lhs_samples = qmc.scale(lhs_samples_unit, param_bounds[:, 0], param_bounds[:, 1])
    
    # Predict LAI for LHS samples
    y_lai_lhs_pred = lr_lai.predict(lhs_samples)
    
    print(f"  LHS predicted LAI range: {y_lai_lhs_pred.min():.4f} to {y_lai_lhs_pred.max():.4f}")
    
    # Create 2D parameter space plot: d2bl1 vs l2fr colored by LAI (only if these parameters exist)
    if 'fates_allom_d2bl1' in params and 'fates_allom_l2fr' in params:
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
    else:
        print(f"\n  Skipping parameter space visualization (requires fates_allom_d2bl1 and fates_allom_l2fr)")
    
    # Create LAI vs parameters plot with emulator samples
    print(f"\n  Creating LAI vs parameters with emulator samples...")
    
    # Generate 2000 LHS samples for emulator
    n_emulator_samples = 2000
    sampler_emul = qmc.LatinHypercube(d=len(params), seed=43)
    lhs_emul_unit = sampler_emul.random(n=n_emulator_samples)
    lhs_emul = qmc.scale(lhs_emul_unit, param_bounds[:, 0], param_bounds[:, 1])
    
    # Predict LAI for these samples
    lai_emul_pred = lr_lai.predict(lhs_emul)
    
    # Also predict GPP for these samples to identify GPP bounds
    # Determine GPP column name based on ensemble
    gpp_col_for_emul = 'FATES_GPP_PPE_2000_V1' if ens_n == 1 else 'FATES_GPP_PPE_V1'
    gpp_emul_pred = None
    gpp_within_bounds_lai_emul = np.zeros(n_emulator_samples, dtype=bool)
    
    if gpp_col_for_emul in df_clean.columns:
        y_gpp = df_clean[gpp_col_for_emul].values
        lr_gpp_lai = get_emulator()
        lr_gpp_lai.fit(X, y_gpp)
        gpp_emul_pred = lr_gpp_lai.predict(lhs_emul)
        
        # Identify samples within GPP bounds
        obs_gpp_fluxcom_lai = obs_gpp_regional.get(region, np.nan)
        obs_gpp_wecann_lai = obs_gpp_regional_wecann.get(region, np.nan)
        
        if not np.isnan(obs_gpp_fluxcom_lai) and not np.isnan(obs_gpp_wecann_lai):
            gpp_min_lai = min(obs_gpp_fluxcom_lai, obs_gpp_wecann_lai)
            gpp_max_lai = max(obs_gpp_fluxcom_lai, obs_gpp_wecann_lai)
            gpp_within_bounds_lai_emul = (gpp_emul_pred >= gpp_min_lai) & (gpp_emul_pred <= gpp_max_lai)
            print(f"  Emulator samples within GPP bounds (for LAI plots): {np.sum(gpp_within_bounds_lai_emul)}/{n_emulator_samples}")
    
    # Get slatop values for coloring
    slatop_idx = params.index('fates_leaf_slatop')
    slatop_values = lhs_emul[:, slatop_idx]
    
    n_rows, n_cols = calculate_subplot_grid(len(params))
    fig_lai_params, axes_lai = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes_lai = axes_lai.flatten() if len(params) > 1 else [axes_lai]
    fig_lai_params.suptitle(f'{lai_label}(2000) vs Parameters with Emulator ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                            fontsize=16, fontweight='bold')
    
    scatter_emul = None  # Initialize outside loop
    for idx, param in enumerate(params):
        ax = axes_lai[idx]
        
        param_idx = params.index(param)
        
        # Plot emulator samples NOT within GPP bounds colored by slatop
        if np.any(~gpp_within_bounds_lai_emul):
            scatter_emul = ax.scatter(lhs_emul[~gpp_within_bounds_lai_emul, param_idx], 
                      lai_emul_pred[~gpp_within_bounds_lai_emul], s=15, alpha=0.6, 
                      c=slatop_values[~gpp_within_bounds_lai_emul], cmap='viridis', edgecolors='none', 
                      label=f'Emulator samples', zorder=1)
        
        # Plot emulator samples WITHIN GPP bounds in green
        if np.any(gpp_within_bounds_lai_emul):
            ax.scatter(lhs_emul[gpp_within_bounds_lai_emul, param_idx], 
                      lai_emul_pred[gpp_within_bounds_lai_emul], s=20, alpha=0.7, 
                      c='limegreen', edgecolors='none',
                      label=f'Within GPP bounds ({np.sum(gpp_within_bounds_lai_emul)} samples)', zorder=2)
        
        # Plot actual ensemble members on top
        ax.scatter(df_clean[param], y_lai, s=100, alpha=0.8, c='red',
                  edgecolors='black', linewidth=1.5, label='Ensemble members', zorder=3)
        
        # Add observational LAI as horizontal line
        if not np.isnan(obs_lai):
            if region == 'ENTs':
                # For ENTs, obs_lai now contains the annual max
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS Annual Max {lai_label}: {obs_lai:.2f}', zorder=2)
            else:
                ax.axhline(obs_lai, color='darkgreen', linestyle='--', linewidth=2,
                          alpha=0.8, label=f'MODIS {lai_label}: {obs_lai:.2f}', zorder=2)
        
        # Add ENT-specific annual max LAI observations
        if region == 'ENTs':
            if region in obs_lai_regional_annual_max_avhrr_ent and not np.isnan(obs_lai_regional_annual_max_avhrr_ent[region]):
                ax.axhline(obs_lai_regional_annual_max_avhrr_ent[region], color='darkorange', linestyle=':', linewidth=2.5,
                          alpha=0.8, label=f'AVHRR Annual Max {lai_label}: {obs_lai_regional_annual_max_avhrr_ent[region]:.2f}', zorder=2)
        
        # Add AVHRR for non-ENT regions
        if region != 'ENTs' and not np.isnan(obs_lai_avhrr):
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
    
    output_file_lai_params = OUTPUT_DIR_LAI_EMULATOR / f"lai_vs_params_with_emulator_{YEAR_RANGE}_{region}_{EMULATOR_TYPE}.png"
    fig_lai_params.savefig(output_file_lai_params, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_lai_params.name}")
    plt.close(fig_lai_params)
    
    # Create second plot colored by VEGC emulator predictions
    print(f"\n  Creating LAI vs parameters colored by VEGC emulator...")
    
    # Determine VEGC column name based on ensemble
    vegc_col = 'FATES_VEGC_PPE_2000_V1' if ens_n == 1 else 'FATES_VEGC_PPE_V1'
    
    # Check if VEGC data is available
    if vegc_col not in df_clean.columns:
        print(f"    WARNING: VEGC column {vegc_col} not available for {region}, skipping VEGC-colored plot")
    else:
        # Train VEGC emulator
        y_vegc = df_clean[vegc_col].values
        lr_vegc = get_emulator()
        print(f"  Using {EMULATOR_TYPE.upper()} emulator")
        lr_vegc.fit(X, y_vegc)
        
        # Predict VEGC for the 1000 LHS samples
        vegc_emul_pred = lr_vegc.predict(lhs_emul)
        
        print(f"    VEGC emulator R²: {lr_vegc.score(X, y_vegc):.4f}")
        print(f"    Predicted VEGC range: {vegc_emul_pred.min():.4f} to {vegc_emul_pred.max():.4f}")
        
        n_rows, n_cols = calculate_subplot_grid(len(params))
        fig_lai_vegc, axes_lai_vegc = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        axes_lai_vegc = axes_lai_vegc.flatten() if len(params) > 1 else [axes_lai_vegc]
        fig_lai_vegc.suptitle(f'{lai_label}(2000) vs Parameters (colored by VEGC emulator) ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                              fontsize=16, fontweight='bold')
        
        scatter_vegc = None  # Initialize outside loop
        for idx, param in enumerate(params):
            ax = axes_lai_vegc[idx]
            
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
        
        output_file_lai_vegc = OUTPUT_DIR_LAI_EMULATOR / f"lai_vs_params_colored_by_vegc_{YEAR_RANGE}_{region}_{EMULATOR_TYPE}.png"
        fig_lai_vegc.savefig(output_file_lai_vegc, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file_lai_vegc.name}")
        plt.close(fig_lai_vegc)

# Create Carbon Use Efficiency (CUE) plots (NPP/GPP)
print("\n" + "="*60)
print("Creating Carbon Use Efficiency (CUE = NPP/GPP) Analysis...")
print("="*60)

# Read observed CUE min/max values
CUE_OBS_FILE = CSV_DIR / "cue.csv"
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

for region in REGIONS:
    print(f"\n  Checking region: {region}")
    if region not in data:
        print(f"    Region {region} not in data dictionary! Skipping.")
        continue
    
    df = data[region]
    
    print(f"\n{'='*50}")
    print(f"Region: {region.replace('_', ' ').title()}")
    print(f"{'='*50}")
    
    # Calculate CUE for ensembles (both V1 and 2000 for ens_n=1, only V1 for ens_n=2)
    if ens_n == 1:
        ensemble_configs = [
            ('V1', 'FATES_NPP_PPE_V1', 'FATES_GPP_PPE_V1'),
            ('2000', 'FATES_NPP_PPE_2000_V1', 'FATES_GPP_PPE_2000_V1')
        ]
    else:
        ensemble_configs = [
            ('V1', 'FATES_NPP_PPE_V1', 'FATES_GPP_PPE_V1')
        ]
    
    for ensemble_name, npp_col, gpp_col in ensemble_configs:
        
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
        
        lr_cue = get_emulator()
        print(f"  Using {EMULATOR_TYPE.upper()} emulator")
        lr_cue.fit(X, y_cue)
        
        y_cue_pred_train = lr_cue.predict(X)
        train_score = lr_cue.score(X, y_cue)
        rmse = np.sqrt(np.mean((y_cue - y_cue_pred_train)**2))
        
        print(f"    Emulator R²: {train_score:.4f}")
        print(f"    Emulator RMSE: {rmse:.4f}")
        print(f"    CUE range: {y_cue.min():.4f} to {y_cue.max():.4f}")
        
        # Generate LHS samples for emulator
        n_samples = 3000
        param_bounds = np.array([[df_clean[param].min(), df_clean[param].max()] for param in params])
        
        sampler = qmc.LatinHypercube(d=len(params), seed=45)
        lhs_unit = sampler.random(n=n_samples)
        lhs_samples = qmc.scale(lhs_unit, param_bounds[:, 0], param_bounds[:, 1])
        
        # Predict CUE for LHS samples
        y_cue_lhs = lr_cue.predict(lhs_samples)
        
        print(f"    LHS predicted CUE range: {y_cue_lhs.min():.4f} to {y_cue_lhs.max():.4f}")
        
        # Create dynamic subplot grid based on number of parameters: CUE vs each parameter with emulator
        n_rows, n_cols = calculate_subplot_grid(len(params))
        fig_cue, axes_cue = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        n_rows, n_cols = calculate_subplot_grid(len(params))
        axes_cue = axes_cue.flatten() if len(params) > 1 else [axes_cue]
        axes_cue = axes_cue.flatten() if len(params) > 1 else [axes_cue]
        fig_cue.suptitle(f'Carbon Use Efficiency (NPP/GPP) - {ensemble_name} ({YEAR_RANGE}) - {region.replace("_", " ").title()}',
                        fontsize=16, fontweight='bold')
        
        for idx, param in enumerate(params):
            ax = axes_cue[idx]
            
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
        
        output_file_cue = OUTPUT_DIR_CUE / f"cue_{ensemble_name.lower()}_{YEAR_RANGE}_{region}_{EMULATOR_TYPE}.png"
        fig_cue.savefig(output_file_cue, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file_cue.name}")
        plt.close(fig_cue)

print("\n" + "="*60)
print("Analysis complete!")
print(f"Output files saved to: {OUTPUT_DIR}")
