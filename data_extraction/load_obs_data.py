#!/usr/bin/env python
"""
Script to read observational GPP and LAI data from pre-extracted CSV files

Run data_extraction/extract_obs_data.py first to generate the CSV files.
This file provides a simple function to load the observational data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_observational_data(data_extraction_dir=None):
    """
    Load observational GPP and LAI data from CSV file
    
    Args:
        data_extraction_dir: Path to data_extraction directory. If None, uses default location.
    
    Returns:
        Dictionary containing:
            - obs_lai_regional: MODIS LAI data (dict)
            - obs_lai_regional_avhrr: AVHRR LAI data (dict)
            - obs_gpp_regional: FLUXCOM GPP data (dict)
            - obs_gpp_regional_wecann: WECANN GPP data (dict)
            - obs_lai_regional_mean_ent: MODIS mean LAI for ENTs (dict)
            - obs_lai_regional_annual_max_ent: MODIS annual max LAI for ENTs (dict)
    """
    if data_extraction_dir is None:
        data_extraction_dir = Path(__file__).parent / "data_extraction"
    else:
        data_extraction_dir = Path(data_extraction_dir)
    
    obs_file = data_extraction_dir / "obs_gpp_lai_data.csv"
    
    # Initialize dictionaries
    obs_lai_regional = {}
    obs_lai_regional_avhrr = {}
    obs_gpp_regional = {}
    obs_gpp_regional_wecann = {}
    obs_lai_regional_mean_ent = {}
    obs_lai_regional_annual_max_ent = {}
    
    REGIONS = ['north_65', 'BDTs', 'BETs', 'ENTs', 'AC3G', 'C3G', 'C4']
    
    if not obs_file.exists():
        print(f"ERROR: Observational data file not found: {obs_file}")
        print(f"Please run data_extraction/extract_obs_data.py first to generate this file.")
        # Initialize with NaN values
        for region in REGIONS:
            obs_lai_regional[region] = np.nan
            obs_lai_regional_avhrr[region] = np.nan
            obs_gpp_regional[region] = np.nan
            obs_gpp_regional_wecann[region] = np.nan
    else:
        try:
            df_obs = pd.read_csv(obs_file)
            print(f"Loaded observational data for {len(df_obs)} regions from: {obs_file}")
            
            # Extract data into dictionaries
            for _, row in df_obs.iterrows():
                region = row['region']
                
                # LAI data
                if 'lai_modis' in df_obs.columns:
                    obs_lai_regional[region] = row.get('lai_modis', np.nan)
                if 'lai_avhrr' in df_obs.columns:
                    obs_lai_regional_avhrr[region] = row.get('lai_avhrr', np.nan)
                if 'lai_modis_summer' in df_obs.columns:
                    if region == 'ENTs' and not pd.isna(row.get('lai_modis_summer')):
                        obs_lai_regional[region] = row.get('lai_modis_summer')
                if 'lai_modis_mean' in df_obs.columns and region == 'ENTs':
                    obs_lai_regional_mean_ent[region] = row.get('lai_modis_mean', np.nan)
                if 'lai_modis_annual_max' in df_obs.columns and region == 'ENTs':
                    obs_lai_regional_annual_max_ent[region] = row.get('lai_modis_annual_max', np.nan)
                
                # GPP data
                if 'gpp_fluxcom' in df_obs.columns:
                    obs_gpp_regional[region] = row.get('gpp_fluxcom', np.nan)
                if 'gpp_wecann' in df_obs.columns:
                    obs_gpp_regional_wecann[region] = row.get('gpp_wecann', np.nan)
            
        except Exception as e:
            print(f"ERROR reading observational data: {str(e)}")
            import traceback
            traceback.print_exc()
            # Initialize with NaN values
            for region in REGIONS:
                obs_lai_regional[region] = np.nan
                obs_lai_regional_avhrr[region] = np.nan
                obs_gpp_regional[region] = np.nan
                obs_gpp_regional_wecann[region] = np.nan
    
    return {
        'obs_lai_regional': obs_lai_regional,
        'obs_lai_regional_avhrr': obs_lai_regional_avhrr,
        'obs_gpp_regional': obs_gpp_regional,
        'obs_gpp_regional_wecann': obs_gpp_regional_wecann,
        'obs_lai_regional_mean_ent': obs_lai_regional_mean_ent,
        'obs_lai_regional_annual_max_ent': obs_lai_regional_annual_max_ent,
    }
