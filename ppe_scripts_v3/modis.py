import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import xesmf as xe

# Read the MODIS LAI data
data_path = '/datalake/NS9560K/diagnostics/ILAMB-Data/DATA/lai/MODIS/lai_0.5x0.5.nc'
ds = xr.open_dataset(data_path, decode_times=False)

print("Dataset structure:")
print(ds)
print("\nDimensions:", ds.dims)
print("Coordinates:", list(ds.coords))

# Compute temporal mean over the time dimension
lai_mean = ds['lai'].mean(dim='time')

# Create summer mask (will be used later)
n_months = len(ds['time'])
n_years = n_months // 12
months = np.tile(np.arange(1, 13), n_years)
summer_mask = np.isin(months, [6, 7])

# Create output directory if it doesn't exist
output_dir = '/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/modis_lai'
os.makedirs(output_dir, exist_ok=True)

# Load ENT grid cell coordinates early so we can add them to all maps
ent_file = '/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/ent_grid_cells.txt'
ent_lats = None
ent_lons = None
if os.path.exists(ent_file):
    print(f"\nLoading ENT grid cells from: {ent_file}")
    ent_data = np.loadtxt(ent_file, skiprows=1)
    ent_lats = ent_data[:, 0]
    ent_lons = ent_data[:, 1]
    ent_lons = ent_lons % 360
    print(f"Loaded {len(ent_lats)} ENT grid points")

# Create figure
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Plot the temporal mean LAI
im = lai_mean.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='jet',
    cbar_kwargs={'label': 'Leaf Area Index (m²/m²)', 'shrink': 0.8},
    vmin=0,
    vmax=7
)

# Add ENT points (will be added after extraction below)
ent_lats_for_map = None
ent_lons_for_map = None
ent_lai_for_map = None

ax.coastlines()
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
ax.set_title('MODIS LAI Temporal Mean', fontsize=14, fontweight='bold')

# Save figure
output_path = os.path.join(output_dir, 'modis_lai_temporal_mean.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Save the temporal mean data as NetCDF
output_nc_mean = os.path.join(output_dir, 'modis_lai_temporal_mean.nc')
try:
    lai_mean.to_netcdf(output_nc_mean, mode='w')
    print(f"Temporal mean data saved to: {output_nc_mean}")
except PermissionError:
    print(f"Warning: Could not save {output_nc_mean} (permission denied)")

# Also print some statistics
print(f"\nTemporal mean statistics:")
print(f"Min: {lai_mean.min().values:.3f}")
print(f"Max: {lai_mean.max().values:.3f}")
print(f"Mean: {lai_mean.mean().values:.3f}")

plt.close()

# Select location at 60N, 60E and plot timeseries
print("\n" + "="*60)
print("Extracting timeseries at 60N, 60E")
print("="*60)

# Select the nearest point to 60N, 60E
lai_point = ds['lai'].sel(lat=60, lon=60, method='nearest')

print(f"\nActual location selected:")
print(f"Lat: {lai_point.lat.values}")
print(f"Lon: {lai_point.lon.values}")

# Create timeseries plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the timeseries
ax.plot(range(len(lai_point)), lai_point.values, marker='o', linestyle='-', linewidth=2, markersize=4, label='LAI timeseries')

# Calculate mean and summer mean for this location
overall_mean = lai_point.mean().values
lai_point_summer = lai_point.isel(time=summer_mask)
summer_mean = lai_point_summer.mean().values

# Add horizontal lines for overall mean and summer mean
ax.axhline(y=overall_mean, color='blue', linestyle='--', linewidth=2, label=f'Annual mean: {overall_mean:.2f}')
ax.axhline(y=summer_mean, color='red', linestyle='--', linewidth=2, label=f'Summer mean: {summer_mean:.2f}')

ax.set_xlabel('Time Index', fontsize=12)
ax.set_ylabel('Leaf Area Index (m²/m²)', fontsize=12)
ax.set_title(f'MODIS LAI Timeseries at {lai_point.lat.values:.1f}°N, {lai_point.lon.values:.1f}°E', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

# Save timeseries plot
output_path_ts = os.path.join(output_dir, 'modis_lai_timeseries_60N_60E.png')
plt.savefig(output_path_ts, dpi=300, bbox_inches='tight')
print(f"\nTimeseries plot saved to: {output_path_ts}")

# Print statistics for this location
print(f"\nTimeseries statistics at this location:")
print(f"Min: {lai_point.min().values:.3f}")
print(f"Max: {lai_point.max().values:.3f}")
print(f"Mean: {lai_point.mean().values:.3f}")
print(f"Std: {lai_point.std().values:.3f}")

plt.close()

# Calculate summer (June-July) mean LAI
print("\n" + "="*60)
print("Calculating summer (June-July) mean LAI")
print("="*60)

# The time dimension is 72 months (6 years * 12 months)
# Extract month information assuming the data cycles through months 1-12
n_months = len(ds['time'])
n_years = n_months // 12

# Create month indices (repeating 1-12 for each year)
months = np.tile(np.arange(1, 13), n_years)

# Select only June (6), July (7), and August (8)
summer_mask = np.isin(months, [6, 7, 8])
print(f"Selected {summer_mask.sum()} summer months out of {n_months} total months")

# Calculate summer mean
lai_summer_mean = ds['lai'].isel(time=summer_mask).mean(dim='time')

# Create summer mean map
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

im = lai_summer_mean.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='jet',
    cbar_kwargs={'label': 'Leaf Area Index (m²/m²)', 'shrink': 0.8},
    vmin=0,
    vmax=7
)

ax.coastlines()
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
ax.set_title('MODIS LAI Summer Mean (June-July)', fontsize=14, fontweight='bold')

# Save figure
output_path_summer = os.path.join(output_dir, 'modis_lai_summer_mean.png')
plt.savefig(output_path_summer, dpi=300, bbox_inches='tight')
print(f"\nSummer mean plot saved to: {output_path_summer}")

# Save the summer mean data as NetCDF
output_nc_path = os.path.join(output_dir, 'modis_lai_summer_mean.nc')
try:
    lai_summer_mean.to_netcdf(output_nc_path, mode='w')
    print(f"Summer mean data saved to: {output_nc_path}")
except PermissionError:
    print(f"Warning: Could not save {output_nc_path} (permission denied)")

# Print statistics
print(f"\nSummer (June-July) mean statistics:")
print(f"Min: {lai_summer_mean.min().values:.3f}")
print(f"Max: {lai_summer_mean.max().values:.3f}")
print(f"Mean: {lai_summer_mean.mean().values:.3f}")

plt.close()

# Calculate mean annual maximum LAI
print("\n" + "="*60)
print("Calculating mean annual maximum LAI")
print("="*60)

# Reshape data to separate years (assuming 12 months per year)
n_months = len(ds['time'])
n_years = n_months // 12
print(f"Data spans {n_years} years with {n_months} total months")

# Reshape LAI to (n_years, 12, lat, lon)
lai_data = ds['lai'].values
lai_reshaped = lai_data[:n_years*12].reshape(n_years, 12, len(ds['lat']), len(ds['lon']))

# Calculate maximum for each year, then mean across years
annual_max = np.max(lai_reshaped, axis=1)  # Max over months for each year
mean_annual_max = np.mean(annual_max, axis=0)  # Mean over years

# Create xarray DataArray for plotting
lai_mean_annual_max = xr.DataArray(
    mean_annual_max,
    coords={'lat': ds['lat'], 'lon': ds['lon']},
    dims=['lat', 'lon']
)

# Create map
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

im = lai_mean_annual_max.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='jet',
    cbar_kwargs={'label': 'Leaf Area Index (m²/m²)', 'shrink': 0.8},
    vmin=0,
    vmax=7
)

ax.coastlines()
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
ax.set_title('MODIS LAI Mean Annual Maximum', fontsize=14, fontweight='bold')

# Save figure
output_path_max = os.path.join(output_dir, 'modis_lai_mean_annual_maximum.png')
plt.savefig(output_path_max, dpi=300, bbox_inches='tight')
print(f"\nMean annual maximum plot saved to: {output_path_max}")

# Save as NetCDF
output_nc_max = os.path.join(output_dir, 'modis_lai_mean_annual_maximum.nc')
try:
    lai_mean_annual_max.to_netcdf(output_nc_max, mode='w')
    print(f"Mean annual maximum data saved to: {output_nc_max}")
except PermissionError:
    print(f"Warning: Could not save {output_nc_max} (permission denied)")

# Print statistics
print(f"\nMean annual maximum statistics:")
print(f"Min: {np.nanmin(mean_annual_max):.3f}")
print(f"Max: {np.nanmax(mean_annual_max):.3f}")
print(f"Mean: {np.nanmean(mean_annual_max):.3f}")

plt.close()

# Calculate and plot difference between summer mean and annual mean
print("\n" + "="*60)
print("Calculating difference (Summer - Annual mean)")
print("="*60)

lai_diff = lai_summer_mean - lai_mean

# Create difference map
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

im = lai_diff.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='jet',
    cbar_kwargs={'label': 'LAI Difference (m²/m²)', 'shrink': 0.8},
    vmin=0,
    vmax=3
)

ax.coastlines()
ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
ax.set_title('MODIS LAI: Summer Mean - Annual Mean', fontsize=14, fontweight='bold')

# Save figure
output_path_diff = os.path.join(output_dir, 'modis_lai_summer_minus_annual.png')
plt.savefig(output_path_diff, dpi=300, bbox_inches='tight')
print(f"\nDifference plot saved to: {output_path_diff}")

# Print statistics
print(f"\nDifference (Summer - Annual) statistics:")
print(f"Min: {lai_diff.min().values:.3f}")
print(f"Max: {lai_diff.max().values:.3f}")
print(f"Mean: {lai_diff.mean().values:.3f}")

plt.close()

# Plot LAI timeseries for all ENT grid points
print("\n" + "="*60)
print("Plotting LAI timeseries for all ENT grid points")
print("="*60)

# Load ENT grid cell coordinates
ent_file = '/datalake/NS9560K/www/diagnostics/noresm/rosief/ppe_diags/ent_grid_cells.txt'
if os.path.exists(ent_file):
    print(f"\nLoading ENT grid cells from: {ent_file}")
    ent_data = np.loadtxt(ent_file, skiprows=1)
    ent_lats = ent_data[:, 0]
    ent_lons = ent_data[:, 1]
    
    # Convert longitude if needed (0-360 to 0-360, -180-180 to 0-360)
    ent_lons = ent_lons % 360
    
    print(f"Loaded {len(ent_lats)} ENT grid points")
    print(f"Latitude range: {ent_lats.min():.2f} to {ent_lats.max():.2f}")
    print(f"Longitude range: {ent_lons.min():.2f} to {ent_lons.max():.2f}")
    
    # Extract timeseries for each ENT point
    print("\nExtracting LAI timeseries for each ENT point...")
    ent_timeseries = []
    actual_lats = []
    actual_lons = []
    
    for i, (lat, lon) in enumerate(zip(ent_lats, ent_lons)):
        lai_point = ds['lai'].sel(lat=lat, lon=lon, method='nearest')
        
        # Check if the temporal mean has ANY valid data (not requiring full timeseries)
        lai_mean_value = np.nanmean(lai_point.values)
        
        if np.isnan(lai_mean_value):
            # Find nearest point with valid data within 1 degree
            print(f"  Point {i+1} at ({lat:.2f}, {lon:.2f}) has no valid data, searching for nearby valid point...")
            
            # Create lat/lon bounds within 1 degree
            lat_min, lat_max = lat - 1, lat + 1
            lon_min, lon_max = lon - 1, lon + 1
            
            # Select region within 1 degree
            lai_region = ds['lai'].sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max)
            )
            
            # Find the nearest point with valid data (check temporal mean)
            lai_region_mean = np.nanmean(lai_region.values, axis=0)
            valid_mask = ~np.isnan(lai_region_mean)
            
            if np.any(valid_mask):
                # Calculate distances to all valid points
                valid_lats, valid_lons = np.meshgrid(lai_region.lat.values, lai_region.lon.values, indexing='ij')
                distances = np.sqrt((valid_lats - lat)**2 + (valid_lons - lon)**2)
                distances[~valid_mask] = np.inf
                
                min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                best_lat = lai_region.lat.values[min_idx[0]]
                best_lon = lai_region.lon.values[min_idx[1]]
                
                print(f"    Found valid point at ({best_lat:.2f}, {best_lon:.2f}), distance: {distances[min_idx]:.2f} degrees")
                lai_point = ds['lai'].sel(lat=best_lat, lon=best_lon, method='nearest')
            else:
                print(f"    WARNING: No valid points found within 1 degree!")
        
        ent_timeseries.append(lai_point.values)
        actual_lats.append(float(lai_point.lat.values))
        actual_lons.append(float(lai_point.lon.values))
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(ent_lats)} points")
    
    ent_timeseries = np.array(ent_timeseries)
    print(f"Extracted timeseries shape: {ent_timeseries.shape}")
    print(f"Number of ENT points with valid data: {len(actual_lats)}")
    
    # Now add ENT points to all the maps created earlier
    print("\nAdding ENT points to maps...")
    
    # Re-create temporal mean map with ENT points
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    lai_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet',
                 cbar_kwargs={'label': 'Leaf Area Index (m²/m²)', 'shrink': 0.8},
                 vmin=0, vmax=7)
    ent_lai = [float(lai_mean.sel(lat=lat, lon=lon, method='nearest').values) for lat, lon in zip(actual_lats, actual_lons)]
    ax.scatter(actual_lons, actual_lats, c=ent_lai, s=100, edgecolors='black', linewidths=1.5,
               transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=7, zorder=100)
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_title('MODIS LAI Temporal Mean', fontsize=14, fontweight='bold')
    output_path = os.path.join(output_dir, 'modis_lai_temporal_mean.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Re-create summer mean map with ENT points
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    lai_summer_mean.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet',
                        cbar_kwargs={'label': 'Leaf Area Index (m²/m²)', 'shrink': 0.8},
                        vmin=0, vmax=7)
    ent_lai = [float(lai_summer_mean.sel(lat=lat, lon=lon, method='nearest').values) for lat, lon in zip(actual_lats, actual_lons)]
    ax.scatter(actual_lons, actual_lats, c=ent_lai, s=100, edgecolors='black', linewidths=1.5,
               transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=7, zorder=100)
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_title('MODIS LAI Summer Mean (June-July)', fontsize=14, fontweight='bold')
    output_path_summer = os.path.join(output_dir, 'modis_lai_summer_mean.png')
    plt.savefig(output_path_summer, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Re-create annual max map with ENT points
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    lai_mean_annual_max.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet',
                            cbar_kwargs={'label': 'Leaf Area Index (m²/m²)', 'shrink': 0.8},
                            vmin=0, vmax=7)
    ent_lai = [float(lai_mean_annual_max.sel(lat=lat, lon=lon, method='nearest').values) for lat, lon in zip(actual_lats, actual_lons)]
    ax.scatter(actual_lons, actual_lats, c=ent_lai, s=100, edgecolors='black', linewidths=1.5,
               transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=7, zorder=100)
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_title('MODIS LAI Mean Annual Maximum', fontsize=14, fontweight='bold')
    output_path_max = os.path.join(output_dir, 'modis_lai_mean_annual_maximum.png')
    plt.savefig(output_path_max, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Re-create difference map with ENT points
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    lai_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet',
                 cbar_kwargs={'label': 'LAI Difference (m²/m²)', 'shrink': 0.8},
                 vmin=0, vmax=3)
    ent_lai = [float(lai_diff.sel(lat=lat, lon=lon, method='nearest').values) for lat, lon in zip(actual_lats, actual_lons)]
    ax.scatter(actual_lons, actual_lats, c=ent_lai, s=100, edgecolors='black', linewidths=1.5,
               transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=3, zorder=100)
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_title('MODIS LAI: Summer Mean - Annual Mean', fontsize=14, fontweight='bold')
    output_path_diff = os.path.join(output_dir, 'modis_lai_summer_minus_annual.png')
    plt.savefig(output_path_diff, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Maps updated with {len(actual_lats)} ENT points")
    
    # Create figure with all timeseries
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All individual timeseries with distinct colors
    time_index = np.arange(len(ds['time']))
    colors = plt.cm.tab20(np.linspace(0, 1, len(ent_timeseries)))
    for i, ts in enumerate(ent_timeseries):
        ax1.plot(time_index, ts, alpha=0.7, linewidth=1.5, color=colors[i], 
                label=f'Point {i+1}' if len(ent_timeseries) <= 20 else None)
    
    # Calculate and plot ensemble statistics
    mean_ts = np.nanmean(ent_timeseries, axis=0)
    std_ts = np.nanstd(ent_timeseries, axis=0)
    
    ax1.plot(time_index, mean_ts, color='darkgreen', linewidth=3, 
             label=f'Mean across {len(ent_timeseries)} points', zorder=100)
    ax1.fill_between(time_index, mean_ts - std_ts, mean_ts + std_ts, 
                     color='darkgreen', alpha=0.2, label='±1 std dev')
    
    # Add overall mean, summer mean, and annual max lines
    overall_mean = np.nanmean(ent_timeseries)
    summer_ts = ent_timeseries[:, summer_mask]
    summer_mean = np.nanmean(summer_ts)
    
    # Calculate mean annual maximum
    ent_annual_max_list = []
    for ts in ent_timeseries:
        ts_reshaped = ts[:n_years*12].reshape(n_years, 12)
        annual_max = np.max(ts_reshaped, axis=1)
        ent_annual_max_list.append(np.mean(annual_max))
    mean_annual_max = np.mean(ent_annual_max_list)
    
    ax1.axhline(y=overall_mean, color='blue', linestyle='--', linewidth=2, 
               label=f'Overall mean: {overall_mean:.2f}', zorder=101)
    ax1.axhline(y=summer_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Summer mean: {summer_mean:.2f}', zorder=101)
    ax1.axhline(y=mean_annual_max, color='purple', linestyle='--', linewidth=2, 
               label=f'Mean annual max: {mean_annual_max:.2f}', zorder=101)
    
    ax1.set_xlabel('Time Index (months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LAI [m²/m²]', fontsize=12, fontweight='bold')
    ax1.set_title(f'MODIS LAI Timeseries for {len(ent_timeseries)} ENT Grid Points', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    
    # Plot 2: Statistics over time
    min_ts = np.nanmin(ent_timeseries, axis=0)
    max_ts = np.nanmax(ent_timeseries, axis=0)
    q25_ts = np.nanpercentile(ent_timeseries, 25, axis=0)
    q75_ts = np.nanpercentile(ent_timeseries, 75, axis=0)
    
    ax2.fill_between(time_index, min_ts, max_ts, color='lightgreen', 
                     alpha=0.3, label='Min-Max range')
    ax2.fill_between(time_index, q25_ts, q75_ts, color='green', 
                     alpha=0.4, label='25th-75th percentile')
    ax2.plot(time_index, mean_ts, color='darkgreen', linewidth=3, 
            label='Mean', zorder=100)
    
    ax2.axhline(y=overall_mean, color='blue', linestyle='--', linewidth=2, 
               label=f'Overall mean: {overall_mean:.2f}')
    ax2.axhline(y=summer_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Summer mean: {summer_mean:.2f}')
    ax2.axhline(y=mean_annual_max, color='purple', linestyle='--', linewidth=2, 
               label=f'Mean annual max: {mean_annual_max:.2f}')
    
    ax2.set_xlabel('Time Index (months)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('LAI [m²/m²]', fontsize=12, fontweight='bold')
    ax2.set_title('ENT LAI Statistics Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path_ent_ts = os.path.join(output_dir, 'modis_lai_ent_timeseries_all_points.png')
    plt.savefig(output_path_ent_ts, dpi=300, bbox_inches='tight')
    print(f"\nENT timeseries plot saved to: {output_path_ent_ts}")
    
    plt.close()
    
    # Print statistics
    print(f"\nENT grid points LAI statistics:")
    print(f"Overall mean: {overall_mean:.3f}")
    print(f"Summer mean: {summer_mean:.3f}")
    print(f"Difference (summer - overall): {summer_mean - overall_mean:.3f}")
    print(f"Min across all points/times: {np.nanmin(ent_timeseries):.3f}")
    print(f"Max across all points/times: {np.nanmax(ent_timeseries):.3f}")
    print(f"Std across points (temporal mean): {np.nanstd(np.nanmean(ent_timeseries, axis=1)):.3f}")
    
    # Calculate mean annual maximum for ENT points
    print("\nCalculating mean annual maximum for ENT points...")
    ent_annual_max_values = []
    for ts in ent_timeseries:
        # Reshape to (years, 12 months) and find max for each year, then average
        ts_reshaped = ts[:n_years*12].reshape(n_years, 12)
        annual_max = np.max(ts_reshaped, axis=1)
        ent_annual_max_values.append(np.mean(annual_max))
    ent_annual_max_values = np.array(ent_annual_max_values)
    
    # Calculate temporal mean and summer mean for each ENT point
    ent_temporal_mean = np.nanmean(ent_timeseries, axis=1)
    ent_summer_mean = np.nanmean(ent_timeseries[:, summer_mask], axis=1)
    
    # Create comparison plot
    print("\nCreating comparison plot of ENT LAI statistics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart comparing the three metrics
    x_pos = np.arange(len(ent_temporal_mean))
    width = 0.25
    
    ax1.bar(x_pos - width, ent_temporal_mean, width, label='Temporal Mean', color='blue', alpha=0.7)
    ax1.bar(x_pos, ent_summer_mean, width, label='Summer Mean', color='orange', alpha=0.7)
    ax1.bar(x_pos + width, ent_annual_max_values, width, label='Mean Annual Max', color='red', alpha=0.7)
    
    ax1.set_xlabel('ENT Point Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LAI [m²/m²]', fontsize=12, fontweight='bold')
    ax1.set_title('Comparison of LAI Statistics for Each ENT Point', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{i+1}' for i in range(len(ent_temporal_mean))])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Box plot showing distributions
    data_for_box = [ent_temporal_mean, ent_summer_mean, ent_annual_max_values]
    bp = ax2.boxplot(data_for_box, labels=['Temporal\nMean', 'Summer\nMean', 'Mean Annual\nMax'],
                     patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = ['blue', 'orange', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('LAI [m²/m²]', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of LAI Statistics Across ENT Points', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, (data, color) in enumerate(zip(data_for_box, colors)):
        mean_val = np.mean(data)
        ax2.text(i+1, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color=color)
    
    plt.tight_layout()
    
    # Save figure
    output_path_ent_comparison = os.path.join(output_dir, 'modis_lai_ent_statistics_comparison.png')
    plt.savefig(output_path_ent_comparison, dpi=300, bbox_inches='tight')
    print(f"ENT statistics comparison plot saved to: {output_path_ent_comparison}")
    
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary statistics across all ENT points:")
    print(f"Temporal Mean    - Min: {ent_temporal_mean.min():.3f}, Max: {ent_temporal_mean.max():.3f}, Mean: {ent_temporal_mean.mean():.3f}")
    print(f"Summer Mean      - Min: {ent_summer_mean.min():.3f}, Max: {ent_summer_mean.max():.3f}, Mean: {ent_summer_mean.mean():.3f}")
    print(f"Mean Annual Max  - Min: {ent_annual_max_values.min():.3f}, Max: {ent_annual_max_values.max():.3f}, Mean: {ent_annual_max_values.mean():.3f}")
    
    # Create a map showing the ENT grid point locations
    print("\nCreating map of ENT grid point locations...")
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Plot the temporal mean LAI as background
    lai_mean.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='jet',
        cbar_kwargs={'label': 'LAI [m²/m²]', 'shrink': 0.8},
        vmin=0,
        vmax=7,
        alpha=0.6
    )
    
    # Extract LAI values at ENT point locations
    ent_lai_values = []
    for lat, lon in zip(actual_lats, actual_lons):
        ent_lai_values.append(float(lai_mean.sel(lat=lat, lon=lon, method='nearest').values))
    ent_lai_values = np.array(ent_lai_values)
    
    # Plot ENT grid points colored by LAI values
    scatter = ax.scatter(actual_lons, actual_lats, c=ent_lai_values, s=100, 
                        edgecolors='black', linewidths=1.5, 
                        transform=ccrs.PlateCarree(), cmap='jet',
                        vmin=0, vmax=7, zorder=100,
                        label=f'{len(actual_lats)} ENT points')
    
    ax.coastlines()
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_title(f'ENT Grid Point Locations ({len(actual_lats)} points)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    
    # Save map
    output_path_ent_map = os.path.join(output_dir, 'ent_grid_points_map.png')
    plt.savefig(output_path_ent_map, dpi=300, bbox_inches='tight')
    print(f"ENT grid points map saved to: {output_path_ent_map}")
    
    plt.close()
    
else:
    print(f"\nWARNING: ENT grid cells file not found: {ent_file}")

# Regrid MODIS LAI to CLM model grid using xESMF
print("\n" + "="*60)
print("Regridding MODIS LAI to CLM model grid")
print("="*60)

clm_file = '/datalake/NS9560K/rosief/i2000.f45_f45_mg37.fates-nocomp.beta01.v25f/lnd/hist/i2000.f45_f45_mg37.fates-nocomp.beta01.v25f.clm2.h0.2001-01.nc'

if os.path.exists(clm_file):
    print(f"\nLoading CLM grid from: {clm_file}")
    ds_clm = xr.open_dataset(clm_file)
    
    print(f"CLM grid dimensions:")
    print(f"  lat: {ds_clm.dims.get('lat', ds_clm.dims.get('lsmlat', 'not found'))}")
    print(f"  lon: {ds_clm.dims.get('lon', ds_clm.dims.get('lsmlon', 'not found'))}")
    
    # Determine coordinate names in CLM file
    if 'lat' in ds_clm.coords and 'lon' in ds_clm.coords:
        clm_lat_name = 'lat'
        clm_lon_name = 'lon'
    elif 'lsmlat' in ds_clm.coords and 'lsmlon' in ds_clm.coords:
        clm_lat_name = 'lsmlat'
        clm_lon_name = 'lsmlon'
    else:
        print("ERROR: Could not find lat/lon coordinates in CLM file")
        clm_lat_name = None
        clm_lon_name = None
    
    if clm_lat_name and clm_lon_name:
        # Prepare target grid (CLM)
        ds_out = xr.Dataset({
            'lat': ds_clm[clm_lat_name],
            'lon': ds_clm[clm_lon_name]
        })
        
        # Check if there's a land mask to apply
        landmask = None
        if 'landmask' in ds_clm:
            landmask = ds_clm['landmask']
            print(f"Found landmask in CLM file")
        elif 'LANDMASK' in ds_clm:
            landmask = ds_clm['LANDMASK']
            print(f"Found LANDMASK in CLM file")
        
        # Prepare source grid (MODIS) - ensure lon is 0-360
        ds_in = xr.Dataset({
            'lat': ds['lat'],
            'lon': ds['lon']
        })
        
        print(f"\nSource (MODIS) grid:")
        print(f"  lat: {len(ds_in['lat'])} points, range [{float(ds_in['lat'].min()):.2f}, {float(ds_in['lat'].max()):.2f}]")
        print(f"  lon: {len(ds_in['lon'])} points, range [{float(ds_in['lon'].min()):.2f}, {float(ds_in['lon'].max()):.2f}]")
        
        print(f"\nTarget (CLM) grid:")
        print(f"  lat: {len(ds_out['lat'])} points, range [{float(ds_out['lat'].min()):.2f}, {float(ds_out['lat'].max()):.2f}]")
        print(f"  lon: {len(ds_out['lon'])} points, range [{float(ds_out['lon'].min()):.2f}, {float(ds_out['lon'].max()):.2f}]")
        
        # Create regridder - using bilinear interpolation
        print("\nCreating xESMF regridder (bilinear)...")
        regridder = xe.Regridder(ds_in, ds_out, 'bilinear', periodic=True)
        print("Regridder created successfully")
        
        # Regrid the temporal mean LAI
        print("\nRegridding temporal mean LAI...")
        lai_mean_regrid = regridder(lai_mean)
        lai_mean_regrid.attrs['long_name'] = 'MODIS LAI temporal mean regridded to CLM grid'
        lai_mean_regrid.attrs['units'] = 'm²/m²'
        
        # Regrid the summer mean LAI
        print("Regridding summer mean LAI...")
        lai_summer_mean_regrid = regridder(lai_summer_mean)
        lai_summer_mean_regrid.attrs['long_name'] = 'MODIS LAI summer mean regridded to CLM grid'
        lai_summer_mean_regrid.attrs['units'] = 'm²/m²'
        
        # Regrid the mean annual maximum LAI
        print("Regridding mean annual maximum LAI...")
        lai_mean_annual_max_regrid = regridder(lai_mean_annual_max)
        lai_mean_annual_max_regrid.attrs['long_name'] = 'MODIS LAI mean annual maximum regridded to CLM grid'
        lai_mean_annual_max_regrid.attrs['units'] = 'm²/m²'
        
        # Save regridded data as NetCDF files
        output_nc_mean_regrid = os.path.join(output_dir, 'modis_lai_temporal_mean_clm_grid.nc')
        try:
            lai_mean_regrid.to_netcdf(output_nc_mean_regrid, mode='w')
            print(f"\nRegridded temporal mean saved to: {output_nc_mean_regrid}")
        except PermissionError:
            print(f"\nWarning: Could not save {output_nc_mean_regrid} (permission denied)")
        
        output_nc_summer_regrid = os.path.join(output_dir, 'modis_lai_summer_mean_clm_grid.nc')
        try:
            lai_summer_mean_regrid.to_netcdf(output_nc_summer_regrid, mode='w')
            print(f"Regridded summer mean saved to: {output_nc_summer_regrid}")
        except PermissionError:
            print(f"Warning: Could not save {output_nc_summer_regrid} (permission denied)")
        
        output_nc_max_regrid = os.path.join(output_dir, 'modis_lai_mean_annual_maximum_clm_grid.nc')
        try:
            lai_mean_annual_max_regrid.to_netcdf(output_nc_max_regrid, mode='w')
            print(f"Regridded mean annual maximum saved to: {output_nc_max_regrid}")
        except PermissionError:
            print(f"Warning: Could not save {output_nc_max_regrid} (permission denied)")
        
        # Create comparison plots
        print("\nCreating comparison plots...")
        
        # Plot comparison for temporal mean
        fig, axes = plt.subplots(1, 2, figsize=(20, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Original MODIS
        im1 = lai_mean.plot(
            ax=axes[0],
            transform=ccrs.PlateCarree(),
            cmap='jet',
            cbar_kwargs={'label': 'LAI [m²/m²]', 'shrink': 0.8},
            vmin=0,
            vmax=7
        )
        axes[0].coastlines()
        axes[0].gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        axes[0].set_title('Original MODIS LAI (0.5°)', fontsize=12, fontweight='bold')
        
        # Regridded to CLM
        im2 = lai_mean_regrid.plot(
            ax=axes[1],
            transform=ccrs.PlateCarree(),
            cmap='jet',
            cbar_kwargs={'label': 'LAI [m²/m²]', 'shrink': 0.8},
            vmin=0,
            vmax=7
        )
        axes[1].coastlines()
        axes[1].gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        axes[1].set_title('Regridded to CLM Grid', fontsize=12, fontweight='bold')
        
        plt.suptitle('MODIS LAI Temporal Mean: Original vs Regridded', fontsize=14, fontweight='bold')
        
        output_path_comparison = os.path.join(output_dir, 'modis_lai_regridding_comparison.png')
        plt.savefig(output_path_comparison, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path_comparison}")
        plt.close()
        
        # Create individual maps for each regridded LAI product on CLM grid
        print("\nCreating individual maps for regridded LAI on CLM grid...")
        
        # 1. Temporal mean on CLM grid
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        im = lai_mean_regrid.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='jet',
            cbar_kwargs={'label': 'LAI [m²/m²]', 'shrink': 0.8},
            vmin=0,
            vmax=7
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        ax.set_title('MODIS LAI Temporal Mean (CLM Grid)', fontsize=14, fontweight='bold')
        
        output_path_mean_clm = os.path.join(output_dir, 'modis_lai_temporal_mean_clm_grid.png')
        plt.savefig(output_path_mean_clm, dpi=300, bbox_inches='tight')
        print(f"CLM grid temporal mean plot saved to: {output_path_mean_clm}")
        plt.close()
        
        # 2. Summer mean on CLM grid
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        im = lai_summer_mean_regrid.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='jet',
            cbar_kwargs={'label': 'LAI [m²/m²]', 'shrink': 0.8},
            vmin=0,
            vmax=7
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        ax.set_title('MODIS LAI Summer Mean (CLM Grid)', fontsize=14, fontweight='bold')
        
        output_path_summer_clm = os.path.join(output_dir, 'modis_lai_summer_mean_clm_grid.png')
        plt.savefig(output_path_summer_clm, dpi=300, bbox_inches='tight')
        print(f"CLM grid summer mean plot saved to: {output_path_summer_clm}")
        plt.close()
        
        # 3. Mean annual maximum on CLM grid
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        im = lai_mean_annual_max_regrid.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='jet',
            cbar_kwargs={'label': 'LAI [m²/m²]', 'shrink': 0.8},
            vmin=0,
            vmax=7
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        ax.set_title('MODIS LAI Mean Annual Maximum (CLM Grid)', fontsize=14, fontweight='bold')
        
        output_path_max_clm = os.path.join(output_dir, 'modis_lai_mean_annual_maximum_clm_grid.png')
        plt.savefig(output_path_max_clm, dpi=300, bbox_inches='tight')
        print(f"CLM grid mean annual maximum plot saved to: {output_path_max_clm}")
        plt.close()
        
        # Print regridded statistics
        print(f"\nRegridded LAI statistics:")
        print(f"Temporal mean - Min: {float(lai_mean_regrid.min()):.3f}, Max: {float(lai_mean_regrid.max()):.3f}, Mean: {float(lai_mean_regrid.mean()):.3f}")
        print(f"Summer mean - Min: {float(lai_summer_mean_regrid.min()):.3f}, Max: {float(lai_summer_mean_regrid.max()):.3f}, Mean: {float(lai_summer_mean_regrid.mean()):.3f}")
        print(f"Mean annual max - Min: {float(lai_mean_annual_max_regrid.min()):.3f}, Max: {float(lai_mean_annual_max_regrid.max()):.3f}, Mean: {float(lai_mean_annual_max_regrid.mean()):.3f}")
        
        # Clean up
        ds_clm.close()
    else:
        print("Could not determine CLM grid coordinates")
else:
    print(f"\nWARNING: CLM file not found: {clm_file}")

print("\n" + "="*60)
print("All plots completed!")
print("="*60)
