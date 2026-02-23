This file has been removed as per the request.
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# CONFIGURATION
CASENAME = "noresm-fates_ne16_transient_1900_borealseeds.2025-11-19"
DATA_PATH = "/datalake/NS9188K/share/jessien/"
DATA_DIR = Path(f"{DATA_PATH}/{CASENAME}/run/")
YEAR_RANGE = (1, 2)
FILE_INTERVAL = 2
VAR_NAME = "FATES_TRANSITION_MATRIX_LULU"
SCALE_FACTOR = 100 * 12


OUTPUT_PATH = f"/datalake/NS9560K/www/diagnostics/noresm/rosief/{CASENAME}/gifs"

# Find files
file_pattern = str(DATA_DIR / "*h0*.nc")
all_files = sorted(glob.glob(file_pattern))
if not all_files:
    print(f"No files found: {file_pattern}")
    exit(1)

# Build time map
file_time_map = []
start_timestep = YEAR_RANGE[0] * 12 if YEAR_RANGE else None
end_timestep = YEAR_RANGE[1] * 12 if YEAR_RANGE else None
timestep_counter = 0
for i, file_path in enumerate(all_files):
    if start_timestep is not None and timestep_counter < start_timestep:
        timestep_counter += 1
        continue
    if end_timestep is not None and timestep_counter >= end_timestep:
        break
    if (timestep_counter - (start_timestep or 0)) % FILE_INTERVAL == 0:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            file_time_map.append((file_path, 0, ds.time.values[0]))
    timestep_counter += 1
n_times = len(file_time_map)

print(f"Processing {n_times} timesteps.")

# Read metadata
with xr.open_dataset(file_time_map[0][0], decode_times=False) as ds:
    var = ds[VAR_NAME]
    if var.shape[1] == 25:
        n_lu = 5
    else:
        n_lu = var.shape[1] if len(var.shape) > 1 else 1
    if 'ncol' in var.dims:
        lat = ds['lat'].values if 'lat' in ds else ds['grid_lat'].values
        lon = ds['lon'].values if 'lon' in ds else ds['grid_lon'].values
    elif 'lndgrid' in var.dims:
        lat = ds['lat'].values
        lon = ds['lon'].values
    else:
        lat = ds.lat.values
        lon = ds.lon.values

# Sum over all subplots and timesteps
sum_map = None
for idx, (file_path, time_idx, _) in enumerate(file_time_map):
    with xr.open_dataset(file_path, decode_times=False) as ds:
        data = ds[VAR_NAME].isel(time=time_idx).values * SCALE_FACTOR
        if data.shape[-1] == 25:
            data = data.reshape(*data.shape[:-1], 5, 5)
        # Sum over LUxLU transitions
        if data.ndim == 3:
            summed = np.nansum(data, axis=(0,1))  # shape (ngrid,)
        elif data.ndim == 2:
            summed = np.nansum(data, axis=0)
        else:
            summed = np.nansum(data)
        if sum_map is None:
            sum_map = np.zeros_like(summed)
        sum_map += summed

# Plot static map
plt.figure(figsize=(8,6))
if lat.ndim == 1 and lon.ndim == 1 and sum_map.shape[0] == len(lon):
    plt.scatter(lon, lat, c=sum_map, cmap='viridis', s=8)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
else:
    plt.imshow(sum_map.reshape(lat.shape), cmap='viridis', origin='lower')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
plt.title('Sum of LUxLU transitions over timeseries')
plt.colorbar(label='Sum (%/month)')
plt.tight_layout()
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
output_png = f"{OUTPUT_PATH}/sum_luxlu_static.png"
plt.savefig(output_png, dpi=120)
plt.close()
print(f"Saved static sum plot: {output_png}")
