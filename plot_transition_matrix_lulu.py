# Land use class abbreviations for subplot titles
LU_ABBR = ['PR', 'SC', 'RG', 'PA', 'CR']
# Scale factor to convert from m2/m2/yr to %/month

#!/usr/bin/env python
"""
Script to visualize FATES_TRANSITION_MATRIX_LULU as a timeseries GIF.
Each subplot shows a map for a LUxLU (land use class) transition combination.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import glob


def to_luxlu_pairs(data, n_lu):
    n_pairs = n_lu * n_lu
    if data.ndim == 3 and data.shape[:2] == (n_lu, n_lu):
        return data.reshape(n_pairs, data.shape[2])
    if data.ndim == 2 and data.shape[0] == n_pairs:
        return data
    if data.ndim == 2 and data.shape[1] == n_pairs:
        return data.T
    if data.ndim == 3 and data.shape[-1] == n_pairs:
        return data.reshape(-1, n_pairs).T
    return None


# CONFIGURATION
jessie_run=1
if(jessie_run==1):
    DATA_PATH="/datalake/NS9188K/share/jessien/"
    CASENAME="noresm-fates_f45_2000s_ent_agbonly.2026-02-03"
    CASENAME='noresm-fates_ne16_transient_1900_borealseeds.2025-11-19'
    DATA_DIR = Path(f"{DATA_PATH}/{CASENAME}/run/")
    YEAR_RANGE = (1, 2)
    FILE_INTERVAL = 1
else:
    CASENAME = "i2000.ne16pg3_tn14.fatesnocomp.noresm3_0_beta10.CRUJRA.06f_14hp_noLUC.2026-02-09"
    DATA_DIR = Path(f"//datalake/NS9560K/rosief/{CASENAME}/lnd/hist")
    YEAR_RANGE = (1, 10)
    FILE_INTERVAL = 12
#  need to convert from per year to per month
#  and then adjust for file interval (e.g., if files are every 6 months, then we want to divide by 6 to get per month rate)

SCALE_FACTOR = 12/FILE_INTERVAL  # Convert from m2/m2/yr to %/month (adjusted for file interval) 
OUTPUT_PATH = f"/datalake/NS9560K/www/diagnostics/noresm/rosief/{CASENAME}/gifs"

VAR_NAME = "FATES_TRANSITION_MATRIX_LULU"
SHARED_CMAP = 'RdBu'
SHARED_CLIM_PERCENTILE = 98.0

# Shared layout (keeps static PNG and GIF colorbars aligned)
SUBPLOT_LEFT = 0.12
SUBPLOT_RIGHT = 0.90
SUBPLOT_WSPACE = 0.25
CBAR_RIGHT_POS = [0.94, 0.15, 0.02, 0.7]
CBAR_LEFT_POS = [0.05, 0.15, 0.02, 0.7]


OUTPUT_FORMAT = "gif"
tint = 200  # ms per frame

# Find files
file_pattern = str(DATA_DIR / "*h0*.nc")
all_files = sorted(glob.glob(file_pattern))
if not all_files:
    print(f"No files found: {file_pattern}")
    exit(1)

print(f"Found {len(all_files)} files matching pattern.")

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
    # Detect grid type
    if 'ncol' in var.dims:
        grid_type = 'ncol'
        lat = ds['lat'].values if 'lat' in ds else ds['grid_lat'].values
        lon = ds['lon'].values if 'lon' in ds else ds['grid_lon'].values
        n_points = len(lat)
    elif 'lndgrid' in var.dims:
        grid_type = 'lndgrid'
        lat = ds['lat'].values
        lon = ds['lon'].values
        n_points = len(lat)
    else:
        grid_type = 'regular'
        lat = ds.lat.values
        lon = ds.lon.values
    # Fix LUxLU unpacking for shape (1, 25, 13824)
    if var.shape[1] == 25:
        n_lu = 5
        print("Detected LUxLU dimension as 25, unpacking to 5x5 array.")
    else:
        n_lu = var.shape[1] if len(var.shape) > 1 else 1
    print(f"Variable shape: {var.shape}, n_lu: {n_lu}, grid_type: {grid_type}")

# Setup figure: LUxLU subplots
fig, axs = plt.subplots(n_lu, n_lu, figsize=(4.5*n_lu, 3*n_lu),
                       subplot_kw={'projection': None} if n_lu <= 4 else {})
if n_lu == 1:
    axs = np.array([[axs]])
# Add a global colorbar axis for animation (further right)
cb_ax = fig.add_axes(CBAR_RIGHT_POS)
fig.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, wspace=SUBPLOT_WSPACE)

# Find vmin/vmax for color scale

# Find vmin/vmax for transition matrix
print("Calculating color scale (vmin/vmax) across all timesteps...")
vmin, vmax = np.inf, -np.inf
# Find vmin/vmax for patch area
patcharea_vmin, patcharea_vmax = np.inf, -np.inf
# Sum LUxLU transitions over period (per pair, per grid point)
transition_sum_pairs = None
transition_abs_values = []
for idx, (file_path, time_idx, _) in enumerate(file_time_map):
    print(f"  Checking file {idx+1}/{len(file_time_map)}: {file_path}")
    with xr.open_dataset(file_path, decode_times=False) as ds:
        data = ds[VAR_NAME].isel(time=time_idx).values 
        if data.shape[-1] == 25:
            data = data.reshape(*data.shape[:-1], 5, 5)
        vmin = min(vmin, np.nanmin(data))
        vmax = max(vmax, np.nanmax(data))
        # Patch area
        if 'FATES_PATCHAREA_LU' in ds:
            patcharea = ds['FATES_PATCHAREA_LU'].isel(time=time_idx).values
            patcharea_vmin = min(patcharea_vmin, np.nanmin(patcharea))
            patcharea_vmax = max(patcharea_vmax, np.nanmax(patcharea))
        pair_data = to_luxlu_pairs(data, n_lu)
        if pair_data is not None:
            if transition_sum_pairs is None:
                transition_sum_pairs = np.zeros_like(pair_data)
            transition_sum_pairs += pair_data
            transition_abs_values.append(np.abs(pair_data).ravel())
print(f"Color scale: vmin={vmin}, vmax={vmax}")
print(f"Patch area color scale: vmin={patcharea_vmin}, vmax={patcharea_vmax}")
if transition_sum_pairs is not None:
    cum_vmin = np.nanmin(transition_sum_pairs)
    cum_vmax = np.nanmax(transition_sum_pairs)
else:
    cum_vmin, cum_vmax = vmin, vmax

if transition_abs_values:
    transition_abs_all = np.concatenate(transition_abs_values)
    transition_abs_all = transition_abs_all[np.isfinite(transition_abs_all)]
    if transition_abs_all.size > 0:
        trans_absmax = np.nanpercentile(transition_abs_all, SHARED_CLIM_PERCENTILE)
    else:
        trans_absmax = max(abs(vmin), abs(vmax))
else:
    trans_absmax = max(abs(vmin), abs(vmax))
if trans_absmax == 0:
    trans_absmax = 1e-12

cum_abs_values = np.abs(transition_sum_pairs) if transition_sum_pairs is not None else np.array([])
cum_abs_values = cum_abs_values[np.isfinite(cum_abs_values)] if cum_abs_values.size > 0 else cum_abs_values
if cum_abs_values.size > 0:
    cum_absmax = np.nanpercentile(cum_abs_values, SHARED_CLIM_PERCENTILE)
else:
    cum_absmax = max(abs(cum_vmin), abs(cum_vmax))
if cum_absmax == 0:
    cum_absmax = 1e-12

# Shared VEGC metadata (used for both VEGC-only and LUxLU diagonal plots)
first_file, _, _ = file_time_map[0]
last_file, _, _ = file_time_map[-1]
vegc_diff_global = None
vegc_absmax_global = 0.0
with xr.open_dataset(first_file, decode_times=False) as ds_first, xr.open_dataset(last_file, decode_times=False) as ds_last:
    if 'FATES_PATCHAREA_LU' in ds_first and 'FATES_PATCHAREA_LU' in ds_last:
        first_time_idx = 0
        last_time_idx = ds_last['FATES_PATCHAREA_LU'].shape[0] - 1
        vegc_first = ds_first['FATES_PATCHAREA_LU'].isel(time=first_time_idx).values
        vegc_last = ds_last['FATES_PATCHAREA_LU'].isel(time=last_time_idx).values
        vegc_diff = (vegc_last - vegc_first)
        if vegc_diff.shape[0] == n_lu:
            vegc_diff_global = vegc_diff
        elif vegc_diff.shape[1] == n_lu:
            vegc_diff_global = vegc_diff.T
        if vegc_diff_global is not None:
            vegc_abs = np.abs(vegc_diff_global)
            vegc_abs = vegc_abs[np.isfinite(vegc_abs)]
            if vegc_abs.size > 0:
                vegc_absmax_global = np.nanpercentile(vegc_abs, SHARED_CLIM_PERCENTILE)

shared_absmax = max(trans_absmax, cum_absmax, vegc_absmax_global)
if shared_absmax == 0:
    shared_absmax = 1e-12
print(f"Cumulative LUxLU color scale: vmin={cum_vmin}, vmax={cum_vmax}")
print(f"Transition color limit (±): {trans_absmax} (p{SHARED_CLIM_PERCENTILE})")
print(f"Cumulative color limit (±): {cum_absmax} (p{SHARED_CLIM_PERCENTILE})")
print(f"VEGC color limit (±): {vegc_absmax_global} (p{SHARED_CLIM_PERCENTILE})")
print(f"Shared color limit (±): {shared_absmax}")

# --- Plot last timestep as static LUxLU transitions map ---
# --- Plot sum of LUxLU transitions over timeseries ---
print("Plotting sum of LUxLU transitions over timeseries...")
sum_data = None
for idx, (file_path, time_idx, _) in enumerate(file_time_map):
    with xr.open_dataset(file_path, decode_times=False) as ds:
        data = ds[VAR_NAME].isel(time=time_idx).values 
        if data.shape[-1] == 25:
            data = data.reshape(*data.shape[:-1], 5, 5)
        if data.ndim == 3:
            summed = np.nansum(data, axis=(0,1))  # shape (ngrid,)
        elif data.ndim == 2:
            summed = np.nansum(data, axis=0)
        else:
            summed = np.nansum(data)
        if sum_data is None:
            sum_data = np.zeros_like(summed)
        sum_data += summed

fig_sum, ax_sum = plt.subplots(figsize=(8,6))
if lat.ndim == 1 and lon.ndim == 1 and sum_data.shape[0] == len(lon):
    sc = ax_sum.scatter(lon, lat, c=sum_data, cmap=SHARED_CMAP, s=8, vmin=-shared_absmax, vmax=shared_absmax)
    ax_sum.set_xlabel('Longitude')
    ax_sum.set_ylabel('Latitude')
else:
    sc = ax_sum.imshow(sum_data.reshape(lat.shape), cmap=SHARED_CMAP, origin='lower', vmin=-shared_absmax, vmax=shared_absmax)
    ax_sum.set_xlabel('Longitude')
    ax_sum.set_ylabel('Latitude')
fig_sum.colorbar(sc, ax=ax_sum, label='Sum of LUxLU transitions (%/month)')
fig_sum.suptitle('Sum of LUxLU transitions over timeseries', fontsize=18)
plt.tight_layout()
output_sum = f"{OUTPUT_PATH}/sum_luxlu_static_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}.png"
fig_sum.savefig(output_sum, dpi=120)
plt.close(fig_sum)
print(f"Saved static sum plot: {output_sum}")

# --- Plot difference between first and last FATES_VEGC_LU maps ---
print("Plotting difference between first and last FATES_PATCHAREA_LU maps as LUxLU grid...")
if vegc_diff_global is not None:
    fig_vegc, axs_vegc = plt.subplots(1, n_lu, figsize=(4.5*n_lu, 3))
    mappables = []
    for i in range(n_lu):
        ax = axs_vegc[i]
        d = vegc_diff_global[i, :]
        if d.shape[0] != len(lon):
            d = np.full(len(lon), np.nan)
        sc = ax.scatter(lon, lat, c=d, cmap=SHARED_CMAP, s=8, vmin=-shared_absmax, vmax=shared_absmax)
        mappables.append(sc)
        ax.set_title(f'LU {LU_ABBR[i]}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    fig_vegc.colorbar(mappables[0], ax=axs_vegc, orientation='vertical', label='Δ FATES_PATCHAREA_LU')
    fig_vegc.suptitle('Difference between first and last FATES_PATCHAREA_LU (per LU class)', fontsize=18)
    plt.tight_layout()
    output_vegc = f"{OUTPUT_PATH}/diff_vegc_lu_static.png"
    fig_vegc.savefig(output_vegc, dpi=120)
    plt.close(fig_vegc)
    print(f"Saved static vegc diff plot: {output_vegc}")
else:
    print("FATES_PATCHAREA_LU not found in files. Skipping vegc diff plot.")
print("Plotting last timestep as static LUxLU transitions map...")
last_file_path, last_time_idx, last_time_val = file_time_map[-1]
with xr.open_dataset(last_file_path, decode_times=False) as ds:
    data = ds[VAR_NAME].isel(time=last_time_idx).values 
    # Always define lat_ and lon_ first
    if grid_type == 'ncol':
        lat_ = ds['lat'].values if 'lat' in ds else ds['grid_lat'].values
        lon_ = ds['lon'].values if 'lon' in ds else ds['grid_lon'].values
    elif grid_type == 'lndgrid':
        lat_ = ds['lat'].values
        lon_ = ds['lon'].values
    else:
        lat_ = lat
        lon_ = lon
    print(f"Static plot: original data shape {data.shape}")
    ngrid = len(lon_)
    sum_pairs_current_grid = transition_sum_pairs
    if sum_pairs_current_grid is None or sum_pairs_current_grid.shape[1] != ngrid:
        current_pairs = to_luxlu_pairs(data, n_lu)
        if current_pairs is not None:
            sum_pairs_current_grid = current_pairs
        else:
            sum_pairs_current_grid = np.full((n_lu * n_lu, ngrid), np.nan)
    print(f"Static plot: cumulative pair data shape {sum_pairs_current_grid.shape}, ngrid={ngrid}, lon_ shape={lon_.shape}")
    if grid_type == 'ncol':
        lat_ = ds['lat'].values if 'lat' in ds else ds['grid_lat'].values
        lon_ = ds['lon'].values if 'lon' in ds else ds['grid_lon'].values
    elif grid_type == 'lndgrid':
        lat_ = ds['lat'].values
        lon_ = ds['lon'].values
    else:
        lat_ = lat
        lon_ = lon

fig_last, axs_last = plt.subplots(n_lu, n_lu, figsize=(4.5*n_lu, 3*n_lu),
                                 subplot_kw={'projection': None} if n_lu <= 4 else {})
cb_ax_last = fig_last.add_axes(CBAR_RIGHT_POS)
cb_ax_patch = fig_last.add_axes(CBAR_LEFT_POS)
fig_last.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, wspace=SUBPLOT_WSPACE)
if n_lu == 1:
    axs_last = np.array([[axs_last]])

# Store mappables for colorbars
mappable = None
diagonal_mappables = []
for j in range(n_lu):
    for i in range(n_lu):
        ax = axs_last[j, i]  # j=row, i=col
        if i == j:
            # Diagonal: plot FATES_PATCHAREA_LU difference (end - start) with diverging colormap
            if vegc_diff_global is not None:
                if vegc_diff_global.shape[0] == n_lu:
                    d = vegc_diff_global[i, :]
                elif vegc_diff_global.shape[1] == n_lu:
                    d = vegc_diff_global[:, i]
                else:
                    d = np.full(len(lon_), np.nan)
            else:
                d = np.full(len(lon_), np.nan)
            if d.shape[0] != len(lon_):
                d = np.full(len(lon_), np.nan)
            sc = ax.scatter(lon_, lat_, c=d, cmap=SHARED_CMAP, s=8, vmin=-shared_absmax, vmax=shared_absmax)
            diagonal_mappables.append(sc)
            ax.set_title(f'Δ VEGC {LU_ABBR[i]}')
        else:
            # Off-diagonal: cumulative transition matrix over full period
            k = i * n_lu + j
            d = sum_pairs_current_grid[k, :]
            if d.shape[0] != len(lon_):
                d = np.full(len(lon_), np.nan)
            sc = ax.scatter(lon_, lat_, c=d, cmap=SHARED_CMAP, s=8, vmin=-shared_absmax, vmax=shared_absmax)
            mappable = sc
            ax.set_title(f'{LU_ABBR[j]} → {LU_ABBR[i]}')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
fig_last.suptitle(f"LUxLU transitions - Cumulative over period", fontsize=24, y=0.96)
output_png = f"{OUTPUT_PATH}/plot_transition_matrix_lulu_last.png"
print(f"Saving static LUxLU plot to {output_png} ...")
# Add global colorbar for static plot (off-diagonal)
if mappable is not None:
    fig_last.colorbar(mappable, cax=cb_ax_last, orientation='vertical', label='Cumulative transition (% over period)')
# Add colorbar for diagonal (patch area) using the first diagonal mappable
if diagonal_mappables:
    fig_last.colorbar(diagonal_mappables[0], cax=cb_ax_patch, orientation='vertical', label='Patch Area (m2/m2)')
fig_last.savefig(output_png, dpi=120)
plt.close(fig_last)
print(f"Saved static LUxLU plot: {output_png}")

# Animation update function
def update(frame):
    file_path, time_idx, time_val = file_time_map[frame]
    print(f"Rendering frame {frame+1}/{n_times}: {file_path}")
    # Extract year and month from file name (e.g., ...h0.1902-01.nc)
    import re
    year_str = month_str = "?"
    match = re.search(r"h0\.(\d{4})-(\d{2})", file_path)
    if match:
        year_str, month_str = match.group(1), match.group(2)
    else:
        # Fallback to time_val logic if not found in filename
        try:
            if hasattr(time_val, 'astype'):
                tval = str(time_val.astype('U'))
            else:
                tval = str(time_val)
            if len(tval) >= 6 and tval[:4].isdigit() and tval[4:6].isdigit():
                year_str = tval[:4]
                month_str = tval[4:6]
            else:
                year_str = tval
        except Exception as e:
            print(f"Warning: Could not parse time_val {time_val}: {e}")
    with xr.open_dataset(file_path, decode_times=False) as ds:
        data = ds[VAR_NAME].isel(time=time_idx).values 
        # Unpack LUxLU if needed
        if data.shape[-1] == 25:
            data = data.reshape(*data.shape[:-1], 5, 5)
        # Get lat/lon for this file (in case of time-varying grid)
        if grid_type == 'ncol':
            lat_ = ds['lat'].values if 'lat' in ds else ds['grid_lat'].values
            lon_ = ds['lon'].values if 'lon' in ds else ds['grid_lon'].values
        elif grid_type == 'lndgrid':
            lat_ = ds['lat'].values
            lon_ = ds['lon'].values
        else:
            lat_ = lat
            lon_ = lon
    # Unpack LUxLU for each grid cell: shape (5, 5, ngrid) or (25, ngrid) or (ngrid, 25)
    ngrid = len(lon_)
    mappable = None
    diagonal_mappables = []
    mappable = None
    for j in range(n_lu):
        for i in range(n_lu):
            ax = axs[j, i]  # j=row, i=col
            ax.clear()
            if i == j:
                # Diagonal: plot FATES_PATCHAREA_LU
                with xr.open_dataset(file_path, decode_times=False) as ds_patch:
                    if 'FATES_PATCHAREA_LU' in ds_patch:
                        patcharea = ds_patch['FATES_PATCHAREA_LU'].isel(time=time_idx).values
                        if patcharea.shape[0] == n_lu:
                            d = patcharea[i, :]
                        elif patcharea.shape[1] == n_lu:
                            d = patcharea[:, i]
                        else:
                            d = np.full(len(lon_), np.nan)
                    else:
                        d = np.full(len(lon_), np.nan)
                if d.shape[0] != len(lon_):
                    d = np.full(len(lon_), np.nan)
                sc = ax.scatter(lon_, lat_, c=d, cmap='plasma', s=8, vmin=patcharea_vmin, vmax=patcharea_vmax)
                diagonal_mappables.append(sc)
                ax.set_title("")  # No label for diagonal
            else:
                # Off-diagonal: plot transition matrix
                if data.shape == (5, 5, ngrid):
                    d = data[i, j, :]
                elif data.shape == (25, ngrid):
                    k = i * n_lu + j
                    d = data[k, :]
                elif data.shape == (ngrid, 25):
                    k = i * n_lu + j
                    d = data[:, k]
                else:
                    d = np.full(len(lon_), np.nan)
                if d.shape[0] != len(lon_):
                    d = np.full(len(lon_), np.nan)
                sc = ax.scatter(lon_, lat_, c=d, cmap=SHARED_CMAP, s=8, vmin=-shared_absmax, vmax=shared_absmax)
                mappable = sc
                ax.set_title(f'{LU_ABBR[j]} → {LU_ABBR[i]}')
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
    # Update the global colorbar for animation (off-diagonal)
    fig.colorbar(mappable, cax=cb_ax, orientation='vertical', label='Transition rate (%/month)')
    # Add colorbar for diagonal (patch area) using the first diagonal mappable
    if hasattr(fig, 'cb_ax_patch'):
        cb_ax_patch = fig.cb_ax_patch
    else:
        cb_ax_patch = fig.add_axes(CBAR_LEFT_POS)
        fig.cb_ax_patch = cb_ax_patch
    if diagonal_mappables:
        fig.colorbar(diagonal_mappables[0], cax=cb_ax_patch, orientation='vertical', label='Patch Area (m2/m2)')
    fig.suptitle(f"Year: {year_str} Month: {month_str}", fontsize=24, y=0.96)
    print(f"  Finished frame {frame+1}/{n_times}")
    return fig.axes

# Create animation
print("Starting animation rendering...")
anim = animation.FuncAnimation(fig, update, frames=n_times, interval=tint, blit=False, repeat=True)

# Save animation
output_file = f"{OUTPUT_PATH}/fates_transition_matrix_lulu_{YEAR_RANGE[0]}-{YEAR_RANGE[1]}.{OUTPUT_FORMAT}"
print(f"Saving animation to {output_file} ...")
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
writer = animation.PillowWriter(fps=1000//tint)
anim.save(output_file, writer=writer, dpi=100)
print(f"Saved animation: {output_file}")
plt.close(fig)
