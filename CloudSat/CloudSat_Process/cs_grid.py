# This program is to collect the data into a grid
# Import package
import sys
import numpy as np
import xarray as xr
import joblib as jl
from joblib import Parallel, delayed

def process_file(f, lat_era5, lon_era5, z_data):
    """Process individual CloudSat file and match to ERA5 grid."""
    cs_lat = f["lat"]
    cs_lon = f["lon"]
    cs_hgt = f["hgt"]
    cs_qlw = f["qlw"]
    cs_qsw = f["qsw"]
    cs_idx = np.arange(len(cs_lat), dtype=int)

    # Find nearest ERA5 grid index
    lat_idx = np.searchsorted(lat_era5, cs_lat, side="right") - 1
    lon_idx = np.searchsorted(lon_era5, cs_lon, side="right") - 1

    # Pair up indices and find unique ones
    coord = np.array(list(zip(lat_idx, lon_idx)))
    unique_pairs, inverse_indices = np.unique(coord, axis=0, return_inverse=True)

    # Store indices where each unique (lat_idx, lon_idx) occurs
    pair_indices = {tuple(pair): np.where(inverse_indices == i)[0].tolist() for i, pair in enumerate(unique_pairs)}

    # Store results
    results = []

    for pair, indices in pair_indices.items():

        if len(indices) > 1:
            # Extract ERA5 data at the matched grid cell
            lat_id, lon_id = pair

            # Get mean ERA5 data for the matched points
            era5_values = z_data[162, :, lat_id, lon_id]  # Extract ERA5 at this grid point
            
            
            
            # Store the matched results
            results.append({
                "lat_idx": lat_id,
                "lon_idx": lon_id,
                "cs_indices": indices,
            })
    
    return results

def main():
    # Load ERA5 data
    with xr.open_dataset("/work/b11209013/2024_Research/nstcCCKW/z/z_2006.nc") as f:
        dims = f.coords
        z_data = (f.z / 9.81).values  # Convert to geopotential height

    # Shift coordinates to match CloudSat data positioning
    lat_era5 = np.array(dims["lat"]) - 0.25
    lon_era5 = np.array(dims["lon"]) - 0.3125

    # Load CloudSat data
    cs_data = jl.load("/work/b11209013/2024_Research/CloudSat/CS_sel/CS_2006_163.joblib")

    print("hgt shape:\t", cs_data[0]["hgt"])
    print("qlw shape:\t", cs_data[0]["qlw"].shape) 

    # Process each CloudSat file in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(f, lat_era5, lon_era5, z_data) for f in cs_data
    )

    # Flatten the results
    results = [item for sublist in results for item in sublist]

    print(len(results))

    # Save processed data
    #jl.dump(results, "/work/b11209013/2024_Research/processed_cloudsat_era5.joblib")
    
    print(f"Processed {len(results)} matched CloudSat-ERA5 grid points.")

if __name__ == "__main__":
    main()
