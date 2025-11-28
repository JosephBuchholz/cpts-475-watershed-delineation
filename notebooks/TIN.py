
import rasterio
from pathlib import Path
import numpy as np
import pymartini
from skimage.transform import resize
from scipy.ndimage import map_coordinates

def get_triangles_from_DEM(dem_path: Path, mesh_level=10):
    # Load DEM and metadata
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata_val = src.nodata

    # Handle NoData values: mask and fill with NaN
    dem_masked = dem.astype(np.float32)
    if nodata_val is not None:
        dem_masked[dem_masked == nodata_val] = np.nan

    # Optionally clip extreme elevation values but ignore NaNs
    valid_vals = dem_masked[~np.isnan(dem_masked)]
    lower_clip = np.percentile(valid_vals, 1)
    upper_clip = np.percentile(valid_vals, 99)
    dem_clipped = np.clip(dem_masked, lower_clip, upper_clip)

    # --- Resize DEM safely

    # we need a 2^k + 1 sized grid for the Martini algorithm to work
    max_size = 1025
    dem_size = dem_clipped.shape[0]
    target_size = min(max_size, 2 ** int(np.floor(np.log2(dem_size - 1))) + 1)

    # resize (commented out: resizing like this causes it not to map to real world coordinates anymore)
    #dem_resized = resize(dem_clipped, (target_size, target_size), preserve_range=True, anti_aliasing=True)
    #dem_resized = np.ascontiguousarray(dem_resized.astype(np.float32))

    print("Size: ", dem_size, "->", target_size, "for Martini mesh generation")

    # crop to target size
    dem_cropped = dem_clipped[:target_size, :target_size]
    dem_resized = np.ascontiguousarray(dem_cropped.astype(np.float32))

    # Build Martini mesh
    grid_size = dem_resized.shape[0]
    martini = pymartini.Martini(grid_size)
    tile = martini.create_tile(dem_resized)

    # Extract mesh at desired level
    vertices, triangles = tile.get_mesh(mesh_level)
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 2) # put into 2 columns
    triangles = np.array(triangles, dtype=np.int32).reshape(-1, 3) # put into 3 columns

    rows = vertices[:, 0]
    cols = vertices[:, 1]

    # Map grid indices to real-world coordinates (x, y)
    rows_int = np.clip(np.round(rows).astype(int), 0, dem_resized.shape[0] - 1)
    cols_int = np.clip(np.round(cols).astype(int), 0, dem_resized.shape[1] - 1)
    xs, ys = rasterio.transform.xy(transform, rows_int, cols_int)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # Bilinear interpolate elevation for fractional vertices (rows, cols)
    coords = np.vstack([rows, cols])
    zs = map_coordinates(dem_resized, coords, order=1, mode='nearest')

    # Fix NaN values by replacing with nearby valid elevations (note: this does not seem to be needed)
    nan_mask = np.isnan(zs)
    if np.any(nan_mask):
        # Replace NaNs by nearest valid value (simple approach)
        zs[nan_mask] = np.nanmean(zs[~nan_mask])

    # Normalize elevation (between 0 and 1) ignoring NaNs
    zs_min = np.nanmin(zs)
    zs_max = np.nanmax(zs)
    zs = (zs - zs_min) / (zs_max - zs_min)

    # Fix vertical exaggeration
    z_scale = 0.01
    zs_scaled = zs * z_scale

    # Stack into final vertices (X, Y, Z)
    vertices_3d = np.column_stack([xs, ys, zs_scaled])

    # Optional: center horizontally for better visualization
    #vertices_3d[:, 0] -= vertices_3d[:, 0].mean()
    #vertices_3d[:, 1] -= vertices_3d[:, 1].mean()

    return vertices_3d, triangles, xs, ys, zs, zs_scaled