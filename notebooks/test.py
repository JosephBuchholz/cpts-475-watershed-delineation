# Libraries

import numpy as np
from skimage.transform import resize
import pymartini
import rasterio
import pyvista as pv
from scipy.ndimage import map_coordinates
import settings
import utils

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
# for interactions
from mpl_toolkits.mplot3d import Axes3D

import TIN_engine
from TIN_engine import *
from TIN_draw import *
from TIN_drainage import *

dem_path = settings.WASHINGTON_SMALL
output_vtp = settings.DATA_DIR / "hyd_na_dem_30s_mesh_3d_corrected3.vtp"

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

# resize
dem_resized = resize(dem_clipped, (target_size, target_size), preserve_range=True, anti_aliasing=True)
dem_resized = np.ascontiguousarray(dem_resized.astype(np.float32))

# Build Martini mesh
grid_size = dem_resized.shape[0]
martini = pymartini.Martini(grid_size)
tile = martini.create_tile(dem_resized)

# Extract mesh at desired level
level = 10
vertices, triangles = tile.get_mesh(level)
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

mesh = get_mesh_from_triangles(triangles, vertices_3d)


# get highest point
max_index = np.nanargmax(zs)
highest_point = [xs[max_index], ys[max_index], zs[max_index]]
highest_point


#triangles_subset = get_subset_of_triangles_from_bounds(triangles, [-116.5, 49.34, -116.52, 49.32], xs, ys)
radius = 0.015
bounds_around_highest = [highest_point[0] + radius, highest_point[1] + radius, highest_point[0] - radius, highest_point[1] - radius]

triangles_subset = get_subset_of_triangles_from_bounds(triangles, bounds_around_highest, xs, ys)


triangle_objects, vertices = convert_to_triangle_and_vertex_objects(triangles_subset, xs, ys, zs)
print(len(triangle_objects), "triangle objects created.")
print(len(vertices), "vertex objects created.")


has_flat_triangles(triangle_objects)
flat = get_flat_triangles(triangle_objects)
len(flat)
unflaten_triangles(triangle_objects)


#fig = plt.figure()
#ax = fig.add_subplot(111)
#for triangle in triangles_subset:
#    draw_triangle(ax, triangle, "#00000055", xs, ys)

fig = plt.figure()
ax = fig.add_subplot(111)

# ---- Drainage network calculation ----
drainage_outlet_nodes = create_drainage_network(ax, triangle_objects)

print(len(drainage_outlet_nodes), "outlet nodes created.")

#plt.show()


"""
ids = [596,597]
vert_ids = [18399, 18361]
colors = ["#ff0000aa", "#00ff00aa"]

fig = plt.figure()
ax = fig.add_subplot(111)
for triangle in triangle_objects:
    draw_triangle_object(ax, triangle, "#00000055")

    ax.text(triangle.get_centroid().x, triangle.get_centroid().y, str(triangle.id), color="blue", fontsize=6)

    if triangle.id in ids:
        color = colors[ids.index(triangle.id)]
        draw_triangle_object(ax, triangle, color)
    
    for vertex in triangle.vertices():
        if vertex.id in vert_ids:
            draw_vertex(ax, vertex, markersize=8)

draw_point(ax, [-116.56631, 49.48797], markersize=5)
draw_point(ax, [-116.5684, 49.489098], markersize=5)
draw_point(ax, [-116.56729, 49.48888], markersize=5)
draw_point(ax, [-116.56598, 49.489582], markersize=5)

plt.show()
"""