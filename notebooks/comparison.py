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
from matplotlib import colors
import matplotlib as matplot

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# plotting
import seaborn as sns
# for interactions
from mpl_toolkits.mplot3d import Axes3D

import TIN_engine
from TIN_engine import *
from TIN_draw import *
from TIN_drainage import *
from TIN_watershed import *
from TIN import *

# pysheds
from pysheds.grid import Grid

# other
import time
import copy

from pyproj import CRS, Transformer
from shapely.geometry import Polygon

def intersection_area_of_triangle_and_raster_cell(triangle: Triangle, cell_coords, cell_width, cell_height, transformer) -> float:
    # Create polygon for triangle
    tri_pts_2d = np.array([triangle.v1.coord2D(), triangle.v2.coord2D(), triangle.v3.coord2D()])
    tri_pts_transformed = np.array([transformer.transform(pt[0], pt[1]) for pt in tri_pts_2d])
    triangle_polygon = Polygon(tri_pts_transformed)

    # Create polygon for raster cell
    cell_x, cell_y = cell_coords
    cell_polygon = Polygon([
        (cell_x, cell_y),
        (cell_x + cell_width, cell_y),
        (cell_x + cell_width, cell_y + cell_height),
        (cell_x, cell_y + cell_height)
    ])

    # Calculate intersection
    intersection = triangle_polygon.intersection(cell_polygon)

    return intersection.area

def calculate_intersection_area_of_watersheds(tin_watershed: set[Triangle], raster_catch, grid: Grid, transformer) -> float:
    total_intersection_area = 0.0

    # Iterate through raster cells
    for i in range(raster_catch.shape[0]):
        for j in range(raster_catch.shape[1]):
            if raster_catch[i, j]:
                # Get cell coordinates
                x_coord, y_coord = rasterio.transform.xy(
                    transform=grid.affine,
                    rows=i,
                    cols=j,
                    offset='ll' # lower-left corner of the pixel
                )
                bottom_left = transformer.transform(x_coord, y_coord)

                x_right, y_top = rasterio.transform.xy(
                    transform=grid.affine,
                    rows=i,
                    cols=j,
                    offset='ur' # upper-right corner of the pixel
                )
                top_right = transformer.transform(x_right, y_top)

                cell_w = top_right[0] - bottom_left[0]
                cell_h = top_right[1] - bottom_left[1]
                cell_coords = bottom_left

                # Check intersection with each triangle in TIN watershed
                for triangle in tin_watershed:
                    intersection_area = intersection_area_of_triangle_and_raster_cell(triangle, cell_coords, cell_w, cell_h, transformer)
                    total_intersection_area += intersection_area

    return total_intersection_area

def calculate_jaccard_index(tin_watershed: set[Triangle], raster_catch, grid: Grid, transformer) -> float:
    # Calculate areas
    tin_area = calculate_watershed_area(tin_watershed, transformer)
    raster_area = calculate_watershed_area_raster(raster_catch, grid, transformer)

    # Calculate intersection area
    intersection_area = calculate_intersection_area_of_watersheds(tin_watershed, raster_catch, grid, transformer)

    # Calculate union area
    union_area = tin_area + raster_area - intersection_area

    if union_area == 0:
        return 0.0

    jaccard_index = intersection_area / union_area
    return jaccard_index

def calculate_watershed_area_raster(catch, grid: Grid, transformer) -> float:
    extent_point1 = transformer.transform(grid.extent[0], grid.extent[2])
    extent_point2 = transformer.transform(grid.extent[1], grid.extent[3])
    w = extent_point2[0] - extent_point1[0]
    h = extent_point2[1] - extent_point1[1]

    cell_w = w / grid.shape[0]
    cell_h = h / grid.shape[1]
    cell_area = cell_w * cell_h # cell area in square meters
    cell_area 

    raster_num_cells = np.sum(catch)
    raster_area = raster_num_cells * cell_area
    return raster_area

def run_comparisons(data_file: Path, transformer: Transformer, mesh_error_level: int, iterations: int, min_size: int, ax, radius: float = 0.2):
    # ---- TIN ----
    vertices_3d, triangles, xs, ys, zs, zs_scaled  = get_triangles_from_DEM(data_file, mesh_level=mesh_error_level)

    # Get highest point
    max_index = np.nanargmax(zs)
    max_index = np.nanargmin(zs)
    chosen_point = [xs[max_index], ys[max_index], zs[max_index]]

    # Get subset of triangles around highest point
    bounds_around_highest = [chosen_point[0] + radius, chosen_point[1] + radius, chosen_point[0] - radius, chosen_point[1] - radius]

    triangles_subset = get_subset_of_triangles_from_bounds(triangles, bounds_around_highest, xs, ys)

    # Convert to triangle and vertex objects
    triangle_objects, vertices = convert_to_triangle_and_vertex_objects(triangles_subset, xs, ys, zs)
    print(len(triangle_objects), "triangle objects created.")
    print(len(vertices), "vertex objects created.")

    # Preprocessing
    has_flat_triangles(triangle_objects)
    flat = get_flat_triangles(triangle_objects)
    unflaten_triangles(triangle_objects)

    # Drainage network calculation
    drainage_outlet_nodes = create_drainage_network(triangle_objects)
    print(len(drainage_outlet_nodes), "outlet nodes created.")

    watersheds = []
    watershed_pour_points = []

    print(f"Running {iterations} watershed delineations...")

    for i in range(iterations):
        # Delineate watershed from nearest outlet to pour point
        watershed, start_node = delineate_random_watershed(drainage_outlet_nodes)
        if len(watershed) < min_size:
            print("Watershed too small, rerunning iteration...")
            continue

        watersheds.append(watershed)
        watershed_pour_points.append(start_node.point)
    
    iterations = len(watersheds)  # update iterations for any skipped due to size

    # ---- Raster ----
    print("Setting up raster...")

    grid = Grid.from_raster(str(data_file))
    dem = grid.read_raster(str(data_file))

    # Compute flow directions using D8
    dirmap = (7, 8, 1, 2, 3, 4, 5, 6)
    fdir = grid.flowdir(dem, dirmap=dirmap)

    # Compute flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)

    catchments = []

    print("Running raster watershed delineations...")

    for i in range(iterations):
        # Snap pour point to high accumulation cell
        x_snap, y_snap = grid.snap_to_mask(acc > 10, watershed_pour_points[i][0:2])

        # Delineate the catchment
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
                            xytype='coordinate')
        
        catchments.append(catch)
    
    print("Calculating areas and comparing...")
    
    average_difference = 0.0
    squared_residuals = 0.0
    average_jaccard = 0.0
    for i in range(iterations):
        print(f"--- Iteration {i+1} ---")
        # TIN watershed area calculation
        tin_area = calculate_watershed_area(watersheds[i], transformer=transformer)
        tin_area = utils.square_meters_to_square_miles(tin_area)
        print("TIN Watershed Area (square miles):", tin_area)

        # Raster watershed area calculation
        raster_area = calculate_watershed_area_raster(catchments[i], grid, transformer=transformer)
        raster_area = utils.square_meters_to_square_miles(raster_area)
        print("Raster Watershed Area (square miles):", raster_area)

        difference = np.abs(tin_area - raster_area) / raster_area # normalized difference
        print("Difference (square miles):", difference)
        average_difference += difference
        squared_residuals += difference ** 2

        jaccard_index = calculate_jaccard_index(watersheds[i], catchments[i], grid, transformer)
        print("Jaccard Index:", jaccard_index)
        average_jaccard += jaccard_index

        if jaccard_index > 0.0:
            # visualize
            im = ax.imshow(np.where(catchments[i], catchments[i], np.nan), extent=grid.extent,
                        zorder=1, cmap='Greys_r')

            # this shows the pour point
            plt.scatter([x_snap], [y_snap], c='red', s=50, marker='o')

            def draw_node(node: Node, depth, color):
                for upstream_node in node.upstream_nodes:
                    draw_line_points(ax, node.point[0:2], upstream_node.point[0:2], color, linewidth=2)
                    draw_node(upstream_node, depth + 1, color)

            # draw watershed
            for triangle in watersheds[i]:
                draw_triangle_object(ax, triangle, "#00ff0055", filled=True)

    average_difference /= iterations
    print("-----------------------")
    print("Average Difference (square miles):", average_difference)
    print("RMSE (square miles):", np.sqrt(squared_residuals / iterations))
    print("Average Jaccard Index:", average_jaccard / iterations)
    print("Iterations:", iterations)

def run_benchmark(data_file: Path, mesh_error_level: int, radius: float = 0.2, trials: int = 5):
    total_drainage_length = 0.0
    total_total_length = 0.0
    for trial in range(trials):
        print(f"--- Trial {trial + 1} ---")
        start_time = time.time()

        vertices_3d, triangles, xs, ys, zs, zs_scaled  = get_triangles_from_DEM(data_file, mesh_level=mesh_error_level)

        # Get highest point
        max_index = np.nanargmax(zs)
        max_index = np.nanargmin(zs)
        chosen_point = [xs[max_index], ys[max_index], zs[max_index]]

        # Get subset of triangles around highest point
        bounds_around_highest = [chosen_point[0] + radius, chosen_point[1] + radius, chosen_point[0] - radius, chosen_point[1] - radius]

        triangles_subset = get_subset_of_triangles_from_bounds(triangles, bounds_around_highest, xs, ys)

        # Convert to triangle and vertex objects
        triangle_objects, vertices = convert_to_triangle_and_vertex_objects(triangles_subset, xs, ys, zs)
        print(len(triangle_objects), "triangle objects created.")
        print(len(vertices), "vertex objects created.")

        # Preprocessing
        unflaten_triangles(triangle_objects)

        drainage_start_time = time.time()
        # Drainage network calculation
        drainage_outlet_nodes = create_drainage_network(triangle_objects)
        drainage_end_time = time.time()

        end_time = time.time()

        drainage_length = drainage_end_time - drainage_start_time
        total_length = end_time - start_time

        print("Outlet nodes created:", len(drainage_outlet_nodes))

        print("Total Time (s):", total_length)
        print("Drainage Network Time (s):", drainage_length)

        total_drainage_length += drainage_length
        total_total_length += total_length

    average_drainage_time = total_drainage_length / trials
    average_total_time = total_total_length / trials

    print("Benchmark Results:")
    print("Average Total Time (s):", average_total_time)
    print("Average Drainage Network Time (s):", average_drainage_time)
    return average_total_time, average_drainage_time
    