
def delineate_catchment_with_raster(grid, dem, start_point):
    #dirmap = (64, 128, 1, 2, 4, 8, 16, 32) # this is the default value
    dirmap = (7, 8, 1, 2, 3, 4, 5, 6) # new value to make the D8 and Dinf plots the same

    fdir = grid.flowdir(dem, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Snap pour point to high accumulation cell
    x_snap, y_snap = grid.snap_to_mask(acc > 10, start_point)
    # Delineate the catchment
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
                        xytype='coordinate')

    return catch