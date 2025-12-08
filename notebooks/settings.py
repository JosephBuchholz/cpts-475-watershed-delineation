from pathlib import Path

DATA_DIR = Path("../data")
EXPORT_DIR = Path("../exports")
NORTH_AMERICA_30 = DATA_DIR / "hyd_na_dem_30s.tif"
WASHINGTON_LARGE = DATA_DIR / "washington_large.tif"
WASHINGTON_MEDIUM = DATA_DIR / "washington_medium.tif"
WASHINGTON_SMALL = DATA_DIR / "washington_small.tif"
WASHINGTON_TINY = DATA_DIR / "washington_tiny.tif"

DEBUG_TIN = True

# This is the EPSG code for the UTM zone for the area of interest
UTM_ZONE_OF_INTEREST = "EPSG:26911"