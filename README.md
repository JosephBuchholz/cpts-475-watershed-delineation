# cpts-475-watershed-delineation
A research project aimed at creating a scalable terrain graph representation for watershed delineation and flow routing.

## Setup Instructions
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1` for Windows PowerShell
    - Windows cmd: `venv/Scripts/activate.bat`
    - Linux or macOS: `soruce venv/bin/activate`
3. Run `pip install -r packages.txt` to install the packages
4. Create a `data` folder and add the DEM files you want to analyze
5. In `notebooks/settings.py`, change the `UTM_ZONE_OF_INTEREST` to use the EPSG code corresponding to your DEM's UTM zone (otherwise area calculations will be off).
6. Also, create an `exports` folder in the root directory for output files.
7. Run `jupyter notebook` or use the VS Code IDE

## After Setup
After everything is setup, to run anything just run the virtual environment (step 2) and start jupyter notebooks.

Try out `notebooks/comparison.ipynb`.

## Adding Packages
Run if you install a new package with `pip`: `pip freeze > packages.txt`

## Contributing
When making changes to a juypter notebook file: before making a commit, clear all the outputs
