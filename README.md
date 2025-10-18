# cpts-475-watershed-delineation
A research project aimed at creating a scalable terrain graph representation for watershed delineation and flow routing.

## Setup Instructions
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1` for Windows PowerShell
    - Windows cmd: `venv/Scripts/activate.bat`
    - Linux or macOS: `soruce venv/bin/activate`
3. `pip install -r packages.txt`
4. Create a `data` folder and add any nessesary data files (see the Google Drive)
5. Run `jupyter notebook` or use the VS Code IDE

## After Setup
After setup just run the virtual environment (step 2) and start jupyter notebooks

## Adding Packages
Run if you install a new package with `pip`: `pip freeze > packages.txt`