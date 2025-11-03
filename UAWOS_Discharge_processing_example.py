# -*- coding: utf-8 -*-
"""

Title: UAWOS_Discharge_processing

Author: Xinqi Hu, Zhen Zhou 11111
Contact: xinqi.hu@tum.de

Description:
    This script script is part of the UAWOS project.
    (Unmanned Airborne Water Observing System, Grant Agreement No: 101081783)
    It is designed to calculate river cross-sectional discharge based on
    pre-processed hydrometric variables derived from drone-borne measurements.
    The required inputs include processed surface flow velocity(Doppler Radar),
    bathymetry (WPR), and water surface elevation (Radar altimetry) and high resolution .
    The script integrates these variables along defined cross-sections to estimate discharge.

Inputs: UAWOS - Level 3 products: velocity, bathymetry, water surface level; Level 4 product: full geometry
Outputs:
    - *_result.csv  
        → Cross-section properties, discharge (Q), and Manning–Strickler coefficients (Ks) for the deterministic (non-ensemble) run.
    - *_ensemble.csv  
        → Ensemble statistics (mean and standard deviation) of discharge and Manning–Strickler coefficients across Monte Carlo samples.
    - *_uncertainty_contribution.csv  
        → Relative contribution of each input variable (e.g., bathymetry, velocity, WSE, WSS) to the total output uncertainty.
    - *-geometry.png  
        → Plot of the cross-section geometry used in the discharge calculation.
    - *-report.html  
        → HTML-formatted summary report including input parameters, computed results, and uncertainty analysis.
    
Created on: 2024-12-13
Last Updated: 2025-07-12

License: CC BY 4.0 (Creative Commons Attribution 4.0 International)

Environment:
    - Developed and tested in: Spyder 5.5.1 (Anaconda, Python 3.12)
    - The script uses '# %%' code cell markers for interactive execution, 
      compatible with Spyder, VSCode, and JupyterLab

Dependencies:
    - Python environment specified in 'uawos.yaml'
    - To create the environment: >>> conda env create -f uawos.yaml >>> conda activate UAWOS

"""

# %%
import os
import sys
sys.path.append(r'h:\UAWOS\uawos_data_processing\revision_11_25')
from UAWOS_Discharge import UAWOS_Discharge

# %% General attributes

general_attributes = dict()
general_attributes['project_path'] = r'h:\UAWOS\uawos_data_processing\revision_11_25\Discharge_processing_example'
general_attributes['XSid'] = 'Po2' # index of calculated cross section
general_attributes['xy_EPSG'] = 'EPSG:3006'
general_attributes['z_EPSG'] = 'EPSG:5613'
general_attributes['dx'] = 1 # Width of water coulmn

# %% Calculation attributes

cal_attributes = general_attributes.copy()
cal_attributes['resultpath'] = general_attributes['project_path'] + os.sep + 'result'
cal_attributes['velocity'] = general_attributes['project_path'] + os.sep + 'XSpo2_forQ.csv'
cal_attributes['bathymetry'] = general_attributes['project_path'] + os.sep + 'XSpo2.csv'
cal_attributes['full_geometry'] =general_attributes['project_path'] + os.sep + 'XSpo2_processed.csv'
cal_attributes['WSE'] = 179.1131     # Water surface elevation (m)
cal_attributes['WSS'] = 129.3703     # Water surface slope (cm/km)
cal_attributes['WSE_sigma'] = 0.03   # Standard deviation of WSE
cal_attributes['WSS_sigma'] = 0.4264 # Standard deviation of WSS
cal_attributes['lbguess'] = -58      # Rough estimate of the land-water left boundary
cal_attributes['rbguess'] = 47       # Rough estimate of the land-water right boundary
cal_attributes['if_calc_uncertainty'] = True # Whether calculate the uncertainty
                                             # related the observation data, if true.
cal_attributes['sample_size'] = 3 # sample size of Monte Carlo simulation

# %%
%matplotlib qt
XS = UAWOS_Discharge(cal_attributes)
aaa = XS.process_Discharge()
