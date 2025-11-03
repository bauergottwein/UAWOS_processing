# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:03:30 2025

@author: vpk410
"""
import os
my_path = r'h:\UAWOS\uawos_data_processing' + os.sep + 'ZAero_processing_example'

from UAWOS_ZAero_flight import UAWOS_ZAero_flight

Orco_general_attributes = dict()

Orco_general_attributes['xy_EPSG'] = 'EPSG:32632'
Orco_general_attributes['z_EPSG'] = 'ITALGEO05'
Orco_general_attributes['geoid_tiff'] = my_path + os.sep + 'ITALGEO2005_Orco_negative.tif'
Orco_general_attributes['chainage_field'] = 'cngmeter'

Orco_1_attributes = Orco_general_attributes.copy()

Orco_1_attributes['name_posfile'] = my_path + os.sep + '2025-04-10-13-37-22-position.csv'
Orco_1_attributes['name_segyfile'] = my_path + os.sep + '2025-04-10-13-37-22-gpr.sgy'
Orco_1_attributes['name_secfile'] = my_path + os.sep + '2025-04-10-13-37-22_sections.xlsx'# This will be created during preprocessing, pls see below
Orco_1_attributes['name_ppkfile'] = r''# if RTK was off during flight, the path to a corresponding ppk file that contains the relevant time stamp range can be provided here
Orco_1_attributes['refline_file'] = my_path + os.sep + 'chain_XSline_Cavour_utm.shp'
Orco_1_attributes['in_situ_L3'] = r''
Orco_1_attributes['WSE'] = 176.02
Orco_1_attributes['cfreq'] = 150 # Antenna frequency. ZondAeroLF has 100, 150 and 300 MHz
Orco_1_attributes['dir'] = 'parallel' # Antenna orientation wrt to flow direction. Parallel is default.
Orco_1_attributes['RTK_on'] = True
Orco_1_attributes['gain_power'] = 3 # These are parameters for the radargram visulization
Orco_1_attributes['gain_lt'] = 0.2 # These are parameters for the radargram visulization
Orco_1_attributes['gain_ut'] = 3 # These are parameters for the radargram visulization
Orco_1_attributes['CL_dist_threshold'] = 100 
Orco_1_attributes['gain_power'] = 3 # These are parameters for the radargram visulization
Orco_1_attributes['gain_lt'] = 0.2 # These are parameters for the radargram visulization
Orco_1_attributes['gain_ut'] = 3 # These are parameters for the radargram visulization

Orco_1 = UAWOS_ZAero_flight(Orco_1_attributes)
#%%
Orco_1 = Orco_1.process_WPR_flight()
Orco_1.plot_Tnum_Alt_vs_XScoord()
Orco_1.plot_XScoord_Alt_vs_Tnum()
Orco_1.plot_sections()
Orco_1.find_WSE()
#%%
%matplotlib Qt
Orco_1.process_WPR_data()
#%%
Orco_1.plot_results([-20,20])
#%%
Orco_1.output_generation()