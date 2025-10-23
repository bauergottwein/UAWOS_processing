# -*- coding: utf-8 -*-
"""
Created on Fri May  9 09:41:39 2025

@author: vpk410
"""

import os
my_path = r'h:\UAWOS\uawos_data_processing' + os.sep + 'ECT_processing_example'

from UAWOS_ECT_flight import UAWOS_ECT_flight

Torne_general_attributes = dict()

Torne_general_attributes['Water temperature'] = 15
Torne_general_attributes['Speed of sound in water'] = 1404.3 + 4.7*Torne_general_attributes['Water temperature'] - 0.04*Torne_general_attributes['Water temperature']**2
Torne_general_attributes['xy_EPSG'] = 'EPSG:3006'
Torne_general_attributes['z_EPSG'] = 'EPSG:5613'
Torne_general_attributes['geoid_tiff'] = my_path + os.sep + r'Torne_SWEN17_RH2000_3006.tif'

Torne_1_attributes = Torne_general_attributes.copy()

Torne_1_attributes['name_posfile'] = my_path + os.sep + r'2024-09-07-08-07-46-position.csv'
Torne_1_attributes['name_isfile'] = my_path + os.sep + r''
Torne_1_attributes['refline_file'] = my_path + os.sep + r'chain_XSline_Torne_utm.shp'
Torne_1_attributes['chainage_field'] = 'cngmeters'
Torne_1_attributes['WSE'] = 46.4
Torne_1_attributes['CL_dist_threshold'] = 100
Torne_1_attributes['Manual_pick'] = False
Torne_1_attributes['FWF_data'] = False
Torne_1_attributes['RTK_on'] = True
Torne_1_attributes['Start_sample'] = 1870
Torne_1_attributes['End_sample'] = 9600
Torne_1_attributes['xlim'] = [-300,300]
Torne_1_attributes['L2_output'] = my_path + os.sep + r'Level_2\2024-09-07-08-07-46-position.csv'
Torne_1_attributes['L3_output'] = my_path + os.sep + r'Level_3\2024-09-07-08-07-46-position.csv'

Torne_1 = UAWOS_ECT_flight(Torne_1_attributes)
#%%
Torne_1 = Torne_1.process_ECT()
Torne_1.plot_Alt_Snum_vs_XS_coord()
Torne_1.plot_XS_coord_Alt_vs_Snum()
Torne_1.find_WSE()
Torne_1.plot_XS()
if not os.path.exists(my_path + os.sep + 'Level_2'):
    os.makedirs(my_path + os.sep + 'Level_2')
if not os.path.exists(my_path + os.sep + 'Level_3'):
    os.makedirs(my_path + os.sep + 'Level_3')
Torne_1.save_L2_L3_data()
Torne_1.plot_ECT_results(Torne_1.xlim)