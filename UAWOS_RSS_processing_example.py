# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:03:30 2025

@author: vpk410
"""
import os
my_path = r'h:\UAWOS\uawos_data_processing' + os.sep + 'RSS_processing_example'

from UAWOS_RSS_flight import UAWOS_RSS_flight

Orco_general_attributes = dict()

Orco_general_attributes['xy_EPSG'] = 'EPSG:32632'
Orco_general_attributes['z_EPSG'] = 'ITALGEO05'
Orco_general_attributes['geoid_tiff'] = my_path + os.sep + 'ITALGEO2005_Orco_negative.tif'
Orco_general_attributes['chainage_field'] = 'cngmeter'

Orco_1_attributes = Orco_general_attributes.copy()

Orco_1_attributes['name_posfile'] = my_path + os.sep + '2025-04-10-13-54-21-position.csv'
Orco_1_attributes['name_RSSfile'] = my_path + os.sep + '2025-04-10-13-54-21-GEOLUXRSS.csv'
Orco_1_attributes['name_segyfile'] = my_path + os.sep + '2025-04-10-13-54-21-GEOLUXRSS.sgy'
Orco_1_attributes['name_wpfile'] = my_path + os.sep + '2025-04-10-13-54-21-waypoints.xlsx'# This will be created during preprocessing, pls see below
Orco_1_attributes['name_ppkfile'] = r''# if RTK was off during flight, the path to a corresponding ppk file that contains the relevant time stamp range can be provided here
Orco_1_attributes['in_situ_L3'] = r''
Orco_1_attributes['refline_file'] = my_path + os.sep + 'chain_XSline_Cavour_utm.shp'
Orco_1_attributes['RTK_on'] = True
Orco_1_attributes['CL_dist_threshold'] = 100 
Orco_1_attributes['L2_output'] = my_path + os.sep + r'Level_2\2025-04-10-13-37-22.csv'
Orco_1_attributes['L3_output'] = my_path + os.sep + r'Level_3\2025-04-10-13-37-22.csv'


Orco_1 = UAWOS_RSS_flight(Orco_1_attributes)
#%%
Orco_1.visualize_radargram()
#%%
Orco_1 = Orco_1.process_RSS_flight()
#%%
%matplotlib qt
Orco_1.plot_Alt_XScoord_vs_Tnum()
Orco_1.plot_WP_XScoord_vs_Tnum()
#%%
%matplotlib inline
Orco_1.make_WPfile()
Orco_1.plot_WPdata()
#%%
Orco_1.process_RSS_data()
#%%
Orco_1.plot_RSS_res()
#%%
if not os.path.exists(my_path + os.sep + 'Level_2'):
    os.makedirs(my_path + os.sep + 'Level_2')
if not os.path.exists(my_path + os.sep + 'Level_3'):
    os.makedirs(my_path + os.sep + 'Level_3')
Orco_1.L2_L3_generation()