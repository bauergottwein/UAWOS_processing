# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:03:30 2025

@author: vpk410
"""
import os
import sys
sys.path.append(r'C:\Users\ge86naf\OneDrive\UAWOS\code processing\uawos_processing\UAWOS_processing')
from UAWOS_RSS_flight import UAWOS_RSS_flight

# %%
my_path = r'C:\Users\ge86naf\OneDrive\UAWOS\aferica\data_processing\Oueme_dataset\Doppler_radar_velocimetry_M300_RSS300\Level_1\8'
xsname = '2025-05-09-12-12-50'
Orco_general_attributes = dict()

Orco_general_attributes['xy_EPSG'] = 'EPSG:32631'
Orco_general_attributes['z_EPSG'] = 'EGM08'
Orco_general_attributes['geoid_tiff'] = r'C:\Users\ge86naf\OneDrive\UAWOS\aferica\data_processing\Oueme_dataset\geoid\us_nga_egm2008_sub.tif'
Orco_general_attributes['chainage_field'] = 'distance along reference line'

Orco_1_attributes = Orco_general_attributes.copy()

Orco_1_attributes['name_posfile'] = [os.path.join(my_path, f) for f in os.listdir(my_path) if 'position' in f][0]
Orco_1_attributes['name_RSSfile'] = [os.path.join(my_path, f) for f in os.listdir(my_path) if 'RSS.csv' in f][0]
Orco_1_attributes['name_segyfile'] = [os.path.join(my_path, f) for f in os.listdir(my_path) if 'sgy' in f][0]
Orco_1_attributes['name_wpfile'] = [os.path.join(my_path, f) for f in os.listdir(my_path) if 'waypoints' in f][0]# This will be created during preprocessing, pls see below
Orco_1_attributes['name_ppkfile'] = [os.path.join(my_path, f) for f in os.listdir(my_path) if '.pos' in f][0]# if RTK was off during flight, the path to a corresponding ppk file that contains the relevant time stamp range can be provided here
Orco_1_attributes['in_situ_L3'] = r''
Orco_1_attributes['refline_file'] = [os.path.join(my_path, f) for f in os.listdir(my_path) if 'kml' in f][0]
Orco_1_attributes['RTK_on'] = False

Orco_1_attributes['CL_dist_threshold'] = 100 
Orco_1_attributes['L2_output'] = my_path + os.sep +'Level_2'+ os.sep + xsname+ '.csv'
Orco_1_attributes['L3_output'] = my_path + os.sep +'Level_3'+ os.sep + xsname+ '.csv'


Orco_1 = UAWOS_RSS_flight(Orco_1_attributes)
#%%
Orco_1.visualize_radargram()
#%%
Orco_1 = Orco_1.process_RSS_flight()
#%%
Orco_1.plot_Alt_XScoord_vs_Tnum()
Orco_1.plot_WP_XScoord_vs_Tnum()
#%%
#Orco_1.make_WPfile()
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