# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:03:30 2025

@author: vpk410
"""
import os
my_path = r'h:\UAWOS\uawos_data_processing' + os.sep + 'LX80_processing_example'

from UAWOS_LX80_flight import UAWOS_LX80_flight

Orco_general_attributes = dict()

Orco_general_attributes['offset_PPK_LX80'] = (450-105)/1000 #from Sune
Orco_general_attributes['offset_RTK_LX80'] = (300-105)/1000 #from Sune
Orco_general_attributes['offset_PPK_NRA24'] = (450-105-195)/1000 #SPH said NRA is 195 mm from LX80
Orco_general_attributes['offset_RTK_NRA24'] = (300-105-195)/1000 #SPH said NRA is 195 mm from LX80
Orco_general_attributes['geoid_tiff'] = my_path + os.sep + 'ITALGEO2005_Orco_negative.tif'
Orco_general_attributes['xy_EPSG'] = 'EPSG:32632'
Orco_general_attributes['z_EPSG'] = 'ITALGEO05'
Orco_general_attributes['range_cal_fac'] = 0.985
Orco_general_attributes['chainage_file'] = my_path + os.sep + 'chain_centerline_orco_utm.shp'
Orco_general_attributes['chainage_field'] = 'cngmeters'

#%%
Orco_cal_attributes = Orco_general_attributes.copy()

Orco_cal_attributes['name_LX80file'] = my_path + os.sep + '2025-04-08-12-35-19-GEOLUXLX80.csv'
Orco_cal_attributes['name_posfile'] = my_path + os.sep + '2025-04-08-12-35-19-position.csv'
Orco_cal_attributes['name_WPfile'] = my_path + os.sep + '2025-04-08-12-35-19-Waypoints.xlsx'
Orco_cal_attributes['name_segyfile'] = my_path + os.sep + '2025-04-08-12-35-19-GEOLUXLX80.sgy'
Orco_cal_attributes['RTK_on'] = True # This flag indicates if RTK was on throughout the flight. 
Orco_cal_attributes['True_WSE_WGS84'] = 235.86

Orco_cal = UAWOS_LX80_flight(Orco_cal_attributes)
Orco_cal.plot_Alt_RTK_vs_sample_no()
Orco_cal.plot_Alt_RTK_minus_range_vs_sample_no(y_lower=235,y_upper=237)
#Orco_cal.create_WP_file(lt_cutoff=5, ut_cutoff=35)
Orco_cal.show_WPfile()
Orco_cal.show_range_corr()
Orco_cal = Orco_cal.calibrate_LX80()
Orco_cal.show_FWF_data()
#%%
Orco_1_attributes = Orco_general_attributes.copy()
Orco_1_attributes['name_posfile'] = my_path + os.sep + '2025-04-08-12-53-22-position.csv'
Orco_1_attributes['name_ppkfile'] = r'' # if RTK was off during flight, the path to a corresponding ppk file that contains the relevant time stamp range can be provided here
Orco_1_attributes['name_LX80file'] = my_path + os.sep + '2025-04-08-12-53-22-GEOLUXLX80.csv'
Orco_1_attributes['L2_output'] = my_path + os.sep + r'Level_2\2024-07-11-16-57-45.csv'
Orco_1_attributes['L3_output'] = my_path + os.sep + r'Level_3\2024-07-11-16-57-45.csv'
Orco_1_attributes['RTK_on'] = True
Orco_1_attributes['ylim'] = [184,191]
Orco_1_attributes['CL_dist_threshold'] = 100 # this is the threshold used to discard points, i.e. any point further than this from the centerline is discarded.

Orco_1 = UAWOS_LX80_flight(Orco_1_attributes)
if not os.path.exists(my_path + os.sep + 'Level_2'):
    os.makedirs(my_path + os.sep + 'Level_2')
if not os.path.exists(my_path + os.sep + 'Level_3'):
    os.makedirs(my_path + os.sep + 'Level_3')
Orco_1 = Orco_1.process_LX80()
Orco_1.plot_LX80_processed()