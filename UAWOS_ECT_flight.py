# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:55:27 2025

@author: vpk410
"""
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import os

from UAWOS_Utilities import raster_interp, hubdist, posinterp    

     
        
class UAWOS_ECT_flight:
    def __init__(self, config_dict):
        print("UAWOS Echologger ECT flight class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************  
        # Column names in various input files - adjust if format changes
        # Skyhub files
        self.skyhub_RTK_Alt_name = 'Altitude RTK'
        self.skyhub_RTK_Lat_name = 'Latitude RTK'
        self.skyhub_RTK_Lon_name = 'Longitude RTK'
        self.skyhub_date_name = 'Date'
        self.skyhub_time_name = 'Time' # assumed to be UTC
        self.skyhub_RTKstatus_item = 'RTK Status'
        self.skyhub_RTKstatus_onstring = 'ON'
        self.skyhub_TTF_item = 'ALT:Altitude'
        self.skyhub_echodepth_item = 'ECHO:True Depth'
        self.skyhub_echopitch_item = 'ECHO:Pitch'
        self.skyhub_echoroll_item = 'ECHO:Roll'
        # shapefiles
        self.shp_geometry_item = 'geometry'
        # chainage df
        self.cdf_easting = 'Easting'
        self.cdf_northing = 'Northing'
        self.cdf_chainage = 'XS_coord'
        # ppk files - these are EMLID studio results
        self.ppk_skiprows = 9
        self.ppk_delimiter = r"\s+"
        self.ppk_date_item = '%'
        self.ppk_time_item = 'UTC'
        self.ppk_lat_item = 'latitude(deg)'
        self.ppk_lon_item = 'longitude(deg)'
        self.ppk_alt_item = 'height(m)'
        # Output column names
        self.out_date_item = 'Date'
        self.out_time_item = 'Time'
        self.out_lon_item = 'Longitude WGS84'
        self.out_lat_item = 'Latitude WGS84'
        self.out_altwgs84_item = 'Altitude WGS84 (m)'
        self.out_altttf_item = 'Altitude TTF (m)'
        self.out_echodepth_item = 'Echo Depth (m)'
        self.out_echopitch_item = 'Echo Pitch (deg)'
        self.out_echoroll_item = 'Echo Roll (deg)'
        self.out_geoid_item = self.z_EPSG + ' (m)'
        self.out_alt_item = 'Altitude.' + self.z_EPSG + ' (m)'
        self.out_botelev_item = 'z.' + self.z_EPSG + ' (m)'
        self.out_Easting_item = 'Easting.' + self.xy_EPSG + ' (m)'
        self.out_Northing_item = 'Northing.' + self.xy_EPSG + ' (m)'
        self.out_Chainage_item = 'Cross section coordinate (m)'
        self.out_DistfromCline_item  = 'Distance from reference line (m)'
        # Miscellaneous
        self.RTK_EPSG = 'EPSG:4326'
    
    def process_ECT(self,**kwargs):
        lat_name = kwargs.get('lat_name', self.skyhub_RTK_Lat_name)
        lon_name = kwargs.get('lon_name', self.skyhub_RTK_Lon_name)
        alt_name = kwargs.get('alt_name', self.skyhub_RTK_Alt_name)
        echo_data = pd.read_csv(self.name_posfile)
        cl = gpd.read_file(self.refline_file)
        cl = cl.to_crs(crs=self.xy_EPSG)
        dfchain = pd.DataFrame(np.array([cl[self.shp_geometry_item].x,cl[self.shp_geometry_item].y,cl[self.chainage_field]]).transpose(), columns = [self.cdf_easting, self.cdf_northing, self.cdf_chainage])
        if not self.RTK_on:
            ppk_data = pd.read_csv(self.name_ppkfile,skiprows = self.ppk_skiprows, delimiter = self.ppk_delimiter)
            echo_data = posinterp(echo_data,ppk_data, self.ppk_date_item, self.ppk_time_item, self.ppk_lat_item, self.ppk_lon_item, self.ppk_alt_item, False,
                                  interp_latname = lat_name, interp_lonname = lon_name,
                                  interp_altname = alt_name)
        echo_data = pd.DataFrame(np.array([echo_data[self.skyhub_date_name], echo_data[self.skyhub_time_name], echo_data[lon_name], echo_data[lat_name], echo_data[alt_name], 
                                           echo_data[self.skyhub_TTF_item], echo_data[self.skyhub_echodepth_item], echo_data[self.skyhub_echopitch_item], 
                                           echo_data[self.skyhub_echoroll_item]]).transpose(), columns = [self.out_date_item, self.out_time_item,
                                           self.out_lon_item, self.out_lat_item, self.out_altwgs84_item,self.out_altttf_item, self.out_echodepth_item, self.out_echopitch_item,
                                           self.out_echoroll_item])
        echo_data.loc[echo_data[self.out_echodepth_item]==0,self.out_echodepth_item] = np.nan
        echo_data[self.shp_geometry_item] = echo_data.apply(lambda x: Point((float(x[self.out_lon_item]), float(x[self.out_lat_item]))), axis=1)
        echo_data = gpd.GeoDataFrame(echo_data, geometry = self.shp_geometry_item, crs = self.RTK_EPSG)
        echo_data = echo_data.to_crs(self.xy_EPSG)
        echo_data = echo_data.loc[~np.isnan(echo_data[self.shp_geometry_item].x.values)]
        geoid = raster_interp(echo_data[self.shp_geometry_item].x.values,echo_data[self.shp_geometry_item].y.values,self.geoid_tiff)
        echo_data[self.out_geoid_item] = geoid
        h_geoid = echo_data[self.out_altwgs84_item] - echo_data[self.out_geoid_item]
        echo_data[self.out_alt_item] = h_geoid
        bot = self.WSE - echo_data[self.out_echodepth_item]
        echo_data[self.out_botelev_item] = bot
        echo_data[self.out_Easting_item] = echo_data[self.shp_geometry_item].x
        echo_data[self.out_Northing_item] = echo_data[self.shp_geometry_item].y
        echo_data = hubdist(echo_data,dfchain,self.CL_dist_threshold,self.cdf_chainage, Easting = self.out_Easting_item, Northing = self.out_Northing_item,
                            Hubs_out = self.out_Chainage_item, Distance_Hubs_out = self.out_DistfromCline_item, Easting_hubs = self.cdf_easting, 
                            Northing_hubs = self.cdf_northing)
        self.echo_data = echo_data
        return self
    
    def plot_Alt_Snum_vs_XS_coord(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.echo_data[self.out_Chainage_item],self.echo_data[self.out_alt_item],'r',label='Altitude')
        ax1.plot(self.echo_data[self.out_Chainage_item],self.echo_data.index, label = 'Trace Number')
        ax1.set_xlabel(self.out_Chainage_item)
        ax1.set_ylabel('ECT sample number')
        ax2.set_ylabel(self.out_alt_item)
        
    def plot_XS_coord_Alt_vs_Snum(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.echo_data.index,self.echo_data[self.out_alt_item],'r')
        ax1.plot(self.echo_data.index,self.echo_data[self.out_Chainage_item])
        ax1.set_xlabel('ECT sample number')
        ax1.set_ylabel(self.out_Chainage_item)
        ax2.set_ylabel(self.out_alt_item)
        
    def find_WSE(self):
        offset_TTF0_true_ground = 0.21 #estimated empirically from in-situ WSE
        echo_data = self.echo_data.loc[self.Start_sample:self.End_sample]
        meanalt = echo_data[self.out_alt_item].mean()
        print(meanalt)
        meanrange = echo_data[self.out_altttf_item].mean()
        print(meanrange)
        print('WSE = ', meanalt-meanrange-offset_TTF0_true_ground)
        
    def plot_XS(self):
        plt.figure()
        plt.plot(self.echo_data[self.out_Chainage_item],self.echo_data[self.out_botelev_item],'b.')
        plt.xlabel(self.out_Chainage_item)
        plt.ylabel('River bottom elevation, m above ' + self.z_EPSG)
        
    def save_L2_L3_data(self):
        L3_data = self.echo_data
        L3_data = L3_data.drop(self.shp_geometry_item, axis=1)
        L3_data.to_csv(self.L3_output,index = False)
        L2_data = L3_data.drop([self.out_Chainage_item,self.out_DistfromCline_item], axis=1)
        L2_data.to_csv(self.L2_output,index = False)
        
    def plot_ECT_results(self,xlim):
        WSE = self.WSE
        if os.path.isfile(self.name_isfile):
            is_data = pd.read_csv(self.name_isfile)
        L3_data = pd.read_csv(self.L3_output)
        plt.figure(figsize=(10,5))
        line1, = plt.plot(L3_data[self.out_Chainage_item], L3_data[self.out_botelev_item],'r.', label='L3 data',markersize=1)
        handles = [line1]
        if os.path.isfile(self.name_isfile):
            line3, = plt.plot(is_data[self.out_Chainage_item],is_data[self.out_botelev_item],'k.', label='in situ')
            handles = handles.append(line3)
        plt.plot(plt.gca().get_xlim(),[WSE,WSE],'b')
        plt.xlim(xlim)
        plt.xlabel(self.out_Chainage_item)
        plt.ylabel(self.out_botelev_item)
        plt.legend(handles=handles)
        plt.title(self.name_posfile.split('\\')[-1] )
        