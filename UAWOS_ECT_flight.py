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
    
    def process_ECT(self,**kwargs):
        lat_name = kwargs.get('lat_name','Latitude RTK')
        lon_name = kwargs.get('lon_name','Longitude RTK')
        alt_name = kwargs.get('alt_name','Altitude RTK')
        echo_data = pd.read_csv(self.name_posfile)
        cl = gpd.read_file(self.refline_file)
        cl = cl.to_crs(crs=self.xy_EPSG)
        dfchain = pd.DataFrame(np.array([cl['geometry'].x,cl['geometry'].y,cl[self.chainage_field]]).transpose(), columns = ['Easting', 'Northing', 'XS_coord'])
        if not self.RTK_on:
            n_ppkfile = self.name_ppkfile
            ppk_data = pd.read_csv(self.name_ppkfile,skiprows = 9,delimiter=r"\s+")
            echo_data = posinterp(echo_data,ppk_data,'%','UTC','latitude(deg)','longitude(deg)', 'height(m)', False)
            lat_name = 'Latitude_interpolated'
            lon_name = 'Longitude_interpolated'
            alt_name = 'Altitude_interpolated'
        echo_data = pd.DataFrame(np.array([echo_data['Date'], echo_data['Time'], echo_data[lon_name], echo_data[lat_name], echo_data[alt_name], echo_data['ALT:Altitude'], echo_data['ECHO:True Depth'], echo_data['ECHO:Pitch'], echo_data['ECHO:Roll']]).transpose(), columns = ['Date','Time','Longitude','Latitude','Altitude WGS84','Altitude TTF','Echo Depth','Pitch','Roll'])
        echo_data.loc[echo_data['Echo Depth']==0,'Echo Depth'] = np.nan
        echo_data['geometry'] = echo_data.apply(lambda x: Point((float(x.Longitude), float(x.Latitude))), axis=1)
        echo_data = gpd.GeoDataFrame(echo_data, geometry='geometry', crs = 4326)
        echo_data = echo_data.to_crs(self.xy_EPSG)
        echo_data = echo_data.loc[~np.isnan(echo_data['geometry'].x.values)]
        geoid = raster_interp(echo_data['geometry'].x.values,echo_data['geometry'].y.values,self.geoid_tiff)
        echo_data[self.z_EPSG + ' (m)'] = geoid
        h_geoid = echo_data['Altitude WGS84'] - echo_data[self.z_EPSG + ' (m)']
        echo_data['Altitude.' + self.z_EPSG + ' (m)'] = h_geoid
        bot = self.WSE - echo_data['Echo Depth']
        echo_data['z.' + self.z_EPSG + ' (m)'] = bot
        echo_data['Easting.' + self.xy_EPSG + ' (m)'] = echo_data['geometry'].x
        echo_data['Northing.' + self.xy_EPSG + ' (m)'] = echo_data['geometry'].y
        echo_data = hubdist(echo_data,dfchain,self.CL_dist_threshold,'XS_coord', Easting = 'Easting.' + self.xy_EPSG + ' (m)', Northing = 'Northing.' + self.xy_EPSG + ' (m)')
        self.echo_data = echo_data
        return self
    
    def plot_Alt_Snum_vs_XS_coord(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.echo_data['Hubs'],self.echo_data['Altitude.' + self.z_EPSG + ' (m)'],'r',label='Altitude')
        ax1.plot(self.echo_data['Hubs'],self.echo_data.index, label = 'Trace Number')
        ax1.set_xlabel('Cross section coordinate (m)')
        ax1.set_ylabel('ECT sample number')
        ax2.set_ylabel('Altitude.' + self.z_EPSG + ' (m)')
        
    def plot_XS_coord_Alt_vs_Snum(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.echo_data.index,self.echo_data['Altitude.' + self.z_EPSG + ' (m)'],'r')
        ax1.plot(self.echo_data.index,self.echo_data['Hubs'])
        ax1.set_xlabel('ECT sample number')
        ax1.set_ylabel('Cross section coordinate (m)')
        ax2.set_ylabel('Altitude.' +  self.z_EPSG + ' (m)')
        
    def find_WSE(self):
        offset_TTF0_true_ground = 0.21 #estimated empirically from in-situ WSE
        echo_data = self.echo_data.loc[self.Start_sample:self.End_sample]
        meanalt = echo_data['Altitude.' + self.z_EPSG + ' (m)'].mean()
        print(meanalt)
        meanrange = echo_data['Altitude TTF'].mean()
        print(meanrange)
        print('WSE = ', meanalt-meanrange-offset_TTF0_true_ground)
        
    def plot_XS(self):
        plt.figure()
        plt.plot(self.echo_data['Hubs'],self.echo_data['z.' + self.z_EPSG + ' (m)'],'b.')
        plt.xlabel('Cross section coordinate (m)')
        plt.ylabel('River bottom elevation, m above ' + self.z_EPSG)
        
    def save_L2_L3_data(self):
        L3_data = self.echo_data
        L3_data = L3_data.rename(columns={'Hubs':'Cross section coordinate (m)', 'Distance_Hubs':'Distance from reference line (m)', })
        L3_data = L3_data.drop('geometry', axis=1)
        L3_data.to_csv(self.L3_output,index = False)
        L2_data = L3_data.drop(['Cross section coordinate (m)','Distance from reference line (m)'], axis=1)
        L2_data.to_csv(self.L2_output,index = False)
        
    def plot_ECT_results(self,xlim):
        WSE = self.WSE
        if os.path.isfile(self.name_isfile):
            is_data = pd.read_csv(self.name_isfile)
        L3_data = pd.read_csv(self.L3_output)
        if self.FWF_data:
            L3_data_fwf = pd.read_csv(self.L3_output_FWF)
        plt.figure(figsize=(10,5))
        line1, = plt.plot(L3_data['Cross section coordinate (m)'], L3_data['z.' + self.z_EPSG + ' (m)'],'r.', label='L3 data',markersize=1)
        handles = [line1]
        if self.FWF_data:
            line2, = plt.plot(L3_data_fwf['Cross section coordinate (m)'], L3_data_fwf['z.' + self.z_EPSG  + ' (m)'],'g.', label='L3 data from full waveform',MarkerSize=1)
            handles = [line1, line2]
        if os.path.isfile(self.name_isfile):
            line3, = plt.plot(is_data['Cross section coordinate (m)'],is_data['z.' + self.z_EPSG + ' (m)'],'k.', label='in situ')
            handles = handles.append(line3)
        plt.plot(plt.gca().get_xlim(),[WSE,WSE],'b')
        plt.xlim(xlim)
        plt.xlabel('Cross section coordinate (m)')
        plt.ylabel('z.' + self.z_EPSG + ' (m)')
        plt.legend(handles=handles)
        plt.title(self.name_posfile.split('\\')[-1] )
        