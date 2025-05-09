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
from sklearn.linear_model import LinearRegression
import segyio
from UAWOS_Utilities import skyhubtime, raster_interp, hubdist     

class UAWOS_LX80_flight:
    def __init__(self, config_dict):
        print("UAWOS LX80 flight class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def plot_Alt_RTK_vs_sample_no(self,**kwargs):
        data = pd.read_csv(self.name_LX80file)
        plt.figure()
        plt.plot(data.index, data['Altitude RTK'])
        plt.xlabel('Sample No')
        plt.ylabel('RTK Elevation, m above WGS84')
        plt.show()
        
    def plot_Alt_RTK_minus_range_vs_sample_no(self,**kwargs):
        data = pd.read_csv(self.name_LX80file)
        plt.figure()
        plt.plot(data.index, data['Altitude RTK']-data['LX:Distance'])
        plt.xlabel('Sample No')
        plt.ylabel('RTK Elevation, m above WGS84')
        y_lower = kwargs.get('y_lower',plt.gca().get_ylim()[0])
        y_upper = kwargs.get('y_upper',plt.gca().get_ylim()[1])
        plt.ylim([y_lower,y_upper])
        plt.show()
        
    def create_WP_file(self,lt_cutoff, ut_cutoff,**kwargs):
        data = pd.read_csv(self.name_LX80file)
        LX80_time = skyhubtime(data,'Date','Time')
        pos_data = pd.read_csv(self.name_posfile)
        pos_time = skyhubtime(pos_data,'Date','Time')
        nwp_interp = np.interp(LX80_time.astype('float'),pos_time.astype('float'),pos_data['Next WP'])

        numwp = np.nanmax(nwp_interp).astype('int')
        wpdict = dict()

        for i in range(1,numwp+1,1):
            wpdict[i] = dict()
            wpdict[i]['Waypoint'] = i
            wpdict[i]['Start Sample'] = data.index[np.where(nwp_interp==i)[0][0]] + lt_cutoff
            wpdict[i]['End Sample'] = data.index[np.where(nwp_interp==i)[0][-1]] - ut_cutoff
            wpdict[i]['Height above water RTK (m)'] = data['Altitude RTK'][np.where(nwp_interp==i)[0][lt_cutoff:-ut_cutoff-1]].median()- self.True_WSE_WGS84 - self.offset_RTK_LX80
            wpdict[i]['Height above water LX80 (m)'] = data['LX:Distance'][np.where(nwp_interp==i)[0][lt_cutoff:-ut_cutoff-1]].median()

        wpframe = pd.DataFrame.from_dict(wpdict).transpose()
        wpframe.to_excel(self.name_WPfile)
        
    def show_WPfile(self):
        wpdata = pd.read_excel(self.name_WPfile,index_col = 0)
        data = pd.read_csv(self.name_LX80file)
        plt.figure()
        plt.plot(data.index,data['Altitude RTK'] - self.True_WSE_WGS84 - self.offset_RTK_LX80)
        plt.plot(wpdata['Start Sample'],wpdata['Height above water RTK (m)'], 'k.',markersize = 10)
        plt.plot(wpdata['End Sample'],wpdata['Height above water RTK (m)'], 'g.', markersize = 10)
        plt.xlabel('LX80 Sample Number')
        plt.ylabel('Height above water (m)')
        plt.show()
        
    def show_range_corr(self):
        wpdata = pd.read_excel(self.name_WPfile,index_col = 0)
        plt.figure()
        plt.plot(wpdata['Height above water LX80 (m)'],wpdata['Height above water RTK (m)'],'.')
        plt.plot(plt.gca().get_xlim(),plt.gca().get_xlim(),'-')
        plt.xlabel('Radar range (m)')
        plt.ylabel('RTK range (m)')
        plt.show()
        
    def calibrate_LX80(self):
        wpdata = pd.read_excel(self.name_WPfile,index_col = 0)
        regi = LinearRegression(fit_intercept=False).fit(np.array(wpdata['Height above water LX80 (m)']).reshape(-1, 1), np.array(wpdata['Height above water RTK (m)']).reshape(-1, 1))
        print(regi.coef_)
        scalef = regi.coef_[0]

        plt.figure()
        plt.plot(scalef*wpdata['Height above water LX80 (m)'],wpdata['Height above water RTK (m)'],'.')
        plt.plot(plt.gca().get_xlim(),plt.gca().get_xlim(),'-')
        plt.xlabel('corrected Radar range (m)')
        plt.ylabel('RTK range (m)')
        plt.show()
        self.range_cal_fac = regi.coef_[0][0]
        return self
    
    def process_LX80(self):
        cl = gpd.read_file(self.chainage_file)
        cl = cl.to_crs(crs=self.xy_EPSG)
        dfchain = pd.DataFrame(np.array([cl['geometry'].x,cl['geometry'].y,cl[self.chainage_field]]).transpose(), columns = ['Easting', 'Northing', 'Chainage'])
        data_pos = pd.read_csv(self.name_posfile)
        data_lx80 = pd.read_csv(self.name_LX80file)
        if self.RTK_on:
            data_pos = data_pos[data_pos['RTK Status'] =='ON']
            data_lx80 = data_lx80[data_lx80['LX:Distance']<100]
            data_lx80['geometry'] = data_lx80.apply(lambda x: Point((float(x.Longitude), float(x.Latitude))), axis=1)
            data_lx80 = gpd.GeoDataFrame(data_lx80, geometry='geometry', crs = 4326)
            data_lx80 = data_lx80.to_crs(self.xy_EPSG)
            geoid = raster_interp(data_lx80['geometry'].x.values,data_lx80['geometry'].y.values,self.geoid_tiff)
            data_lx80[self.z_EPSG + ' (m)'] = geoid
            h_geoid = data_lx80['Altitude RTK'] - data_lx80[self.z_EPSG + ' (m)']
            data_lx80['Altitude.' + self.z_EPSG + ' (m)'] = h_geoid
            data_lx80['LX80 calibrated range (m)'] = self.range_cal_fac*data_lx80['LX:Distance']
            wse_LX80 = data_lx80['Altitude.' + self.z_EPSG + ' (m)'] - data_lx80['LX80 calibrated range (m)'] - self.offset_RTK_LX80
            data_lx80['WSE.'  + self.z_EPSG + ' (m)'] = wse_LX80
            data_lx80['Easting.' + self.xy_EPSG + ' (m)'] = data_lx80['geometry'].x
            data_lx80['Northing.' + self.xy_EPSG + ' (m)'] = data_lx80['geometry'].y
            data_lx80 = data_lx80.drop(columns = ['Sensor ID','Latitude','Longitude','Heading','LX:Temperature','LX:SNR','LX:SD','LX:Trace'])
            data_lx80 = data_lx80.drop(columns = ['geometry'])
            data_lx80 = hubdist(data_lx80,dfchain,self.CL_dist_threshold,'Chainage', Easting = 'Easting.' + self.xy_EPSG + ' (m)', Northing = 'Northing.' + self.xy_EPSG + ' (m)')
            data_lx80 = data_lx80.rename(columns={ 'Hubs': 'Chainage (m)', 'Distance_Hubs': 'Distance from centerline (m)'})
            self.processed_data = data_lx80
            self.processed_data.to_csv(self.L3_output,index = False)
            l2out = self.processed_data.drop(columns = ['Chainage (m)', 'Distance from centerline (m)'])
            l2out.to_csv(self.L2_output,index = False)
            return self
        
    def plot_LX80_processed(self):
        plt.figure()
        plt.plot(self.processed_data['Chainage (m)'],self.processed_data['WSE.' + self.z_EPSG + ' (m)'],'k.',markersize = 0.3)
        plt.ylim(self.ylim)
        plt.xlabel('chainage, m')
        plt.ylabel('WSE, m above ' + self.z_EPSG)
        plt.title(self.name_LX80file.split('\\')[-1] )
        plt.show()
       
    def show_FWF_data(self):        
        # Read data
        with segyio.open(self.name_segyfile,strict = False,endian = 'little') as f:
            l = [np.copy(tr) for tr in f.trace]
        ntrace = len(l)
        print('Number of traces in dataset: ', ntrace)
        trax = np.linspace(1,ntrace,ntrace)
        nsamples = len(l[0])
        print('Number of samples in each trace: ',nsamples)
        ll = np.transpose(np.array(l))
        #These calibration factors are from Geolux, see also UAWOS D2.1
        maxrange = 50
        extension = 0.3
        fft_halfsize = 2048
        internal_dist = 0.1744
        corrfac = 1.1469
        rangeax = np.linspace(1,nsamples,nsamples) -1
        rangeax = rangeax * (maxrange + extension)/(fft_halfsize *corrfac) - internal_dist

        # Plot FWF
        XXv,YYv = np.meshgrid(trax,rangeax)
        plt.figure()
        plt.pcolor(XXv,YYv,ll,vmin = 0, vmax = 100)
        plt.ylim(-1,40)
        plt.xlabel('traces')
        plt.ylabel('Range, m')
        plt.show()
        
