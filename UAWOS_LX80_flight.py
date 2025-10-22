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
from UAWOS_Utilities import skyhubtime, raster_interp, hubdist, posinterp     

class UAWOS_LX80_flight:
    def __init__(self, config_dict):
        print("UAWOS LX80 flight class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************  
        #These calibration factors are from Geolux, see also UAWOS D2.1
        self.maxrange = 50
        self.extension = 0.3
        self.fft_halfsize = 2048
        self.internal_dist = 0.1744
        self.corrfac = 1.1469
        # Column names in various input files - adjust if format changes
        # Skyhub files
        self.skyhub_RTK_Alt_name = 'Altitude RTK'
        self.skyhub_RTK_Lat_name = 'Latitude RTK'
        self.skyhub_RTK_Lon_name = 'Longitude RTK'
        self.skyhub_LX_distance_name = 'LX:Distance'
        self.skyhub_date_name = 'Date'
        self.skyhub_time_name = 'Time' # assumed to be UTC
        self.skyhub_RTKstatus_item = 'RTK Status'
        self.skyhub_RTKstatus_onstring = 'ON'
        # Waypoint file
        self.skyhub_WPcol_name = 'Next WP'
        self.WPfile_WP_item = 'Waypoint'
        self.WPfile_StartSample_item = 'Start Sample'
        self.WPfile_EndSample_item = 'End Sample'
        self.WPfile_HaboveWater_RTK_item = 'Height above water RTK (m)'
        self.WPfile_HaboveWater_LX80_item = 'Height above water LX80 (m)'
        # shapefiles
        self.shp_geometry_item = 'geometry'
        # chainage df
        self.cdf_easting = 'Easting'
        self.cdf_northing = 'Northing'
        self.cdf_chainage = 'Chainage'
        # ppk files - these are EMLID studio results
        self.ppk_skiprows = 9
        self.ppk_delimiter = r"\s+"
        self.ppk_date_item = '%'
        self.ppk_time_item = 'UTC'
        self.ppk_lat_item = 'latitude(deg)'
        self.ppk_lon_item = 'longitude(deg)'
        self.ppk_alt_item = 'height(m)'
        # Names for interpolated coordinates
        self.lat_interp = 'Latitude_interpolated'
        self.lon_interp = 'Longitude_interpolated'
        self.alt_interp = 'Altitude_interpolated'
        # Output column names
        self.out_geoid_item = self.z_EPSG + ' (m)'
        self.out_alt_item = 'Altitude.' + self.z_EPSG + ' (m)'
        self.out_calrange_item = 'LX80 calibrated range (m)'
        self.out_WSE_item = 'WSE.'  + self.z_EPSG + ' (m)'
        self.out_Easting_item = 'Easting.' + self.xy_EPSG + ' (m)'
        self.out_Northing_item = 'Northing.' + self.xy_EPSG + ' (m)'
        self.out_Chainage_item = 'Chainage (m)'
        self.out_DistfromCline_item  = 'Distance from centerline (m)'
        # Miscellaneous
        self.RTK_EPSG = 'EPSG:4326'
    
    def plot_Alt_RTK_vs_sample_no(self,**kwargs):
        data = pd.read_csv(self.name_LX80file)
        plt.figure()
        plt.plot(data.index, data[self.skyhub_RTK_Alt_name])
        plt.xlabel('Sample No')
        plt.ylabel('RTK Elevation, m above WGS84')
        plt.show()
        
    def plot_Alt_RTK_minus_range_vs_sample_no(self,**kwargs):
        data = pd.read_csv(self.name_LX80file)
        plt.figure()
        plt.plot(data.index, data[self.skyhub_RTK_Alt_name]-data[self.skyhub_LX_distance_name])
        plt.xlabel('Sample No')
        plt.ylabel('RTK Elevation, m above WGS84')
        y_lower = kwargs.get('y_lower',plt.gca().get_ylim()[0])
        y_upper = kwargs.get('y_upper',plt.gca().get_ylim()[1])
        plt.ylim([y_lower,y_upper])
        plt.show()
        
    def create_WP_file(self,lt_cutoff, ut_cutoff,**kwargs):
        data = pd.read_csv(self.name_LX80file)
        LX80_time = skyhubtime(data,self.skyhub_date_name,self.skyhub_time_name)
        pos_data = pd.read_csv(self.name_posfile)
        pos_time = skyhubtime(pos_data,self.skyhub_date_name,self.skyhub_time_name)
        nwp_interp = np.interp(LX80_time.astype('float'),pos_time.astype('float'),pos_data[self.skyhub_WPcol_name])

        numwp = np.nanmax(nwp_interp).astype('int')
        wpdict = dict()

        for i in range(1,numwp+1,1):
            wpdict[i] = dict()
            wpdict[i][self.WPfile_WP_item] = i
            wpdict[i][self.WPfile_StartSample_item] = data.index[np.where(nwp_interp==i)[0][0]] + lt_cutoff
            wpdict[i][self.WPfile_EndSample_item] = data.index[np.where(nwp_interp==i)[0][-1]] - ut_cutoff
            wpdict[i][self.WPfile_HaboveWater_RTK_item] = data[self.skyhub_RTK_Alt_name][np.where(nwp_interp==i)[0][
                lt_cutoff:-ut_cutoff-1]].median()- self.True_WSE_WGS84 - self.offset_RTK_LX80
            wpdict[i][self.WPfile_HaboveWater_LX80_item] = data[self.skyhub_LX_distance_name][np.where(nwp_interp==i)[0][lt_cutoff:-ut_cutoff-1]].median()

        wpframe = pd.DataFrame.from_dict(wpdict).transpose()
        wpframe.to_excel(self.name_WPfile)
        
    def show_WPfile(self):
        wpdata = pd.read_excel(self.name_WPfile,index_col = 0)
        data = pd.read_csv(self.name_LX80file)
        plt.figure()
        plt.plot(data.index,data[self.skyhub_RTK_Alt_name] - self.True_WSE_WGS84 - self.offset_RTK_LX80)
        plt.plot(wpdata[self.WPfile_StartSample_item],wpdata[self.WPfile_HaboveWater_RTK_item], 'k.',markersize = 10)
        plt.plot(wpdata[self.WPfile_EndSample_item],wpdata[self.WPfile_HaboveWater_RTK_item], 'g.', markersize = 10)
        plt.xlabel('LX80 Sample Number')
        plt.ylabel('Height above water (m)')
        plt.show()
        
    def show_range_corr(self):
        wpdata = pd.read_excel(self.name_WPfile,index_col = 0)
        plt.figure()
        plt.plot(wpdata[self.WPfile_HaboveWater_LX80_item],wpdata[self.WPfile_HaboveWater_RTK_item],'.')
        plt.plot(plt.gca().get_xlim(),plt.gca().get_xlim(),'-')
        plt.xlabel('Radar range (m)')
        plt.ylabel('RTK range (m)')
        plt.show()
        
    def calibrate_LX80(self):
        wpdata = pd.read_excel(self.name_WPfile,index_col = 0)
        regi = LinearRegression(fit_intercept=False).fit(np.array(wpdata[self.WPfile_HaboveWater_LX80_item]).reshape(-1, 1), np.array(
            wpdata[self.WPfile_HaboveWater_RTK_item]).reshape(-1, 1))
        print(regi.coef_)
        scalef = regi.coef_[0]

        plt.figure()
        plt.plot(scalef*wpdata[self.WPfile_HaboveWater_LX80_item],wpdata[self.WPfile_HaboveWater_RTK_item],'.')
        plt.plot(plt.gca().get_xlim(),plt.gca().get_xlim(),'-')
        plt.xlabel('corrected Radar range (m)')
        plt.ylabel('RTK range (m)')
        plt.show()
        self.range_cal_fac = regi.coef_[0][0]
        return self
    
    def process_LX80(self):
        #*****************************************************************************
        # Adjustable input
        #***************************************************************************** 
        range_upper_bound = 100.
        dropcol_list = ['Sensor ID','Latitude','Longitude','Heading','LX:Temperature','LX:SNR','LX:SD','LX:Trace'] # columns to be dropped from final output
        #***************************************************************************** 
        cl = gpd.read_file(self.chainage_file)
        cl = cl.to_crs(crs=self.xy_EPSG)
        dfchain = pd.DataFrame(np.array([cl[self.shp_geometry_item].x,cl[self.shp_geometry_item].y,cl[self.chainage_field]]).transpose(), columns = [
            self.cdf_easting, self.cdf_northing, self.cdf_chainage])
        data_pos = pd.read_csv(self.name_posfile)
        data_lx80 = pd.read_csv(self.name_LX80file)
        data_pos = data_pos[data_pos[self.skyhub_RTKstatus_item] == self.skyhub_RTKstatus_onstring]
        data_lx80 = data_lx80[data_lx80[self.skyhub_LX_distance_name] < range_upper_bound]
        if self.RTK_on:
            data_lx80 = posinterp(data_lx80, data_pos,self.skyhub_date_name, self.skyhub_time_name, self.skyhub_RTK_Lat_name, 
                                  self.skyhub_RTK_Lon_name, self.skyhub_RTK_Alt_name, False, interp_latname = self.lat_interp, interp_lonname = self.lon_interp,
                                  interp_altname = self.alt_interp)
        else:
            ppk_data = pd.read_csv(self.name_ppkfile, skiprows = self.ppk_skiprows, delimiter = self.ppk_delimiter)
            data_lx80 = posinterp(data_lx80, ppk_data, self.ppk_date_item, self.ppk_time_item, self.ppk_lat_item, self.ppk_lon_item,
                                  self.ppk_alt_item, False, interp_latname = self.lat_interp, interp_lonname = self.lon_interp,
                                  interp_altname = self.alt_interp)
        
        data_lx80[self.shp_geometry_item] = data_lx80.apply(lambda x: Point((float(x[self.lon_interp]), float(x[self.lat_interp]))), axis=1)
        data_lx80 = gpd.GeoDataFrame(data_lx80, geometry=self.shp_geometry_item, crs = self.RTK_EPSG)
        data_lx80 = data_lx80.to_crs(self.xy_EPSG)
        geoid = raster_interp(data_lx80[self.shp_geometry_item].x.values,data_lx80[self.shp_geometry_item].y.values,self.geoid_tiff)
        data_lx80[self.out_geoid_item] = geoid
        h_geoid = data_lx80[self.alt_interp] - data_lx80[self.out_geoid_item]
        data_lx80[self.out_alt_item] = h_geoid
        data_lx80[self.out_calrange_item] = self.range_cal_fac*data_lx80[self.skyhub_LX_distance_name]
        wse_LX80 = data_lx80[self.out_alt_item] - data_lx80[self.out_calrange_item] - self.offset_RTK_LX80
        data_lx80[self.out_WSE_item] = wse_LX80
        data_lx80[self.out_Easting_item] = data_lx80[self.shp_geometry_item].x
        data_lx80[self.out_Northing_item] = data_lx80[self.shp_geometry_item].y
        data_lx80 = data_lx80.drop(columns = dropcol_list)
        data_lx80 = data_lx80.drop(columns = [self.shp_geometry_item])
        data_lx80 = hubdist(data_lx80,dfchain,self.CL_dist_threshold, self.cdf_chainage, Easting = self.out_Easting_item, Northing = self.out_Northing_item,
                            Hubs_out = self.out_Chainage_item, Distance_Hubs_out = self.out_DistfromCline_item, Easting_hubs = self.cdf_easting, 
                            Northing_hubs = self.cdf_northing)
        self.processed_data = data_lx80
        self.processed_data.to_csv(self.L3_output,index = False)
        l2out = self.processed_data.drop(columns = [self.out_Chainage_item, self.out_DistfromCline_item])
        l2out.to_csv(self.L2_output,index = False)
        return self
        
    def plot_LX80_processed(self):
        plt.figure()
        plt.plot(self.processed_data[self.out_Chainage_item],self.processed_data[self.out_WSE_item],'k.',markersize = 0.3)
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
        rangeax = np.linspace(1,nsamples,nsamples) -1
        rangeax = rangeax * (self.maxrange + self.extension)/(self.fft_halfsize *self.corrfac) - self.internal_dist

        # Plot FWF
        XXv,YYv = np.meshgrid(trax,rangeax)
        plt.figure()
        plt.pcolor(XXv,YYv,ll,vmin = 0, vmax = 100)
        plt.ylim(-1,40)
        plt.xlabel('traces')
        plt.ylabel('Range, m')
        plt.show()
        
