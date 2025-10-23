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
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import os
from UAWOS_Utilities import posinterp, raster_interp, hubdist, skyhubtime, readsegy 
        
class UAWOS_ZAero_flight:
    def __init__(self, config_dict):
        print("UAWOS Zond Areo LF flight class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************
        eps = 81 #relative permittivity of water
        self.vwat = 3E8/eps**0.5
        self.offset_TTF0_true_ground = 0.21 #estimated empirically from in-situ WSE
        # Column names in various input files - adjust if format changes
        # Skyhub files
        self.skyhub_RTK_Alt_name = 'Altitude RTK'
        self.skyhub_RTK_Lat_name = 'Latitude RTK'
        self.skyhub_RTK_Lon_name = 'Longitude RTK'
        self.skyhub_date_name = 'Date'
        self.skyhub_time_name = 'Time' # assumed to be UTC
        self.skyhub_trace_item = 'zGPR:Trace'
        self.skyhub_alt_ttf_item = 'ALT:Altitude'
        self.skyhub_RTKstatus_item = 'RTK Status'
        self.skyhub_RTKstatus_onstring = 'ON'
        # Sections file
        self.secfile_sec_item = 'Section'
        self.secfile_start_trace_item = 'Start Trace'
        self.secfile_end_trace_item = 'End Trace'
        # shapefiles
        self.shp_geometry_item = 'geometry'
        # chainage df
        self.cdf_easting = 'Easting'
        self.cdf_northing = 'Northing'
        self.cdf_chainage = 'XS_coord'
        # ppk files - these are for EMLID studio results
        self.ppk_skiprows = 9
        self.ppk_delimiter = r"\s+"
        self.ppk_date_item = '%'
        self.ppk_time_item = 'UTC'
        self.ppk_lat_item = 'latitude(deg)'
        self.ppk_lon_item = 'longitude(deg)'
        self.ppk_alt_item = 'height(m)'
        # Output column names
        self.out_date_item = 'Date'
        self.out_time_item = 'Time (UTC)'
        self.out_trace_item = 'Trace'
        self.out_lon_item = 'Longitude WGS84'
        self.out_lat_item = 'Latitude WGS84'
        self.out_altwgs84_item = 'Altitude WGS84 (m)'
        self.out_altttf_item = 'Altitude TTF (m)'
        self.out_freq_item = 'Frequency (MHz)'
        self.out_antdir_item = 'Antenna orientation'
        self.out_geoid_item = self.z_EPSG + ' (m)'
        self.out_alt_item = 'Altitude.' + self.z_EPSG + ' (m)'
        self.out_botelev_item = 'z.' + self.z_EPSG + ' (m)'
        self.out_Easting_item = 'Easting.' + self.xy_EPSG + ' (m)'
        self.out_Northing_item = 'Northing.' + self.xy_EPSG + ' (m)'
        self.out_Chainage_item = 'Cross section coordinate (m)'
        self.out_WSE_item = 'WSE.' + self.z_EPSG + ' (m)'
        self.out_DistfromCline_item  = 'Distance from reference line (m)'
        # in situ data
        self.isdata_botel_item = 'Elevation'
        # Miscellaneous
        self.RTK_EPSG = 'EPSG:4326'
        # Plot labels
        self.range_ax_label = 'zGPR range, m'
        self.range_ax_time_label = 'zGPR range, bins'
        self.trace_number_label = 'Trace number'
        
            
    def process_WPR_flight(self,**kwargs):
        geoid_exists = kwargs.get('geoid_exists',True)
        cl = gpd.read_file(self.refline_file)
        cl = cl.to_crs(crs = self.xy_EPSG)
        dfref = pd.DataFrame(np.array([cl[self.shp_geometry_item].x,cl[self.shp_geometry_item].y,cl[self.chainage_field]]).transpose(),
                             columns = [self.cdf_easting, self.cdf_northing, self.cdf_chainage])
        n_posfile = self.name_posfile
        pos_data = pd.read_csv(n_posfile)
        pos_data = pos_data.loc[~np.isnan(pos_data[self.skyhub_trace_item])]
        if not self.RTK_on:
            n_ppkfile = self.name_ppkfile
            ppk_data = pd.read_csv(n_ppkfile,skiprows = self.ppk_skiprows,delimiter=self.ppk_delimiter)
            pos_data = posinterp(pos_data,ppk_data,self.ppk_date_item, self.ppk_time_item, self.ppk_lat_item, self.ppk_lon_item, self.ppk_alt_item, False,
                                  interp_latname = self.skyhub_RTK_Lat_name, interp_lonname = self.skyhub_RTK_Lon_name,
                                  interp_altname = self.skyhub_RTK_Alt_name)
        df = pd.DataFrame(np.array([pos_data[self.skyhub_date_name], pos_data[self.skyhub_time_name], pos_data[self.skyhub_RTK_Lon_name],
            pos_data[self.skyhub_RTK_Lat_name], pos_data[self.skyhub_RTK_Alt_name], pos_data[self.skyhub_alt_ttf_item],
            pos_data[self.skyhub_trace_item]]).transpose(),
            columns = [self.out_date_item,self.out_time_item,self.out_lon_item,self.out_lat_item,self.out_altwgs84_item,
            self.out_altttf_item,self.out_trace_item]) 
        df[self.shp_geometry_item] = df.apply(lambda x: Point((float(x[self.out_lon_item]), float(x[self.out_lat_item]))), axis=1)
        df = gpd.GeoDataFrame(df, geometry=self.shp_geometry_item, crs = self.RTK_EPSG)
        df = df.to_crs(self.xy_EPSG)
        if geoid_exists:
            Geoid = raster_interp(df[self.shp_geometry_item].x.values,df[self.shp_geometry_item].y.values,self.geoid_tiff)
            df[self.out_geoid_item] = Geoid
            h_geoid = df[self.out_altwgs84_item] - df[self.out_geoid_item]
            df[self.out_alt_item] = h_geoid
        df[self.out_Easting_item] = df[self.shp_geometry_item].x
        df[self.out_Northing_item] = df[self.shp_geometry_item].y
        df = hubdist(df,dfref,self.CL_dist_threshold,self.cdf_chainage,Easting = self.out_Easting_item, Northing = self.out_Northing_item,
                     Hubs_out = self.out_Chainage_item, Distance_Hubs_out = self.out_DistfromCline_item, Easting_hubs = self.cdf_easting, 
                     Northing_hubs = self.cdf_northing)
        self.pos_data = df
        return self
    
    def plot_Tnum_Alt_vs_XScoord(self,**kwargs):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        pos_data = self.pos_data
        Alt_field = kwargs.get('Alt_field', self.out_alt_item)
        ax2.plot(pos_data[self.out_Chainage_item],pos_data[Alt_field],'r',label= Alt_field)
        ax1.plot(pos_data[self.out_Chainage_item],pos_data[self.out_trace_item], label = self.trace_number_label)
        ax1.set_xlabel(self.out_Chainage_item)
        ax1.set_ylabel(self.trace_number_label)
        ax2.set_ylabel(self.out_alt_item)
        
    def plot_XScoord_Alt_vs_Tnum(self,**kwargs):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        pos_data = self.pos_data
        Alt_field = kwargs.get('Alt_field',self.out_alt_item)
        ax2.plot(pos_data[self.out_trace_item],pos_data[Alt_field],'r')
        ax1.plot(pos_data[self.out_trace_item],pos_data[self.out_Chainage_item])
        ax1.set_xlabel(self.trace_number_label)
        ax1.set_ylabel(self.out_Chainage_item)
        ax2.set_ylabel(self.out_alt_item)
        
    def plot_sections(self):
        sections = pd.read_excel(self.name_secfile,index_col = self.secfile_sec_item)
        pos_data = self.pos_data
        plt.figure()
        for s in sections.index:
                plt.plot(pos_data[self.out_trace_item],pos_data[self.out_Chainage_item],'b')
                plt.plot([sections[self.secfile_start_trace_item][s],sections[self.secfile_start_trace_item][s]],plt.gca().get_ylim(), 'k',markersize = 10)
                plt.plot([sections[self.secfile_end_trace_item][s],sections[self.secfile_end_trace_item][s]],plt.gca().get_ylim(), 'g',markersize = 10)
        plt.xlabel(self.trace_number_label)
        plt.ylabel(self.out_Chainage_item)
        plt.show()
        
    def find_WSE(self):
        sections = pd.read_excel(self.name_secfile,index_col = self.secfile_sec_item)
        pos_data = self.pos_data
        for s in sections.index:
            startt = sections[self.secfile_start_trace_item][s]
            endt = sections[self.secfile_end_trace_item][s]
            meanalt = pos_data[self.out_alt_item][startt:endt].median(skipna = True)
            print('Mean altitude, section ', s, meanalt)
            meanrange = pos_data[self.out_altttf_item][startt:endt].median(skipna = True)
            print('Mean range, section ', s, meanrange)
            print('WSE section ', s, meanalt-meanrange-self.offset_TTF0_true_ground)
            
    def process_WPR_data(self):
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************
        fontsize_plot = 18
        sigma_gauss = 2.
        color_lower_percentile = 0.05
        color_upper_percentile = 99.9
        plot_trace_margin = 10
        #*****************************************************************************
        sections = pd.read_excel(self.name_secfile,index_col = self.secfile_sec_item)
        pos_data = self.pos_data
        data,headers, sample_rate, ntrace, nsamples = readsegy(self.name_segyfile)
        ttorange = sample_rate*1E-9*self.vwat/2 #sampling rate in ns
        self.automatic_bathymetry_points = dict()
        for s in sections.index:
            starti = pos_data.loc[pos_data[self.out_trace_item] == sections[self.secfile_start_trace_item][s]].index -1
            endi = pos_data.loc[pos_data[self.out_trace_item] == sections[self.secfile_end_trace_item][s]].index +1
            indeces = range(starti[0],endi[0])
            spos_data = pos_data.iloc[np.array(indeces)]
            stime = skyhubtime(spos_data,self.out_date_item,self.out_time_item).astype('float').astype('datetime64[us]')
            feast = interpolate.interp1d(spos_data[self.out_Chainage_item],spos_data[self.out_Easting_item])
            fnorth = interpolate.interp1d(spos_data[self.out_Chainage_item],spos_data[self.out_Northing_item])
            falt = interpolate.interp1d(spos_data[self.out_Chainage_item],spos_data[self.out_alt_item])
            fts = interpolate.interp1d(spos_data[self.out_Chainage_item],stime)
            subdata = data[:,sections[self.secfile_start_trace_item][s]:sections[self.secfile_end_trace_item][s]+1]
            #Background removal
            mdata = subdata.mean(1)
            mmdata = np.tile(mdata,(subdata.shape[1],1)).transpose()
            ndata = subdata-mmdata
            #surface identificatation
            otrace = np.abs(ndata).mean(1)
            surf = np.where(otrace==max(otrace))[0][0]*ttorange
            # Linear gain
            traceindex = np.empty(subdata.shape[1])
            distindex = np.empty(subdata.shape[1])
            for i in range(subdata.shape[1]):
                traceindex[i] = sections[self.secfile_start_trace_item][s] + i
                distindex[i] = pos_data.loc[pos_data[self.out_trace_item] == sections[self.secfile_start_trace_item][s]+ i][self.out_Chainage_item].values
            rangeax = np.linspace(1,nsamples,nsamples)*ttorange
            gain = ((rangeax-self.gain_lt)/(self.gain_ut-self.gain_lt))**self.gain_power
            gain[rangeax<self.gain_lt]=0
            gain[rangeax>self.gain_ut]=1
            mgain = np.tile(gain,(subdata.shape[1],1)).transpose()
            gndata = ndata*mgain
            XXv, YYv = np.meshgrid(distindex, rangeax)
            # Pick some points for constructing a filter, which was used to filter out signals with shorter time (values =0)
            print('Please pick a rough lower bound on the bathymetry range by right-clicking your mouse, and clicking the middle for the ending:')
            # Figure 1
            plt.figure()
            idx = np.argmax(gndata, axis=0)
            plt.pcolor(gndata, vmin=np.percentile(gndata, color_lower_percentile),vmax=np.percentile(gndata, color_upper_percentile))
            plt.plot(idx, 'r.')
            plt.xlim(-plot_trace_margin , gndata.shape[1]+plot_trace_margin)
            plt.xlabel(self.trace_number_label, fontsize=fontsize_plot)
            plt.ylabel( self.range_ax_time_label, fontsize=fontsize_plot)
            plt.title(self.name_segyfile.split('\\')[-1] )
            points_prof = plt.ginput(-1, timeout=-1, show_clicks=True,mouse_add=3, mouse_pop=1, mouse_stop=2)
            points_prof = np.array(points_prof)
            points_prof[0, 0] = 0
            points_prof[-1, 0] = gndata.shape[1]
            f = interpolate.interp1d(points_prof[:, 0], points_prof[:, 1])
            trax = np.linspace(0, gndata.shape[1]-1, gndata.shape[1])
            rax = np.linspace(0, gndata.shape[0]-1, gndata.shape[0])
            thresh_range = f(trax)
            plt.plot(trax, thresh_range, 'w.')
            plt.tick_params(labelsize=fontsize_plot)
            plt.draw() 
            plt.pause(0.01)
            tthresh_range = np.tile(thresh_range, (subdata.shape[0], 1))
            thrx, thry = np.meshgrid(trax, rax)
            tgndata = gndata*(thry > tthresh_range)
            # To construct a Gaussian lowpass filter
            filtedData = gaussian_filter(tgndata, sigma=sigma_gauss)
            tgndata=filtedData
            # Figure 2 (after filtering by a lowpass filter)
            plt.figure()
            plt.pcolor( tgndata, vmin=np.percentile(tgndata, color_lower_percentile),vmax=np.percentile(tgndata, color_upper_percentile))
            plt.xlabel(self.trace_number_label, fontsize=fontsize_plot)
            plt.ylabel( self.range_ax_time_label, fontsize=fontsize_plot)
            plt.title(self.name_segyfile.split('\\')[-1] )
            plt.tick_params(labelsize=fontsize_plot)
            plt.xlabel(self.trace_number_label, fontsize=fontsize_plot)
            plt.ylabel( self.range_ax_time_label, fontsize=fontsize_plot)
            # Automatic picking max value of each trace
            gndata = filtedData
            idx = np.argmax(gndata, axis=0)
            m = len(gndata[1])
            points_prof = np.zeros((m, 2))
            points_prof = np.array(points_prof)
            num_1 = 0
            for ii in range(0, m):
                points_prof[ii, 0] = num_1
                points_prof[ii, 1] = idx[ii]
                num_1 = num_1+1
            points_prof[:, 1] = np.around(points_prof[:, 1], 4)
            plt.plot(points_prof[:, 0], points_prof[:, 1], 'r.')
            # manual updating by picking
            plt.xlim(-plot_trace_margin , gndata.shape[1]+plot_trace_margin)
            print('Please manual updating by right-clicking your mouse, and clicking the middle for the ending:')
            points_surf = plt.ginput(-1, timeout=-1, show_clicks=True,mouse_add=3, mouse_pop=1, mouse_stop=2)
            points_manu = np.array(points_surf)
            plt.plot(points_manu[:, 0], points_manu[:, 1], 'w*')
            for idx in range(points_manu.shape[0]):
                new_idx = int(np.around(points_manu[idx, 0]))
                points_prof[new_idx, 1] = points_manu[idx, 1]
            plt.draw() 
            plt.pause(0.01)
            # Figure 3 for showing updated results
            plt.figure() 
            plt.pcolor(XXv, YYv, gndata, vmin=np.percentile(gndata, color_lower_percentile), vmax=np.percentile(gndata, color_upper_percentile))
            plt.plot(distindex, points_prof[:, 1]*ttorange, 'r.')
            plt.xlabel(self.out_Chainage_item, fontsize=fontsize_plot)
            plt.ylabel(self.range_ax_label, fontsize=fontsize_plot)
            plt.title(self.name_segyfile.split('\\')[-1] )
            plt.tick_params(labelsize=fontsize_plot)
            plt.draw() 
            plt.pause(0.01)
            # Caculating the bathymetry
            points_prof[:, 1] = points_prof[:, 1]*ttorange-surf
            wse = self.WSE
            points_prof[:, 1] = wse - points_prof[:, 1]
            dfbathyauto = pd.DataFrame({self.out_Chainage_item: distindex, self.out_botelev_item: points_prof[:, 1]})
            dfbathyauto[self.out_Easting_item] = feast(dfbathyauto[self.out_Chainage_item])
            dfbathyauto[self.out_Northing_item ] = fnorth(dfbathyauto[self.out_Chainage_item])
            dfbathyauto[self.out_alt_item] = falt(dfbathyauto[self.out_Chainage_item])
            dfbathyauto[self.out_time_item] = fts(dfbathyauto[self.out_Chainage_item]).astype('float').astype('datetime64[us]')
            self.automatic_bathymetry_points[s] = dfbathyauto
        return self
            
    def plot_results(self, xlim):
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************
        fontsize_plot = 18
        #*****************************************************************************
        plt.figure()
        WSE = self.WSE
        if os.path.isfile(self.in_situ_L3):
            is_data = pd.read_csv(self.in_situ_L3)
        labels = [];
        XS_L3_auto =  self.automatic_bathymetry_points[1].head(0) 
        for s in  self.automatic_bathymetry_points.keys():
            XS_L3_auto = np.append(XS_L3_auto,  self.automatic_bathymetry_points[s], axis=0)
        
        if XS_L3_auto.size !=0:
            l3,= plt.plot(XS_L3_auto[:,0],XS_L3_auto[:,1],'r.', markersize = 3,label='Zond Aero automatic')
            labels.append(l3)  
        if os.path.isfile(self.in_situ_L3):
            l1, = plt.plot(is_data[self.out_Chainage_item],is_data[self.isdata_botel_item],'k*',label='in situ')
            labels.append(l1)
        
        plt.plot(plt.gca().get_xlim(),[WSE,WSE],'b')
        plt.xlim(xlim)
        plt.xlabel(self.out_Chainage_item,fontsize=fontsize_plot)
        plt.ylabel(self.out_botelev_item,fontsize=fontsize_plot)
        plt.title(self.name_segyfile.split('\\')[-1] )
        plt.tick_params(labelsize=fontsize_plot)
        plt.legend(handles = labels,fontsize=fontsize_plot-3)
        
    def L2_L3_generation(self):
        XS_L3_auto = self.automatic_bathymetry_points[1].head(0)
        XS_L3_auto[self.out_freq_item] = None
        XS_L3_auto[self.out_antdir_item] = None
        XS_L3_auto[self.out_WSE_item] = None
        for s in self.automatic_bathymetry_points:
            df = self.automatic_bathymetry_points[s]
            df[self.out_freq_item] = self.cfreq
            df[self.out_antdir_item] = self.dir
            df[self.out_WSE_item] = self.WSE
            XS_L3_auto = pd.concat([XS_L3_auto, df], axis=0)
        XS_L3_auto = XS_L3_auto[XS_L3_auto[self.out_botelev_item].notna()]
        XS_L3_auto.to_csv(self.L3_output, index = False)
        XS_L2_auto=XS_L3_auto.drop(columns=[self.out_Chainage_item])
        XS_L2_auto.to_csv(self.L2_output, index = False )
        
    def field_test(self,starttrace, endtrace,**kwargs):
        gain_lt = kwargs.get('gain_lt',1)
        gain_ut = kwargs.get('gain_ut',3)
        gain_power = kwargs.get('gain_power',2)
        data,headers, sample_rate, ntrace, nsamples = readsegy(self.name_segyfile)
        ttorange = sample_rate*1E-9*self.vwat/2 #sampling rate in ns
        subdata = data[:,starttrace:endtrace+1]
        rangeax = np.linspace(1,nsamples,nsamples)*ttorange
        trax = np.linspace(0,subdata.shape[1]-1,subdata.shape[1])
        mdata = subdata.mean(1)
        mmdata = np.tile(mdata,(subdata.shape[1],1)).transpose()
        ndata = subdata-mmdata
        gain = ((rangeax-gain_lt)/(gain_ut-gain_lt))**gain_power
        gain[rangeax<gain_lt]=0
        gain[rangeax>gain_ut]=1
        mgain = np.tile(gain,(subdata.shape[1],1)).transpose()
        gndata = ndata*mgain
        XXv,YYv = np.meshgrid(trax,rangeax)
        plt.figure()
        plt.pcolor(XXv,YYv,gndata,vmin=np.percentile(gndata,0.1),vmax=np.percentile(gndata,99.9))
        plt.xlabel(self.trace_number_label)
        plt.ylabel(self.range_ax_label)
        plt.title(self.name_segyfile.split('\\')[-1] )
        plt.show()
        
