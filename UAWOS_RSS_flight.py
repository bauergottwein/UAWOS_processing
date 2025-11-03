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
from scipy.optimize import curve_fit
import os
import fiona
from fiona.drvsupport import supported_drivers
fiona.supported_drivers['KML'] = 'rw'
supported_drivers['LIBKML'] = 'rw'
from UAWOS_Utilities import readsegy, skyhubtime, raster_interp, hubdist, peakmodel1, peakmodel2, posinterp
          
class UAWOS_RSS_flight:
    def __init__(self, config_dict):
        print("UAWOS RSS flight class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************
        cfreq = 24E9 #Hz, center frequency of the Doppler radar
        spol = 3E8 # Speed of light in air, m/s
        wlen = spol/cfreq # radar wavelength, m
        print('Wavelength of Doppler radar in m:', wlen)
        dshift = 2/wlen # 1/m, i.e. Hz/(m/s),i.e. Doppler shift per velocity
        print('Doppler shift per unit speed in 1/m or Hz/(m/s): ', dshift)
        self.vres = 7.3921 # mm/s velocity resolution of the Doppler radar, according to Geolux
        specres = specres = self.vres/1000*dshift #spectral resolution of the Doppler radar
        print('Spectral resolution of Doppler spectrum in Hz: ', specres)
        self.angle = 45 #
        # Column names in various input files - adjust if format changes
        # Skyhub files
        self.skyhub_RTK_Alt_name = 'Altitude RTK'
        self.skyhub_RTK_Lat_name = 'Latitude'
        self.skyhub_RTK_Lon_name = 'Longitude'
        self.skyhub_date_name = 'Date'
        self.skyhub_time_name = 'Time' # assumed to be UTC
        self.skyhub_rawvel_item = 'RSS:Velocity raw'
        self.skyhub_filtvel_item = 'RSS:Velocity filtered'
        self.skyhub_tilt_item = 'RSS:Tilt'
        self.skyhub_trace_item = 'RSS:Trace'
        self.skyhub_dir_item = 'RSS:Direction' #1 incoming, -1 outgoing
        self.skyhub_RTKstatus_item = 'RTK Status'
        self.skyhub_RTKstatus_onstring = 'ON'
        # Waypoint file
        self.skyhub_WPcol_name = 'Next WP'
        self.WPfile_WP_item = 'Waypoint'
        self.WPfile_StartTrace_item = 'Start Trace'
        self.WPfile_EndTrace_item = 'End Trace'
        self.WPfile_flightalt_item = 'Flight Altitude (m)'
        self.WPfile_Direction_item = 'Direction'
        self.WPfile_vupper_item = 'v upper (cm/s)'
        self.WPfile_vlower_item = 'v lower (cm/s)'
        self.WPfile_vmask_item = 'v mask (cm/s)'
        self.WPfile_fmodel_item = 'Fitting Model'
        self.WPfile_m1_item = 'm1'
        self.WPfile_m2_item = 'm2'
        self.WPfile_s1_item = 's1'
        self.WPfile_s2_item = 's2'
        self.WPfile_f1_item = 'f1'
        self.WPfile_f2_item = 'f2'
        self.WPfile_rtvel_item = 'Velocity raw cm/s'
        self.WPfile_tstamp_item ='Time stamp'
        self.WPfile_fit_vax_item = 'fit_vax'
        self.WPfile_fit_data_item = 'fit_data'
        self.WPfile_FWvel_item = 'Velocity full waveform data (cm/s)'
        self.WPfile_downwashvel_item = 'Downwash Velocity (cm/s)'
        self.WPfile_popt_item = 'popt'
        self.WPfile_pcov_item = 'pcov'
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
        self.out_time_item = 'Time (UTC)'
        self.out_lon_item = 'Longitude WGS84'
        self.out_lat_item = 'Latitude WGS84'
        self.out_altwgs84_item = 'Altitude WGS84 (m)'
        self.out_altttf_item = 'Altitude TTF (m)'
        self.out_velraw_item = 'Velocity raw (m/s)'
        self.out_velfilt_item = 'Velocity filtered (m/s)'
        self.out_tilt_item = 'Tilt (deg)'
        self.out_trace_item = 'Trace'
        self.out_dir_item = 'Direction' #1 incoming, -1 outgoing
        self.out_WP_item = self.skyhub_WPcol_name
        self.out_geoid_item = self.z_EPSG + ' (m)'
        self.out_alt_item = 'Altitude.' + self.z_EPSG + ' (m)'
        self.out_botelev_item = 'z.' + self.z_EPSG + ' (m)'
        self.out_Easting_item = 'Easting.' + self.xy_EPSG + ' (m)'
        self.out_Northing_item = 'Northing.' + self.xy_EPSG + ' (m)'
        self.out_Chainage_item = 'Cross section coordinate (m)'
        self.out_DistfromCline_item  = 'Distance from reference line (m)'
        # in situ data
        self.isdata_surfvel_item = 'Surface velocity (cm/s)'
        self.isdata_surfvel_label_item = 'OTT MFPro'
        # Miscellaneous
        self.RTK_EPSG = 'EPSG:4326'
        # Plot labels
        self.vel_ax_label = 'Surface velocity, cm/s'
        self.return_power_label = 'Return power'
        self.trace_number_label = 'Trace number'
        # output file
        self.outfile = self.name_RSSfile.split('.')[0] + '_processed.csv'
        
    def visualize_radargram(self):
        data,headers, sample_rate, ntrace, nsamples = readsegy(self.name_segyfile)
        data = np.concatenate((data[2048:,:],data[0:2048,:]))
        vaxis_sort = np.concatenate((np.linspace(-2048,-1,2048)*self.vres, np.linspace(1,2048,2048)*self.vres))/10/ np.cos(self.angle/360*2*np.pi)
        plt.figure()
        plt.pcolor(np.linspace(1,data.shape[1],data.shape[1]),vaxis_sort,data,vmin = 0, vmax = 500)
        plt.ylim([-300, 300])
        plt.xlabel(self.trace_number_label)
        plt.ylabel(self.vel_ax_label)
        plt.show()
    
    def process_RSS_flight(self):
        cl = gpd.read_file(self.refline_file)
        cl = cl.to_crs(crs = self.xy_EPSG)
        dfref = pd.DataFrame(np.array([cl[self.shp_geometry_item].x,cl[self.shp_geometry_item].y,cl[self.chainage_field]]).transpose(), columns = [self.cdf_easting, self.cdf_northing, self.cdf_chainage])
        pos_data = pd.read_csv(self.name_posfile)
        RSS_data = pd.read_csv(self.name_RSSfile)
        if not self.RTK_on:
            n_ppkfile = self.name_ppkfile
            ppk_data = pd.read_csv(n_ppkfile,skiprows = self.ppk_skiprows, delimiter = self.ppk_delimiter)
            RSS_data = posinterp(RSS_data,ppk_data, self.ppk_date_item, self.ppk_time_item, self.ppk_lat_item, self.ppk_lon_item, self.ppk_alt_item, False,
                                  interp_latname = self.skyhub_RTK_Lat_name, interp_lonname = self.skyhub_RTK_Lon_name,
                                  interp_altname = self.skyhub_RTK_Alt_name)
        RSS_time = skyhubtime(RSS_data, self.skyhub_date_name, self.skyhub_time_name)
        pos_time = skyhubtime(pos_data,self.skyhub_date_name, self.skyhub_time_name)
        nwp_interp = np.interp(RSS_time.astype('float'),pos_time.astype('float'),pos_data[self.skyhub_WPcol_name])
        df = pd.DataFrame(np.array([RSS_data[self.skyhub_date_name], RSS_data[self.skyhub_time_name], RSS_data[self.skyhub_RTK_Lon_name], RSS_data[self.skyhub_RTK_Lat_name],
                                    RSS_data[self.skyhub_RTK_Alt_name], 
                                    RSS_data[self.skyhub_rawvel_item], RSS_data[self.skyhub_filtvel_item], RSS_data[self.skyhub_tilt_item], 
                                    RSS_data[self.skyhub_trace_item], RSS_data[self.skyhub_dir_item], nwp_interp]).transpose(), columns = [self.out_date_item,self.out_time_item,
                                    self.out_lon_item, self.out_lat_item, self.out_altwgs84_item,
                                    self.out_velraw_item, self.out_velfilt_item, self.out_tilt_item , self.out_trace_item, self.out_dir_item, self.out_WP_item]) 
        df[self.shp_geometry_item] = df.apply(lambda x: Point((float(x[self.out_lon_item]), float(x[self.out_lat_item]))), axis=1)
        df = gpd.GeoDataFrame(df, geometry=self.shp_geometry_item, crs = self.RTK_EPSG)
        df = df.to_crs(self.xy_EPSG)
        Geoid = raster_interp(df[self.shp_geometry_item].x.values,df[self.shp_geometry_item].y.values,self.geoid_tiff)
        df[self.z_EPSG] = Geoid
        h_geoid = df[self.out_altwgs84_item] - df[self.z_EPSG]
        df[self.out_alt_item] = h_geoid
        df[self.out_Easting_item] = df[self.shp_geometry_item].x
        df[self.out_Northing_item] = df[self.shp_geometry_item].y
        df = hubdist(df,dfref,self.CL_dist_threshold, self.cdf_chainage, Easting = self.out_Easting_item, Northing = self.out_Northing_item,
                     Hubs_out = self.out_Chainage_item, Distance_Hubs_out = self.out_DistfromCline_item, Easting_hubs = self.cdf_easting, 
                     Northing_hubs = self.cdf_northing)
        self.RSS_data = df
        return self
    
    def plot_Alt_XScoord_vs_Tnum(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.RSS_data[self.out_trace_item],self.RSS_data[self.out_alt_item],'r')
        ax1.plot(self.RSS_data[self.out_trace_item],self.RSS_data[self.out_Chainage_item])
        ax1.set_xlabel(self.trace_number_label)
        ax1.set_ylabel(self.out_Chainage_item)
        ax2.set_ylabel(self.out_Chainage_item)

    def plot_WP_XScoord_vs_Tnum(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.RSS_data[self.out_trace_item],self.RSS_data[self.out_WP_item],'r')
        ax1.plot(self.RSS_data[self.out_trace_item],self.RSS_data[self.out_Chainage_item])
        ax1.set_xlabel(self.trace_number_label)
        ax1.set_ylabel(self.out_Chainage_item)
        ax2.set_ylabel(self.out_WP_item)
        
    def make_WPfile(self,**kwargs):
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************
        lt_cutoff_default = 5
        ut_cutoff_default = 10
        flight_alt_default = 4
        fm_default = 'Single Peak'
        m1_default = 50
        m2_default = 50
        s1_default = 10
        s2_default = 10
        f1_default = 1000
        f2_default = 1000
        vel_lower_default = 30
        vel_upper_default = 200
        vel_mask_default = 15
        #*****************************************************************************
        nwp_interp = self.RSS_data[self.skyhub_WPcol_name]
        numwp = int(np.nanmax(nwp_interp))
        wpdict = dict()
        lt_cutoff = kwargs.get('lt_cutoff',lt_cutoff_default)# trace cutoff at lower end of the automatically identified trace window
        ut_cutoff = kwargs.get('ut_cutoff',ut_cutoff_default)# trace cutoff at upper end of the automatically identified trace window 
        flight_alt = kwargs.get('flight_alt',flight_alt_default) # Flight altitude (is 4 m above water in the standard scenario)
        for i in range(1,numwp+1,1):
            wpdict[i] = dict()
            wpdict[i][self.WPfile_WP_item] = i
            wpdict[i][self.WPfile_StartTrace_item] = self.RSS_data[self.out_trace_item][np.where(nwp_interp==i)[0][0]] + lt_cutoff
            wpdict[i][self.WPfile_EndTrace_item] = self.RSS_data[self.out_trace_item][np.where(nwp_interp==i)[0][-1]] - ut_cutoff
            wpdict[i][self.WPfile_flightalt_item] = flight_alt
            values, counts = np.unique(self.RSS_data[self.out_dir_item][np.where(nwp_interp==i)[0]], return_counts = True)
            ind = np.argmax(counts)
            wpdict[i][self.WPfile_Direction_item] = values[ind]
            if wpdict[i][self.WPfile_Direction_item] == 1:
                wpdict[i][self.WPfile_vupper_item] = -vel_lower_default #upper bound of fitting window, can be adjusted per waypoint
                wpdict[i][self.WPfile_vlower_item] = -vel_upper_default #lower bound of fitting window, can be adjusted per waypoint
                wpdict[i][self.WPfile_vmask_item] = -vel_mask_default # Any values inside the blind zone will be ignored
            else:
                wpdict[i][self.WPfile_vupper_item] = vel_upper_default
                wpdict[i][self.WPfile_vlower_item] = vel_lower_default
                wpdict[i][self.WPfile_vmask_item] = vel_mask_default
            wpdict[i][self.WPfile_fmodel_item] = fm_default # Single or Double peak
            # Setting starting parameters for the fitting model
            if wpdict[i][self.WPfile_Direction_item] == 1: 
                wpdict[i][self.WPfile_m1_item] = -m1_default
                wpdict[i][self.WPfile_m2_item] = -m2_default
            else:
                wpdict[i][self.WPfile_m1_item] = m1_default
                wpdict[i][self.WPfile_m2_item] = m2_default
            wpdict[i][self.WPfile_s1_item] = s1_default
            wpdict[i][self.WPfile_s2_item] = s2_default
            wpdict[i][self.WPfile_f1_item] = f1_default
            wpdict[i][self.WPfile_f2_item] = f2_default

        wpframe = pd.DataFrame.from_dict(wpdict).transpose()
        wpframe.to_excel(self.name_wpfile)
        
    def plot_WPdata(self):
        wpdata = pd.read_excel(self.name_wpfile,index_col = 0)
        wpdata[self.out_Chainage_item] = ""
        for w in wpdata.index:
            subflight = self.RSS_data[np.logical_and(self.RSS_data[self.out_trace_item] >
                            wpdata[self.WPfile_StartTrace_item][w], self.RSS_data[self.out_trace_item] < wpdata[self.WPfile_EndTrace_item][w])]
            wpdata.loc[w,self.out_Chainage_item] = subflight[self.out_Chainage_item].values.mean()

        plt.figure()
        plt.plot(self.RSS_data[self.out_trace_item],self.RSS_data[self.out_Chainage_item])
        plt.plot(wpdata[self.WPfile_StartTrace_item],wpdata[self.out_Chainage_item], 'k.',markersize = 10)
        plt.plot(wpdata[self.WPfile_EndTrace_item],wpdata[self.out_Chainage_item], 'g.', markersize = 10)
        plt.xlabel(self.trace_number_label)
        plt.ylabel(self.out_Chainage_item)
        plt.show()
        
    def process_RSS_data(self):
        #*****************************************************************************
        # Adjustable input
        #*****************************************************************************
        vmin_plot = 0
        vmax_plot = 500
        ymin_plot = -200
        ymax_plot = 200
        #*****************************************************************************
        n_segyfile = self.name_segyfile
        data,headers, sample_rate, ntrace, nsamples = readsegy(n_segyfile)
        self.fwf_data =np.concatenate((data[2048:,:],data[0:2048,:]))
        self.vaxis_sort = np.concatenate((np.linspace(-2048,-1,2048)*self.vres, np.linspace(1,2048,2048)*self.vres))/10/ np.cos(self.angle/360*2*np.pi)
        self.waypoints = pd.read_excel(self.name_wpfile)
        self.waypoints[self.WPfile_rtvel_item] = ""
        self.waypoints[self.WPfile_tstamp_item] = ""
        self.waypoints[self.out_Easting_item] = ""
        self.waypoints[self.out_Northing_item] = ""
        self.waypoints[self.out_alt_item] = ""
        self.waypoints[self.WPfile_FWvel_item] = np.nan
        self.waypoints[self.WPfile_downwashvel_item] = np.nan
        self.waypoints[self.WPfile_popt_item] = dict()
        self.waypoints[self.WPfile_pcov_item] = dict()
        self.waypoints[self.WPfile_fit_vax_item] = dict()
        self.waypoints[self.WPfile_fit_data_item] = dict()
        for wp_idx , w in enumerate(self.waypoints.index):
            subflight = self.RSS_data[np.logical_and(self.RSS_data[self.out_trace_item] > self.waypoints[self.WPfile_StartTrace_item][w],
                                                     self.RSS_data[self.out_trace_item] < self.waypoints[self.WPfile_EndTrace_item][w])]
            subdata = self.fwf_data[: , self.waypoints[self.WPfile_StartTrace_item][w] : self.waypoints[self.WPfile_EndTrace_item][w] + 1]
            subdata_m = subdata.mean(1)
            vax = self.vaxis_sort
            vlower = self.waypoints[self.WPfile_vlower_item][w]
            vupper = self.waypoints[self.WPfile_vupper_item][w]
            ind = np.logical_and(vax >vlower,vax <vupper)
            vax = vax[ind]
            subdata_m = subdata_m[ind]
            vmask = self.waypoints[self.WPfile_vmask_item][w]
            ind = np.logical_or(vax <-vmask,vax >vmask)
            vax = vax[ind]
            subdata_m = subdata_m[ind]
            plt.figure()
            plt.title('Waypoint' + str(wp_idx+1))
            plt.pcolor(np.linspace(1,subdata.shape[1],subdata.shape[1]),self.vaxis_sort,subdata,vmin = vmin_plot, vmax = vmax_plot)
            plt.ylim([ymin_plot, ymax_plot])
            plt.xlabel(self.trace_number_label)
            plt.ylabel(self.vel_ax_label)
            self.waypoints.loc[w, self.out_Chainage_item] = subflight[self.out_Chainage_item].values.mean()
            self.waypoints.loc[w, self.out_Easting_item] = subflight[self.out_Easting_item].values.mean()
            self.waypoints.loc[w, self.out_Northing_item] = subflight[self.out_Northing_item].values.mean()
            self.waypoints.loc[w, self.out_alt_item] = subflight[self.out_alt_item].values.mean()
            self.waypoints.loc[w, self.WPfile_rtvel_item] = subflight[self.out_velraw_item].values.mean()*100.
            self.waypoints.loc[w, self.WPfile_tstamp_item] = skyhubtime(subflight,self.out_date_item,self.out_time_item).astype('float').mean().astype('datetime64[us]')
            try:
                if  self.waypoints[self.WPfile_fmodel_item][w] == 'Single Peak':
                    p0 = [self.waypoints[self.WPfile_m1_item][w],self.waypoints[self.WPfile_s1_item][w],self.waypoints[self.WPfile_f1_item][w]]
                    popt,pcov = curve_fit(peakmodel1, vax, subdata_m, p0=p0)#,bounds=flight_dict ['Aux data']['pm2 bounds'])
                elif self.waypoints[self.WPfile_fmodel_item][w]  == 'Double Peak':
                    p0 = [self.waypoints[self.WPfile_m1_item][w],self.waypoints[self.WPfile_m2_item][w],self.waypoints[self.WPfile_s1_item][w],
                          self.waypoints[self.WPfile_s2_item][w],self.waypoints[self.WPfile_f1_item][w],self.waypoints[self.WPfile_f2_item][w]]
                    popt,pcov = curve_fit(peakmodel2, vax, subdata_m, p0=p0)
            except RuntimeError:
                print('an error occured in the fitting')
                plt.plot(vax,subdata_m,'k.')
                print(' Waypoint ' + str(w) +  ' could not be fitted')
                plt.title (' Waypoint ' + str(w))
                plt.show()
                popt = np.nan
                pcov = np.nan
            else:
                if  self.waypoints[self.WPfile_fmodel_item][w] == 'Single Peak':
                    p0 = [self.waypoints[self.WPfile_m1_item][w],self.waypoints[self.WPfile_s1_item][w],self.waypoints[self.WPfile_f1_item][w]]
                    popt,pcov = curve_fit(peakmodel1, vax, subdata_m, p0=p0)#,bounds=flight_dict ['Aux data']['pm2 bounds'])
                    fdata = peakmodel1(vax,popt[0],popt[1],popt[2])
                    self.waypoints.loc[w,self.WPfile_FWvel_item] = popt[0]

                elif self.waypoints[self.WPfile_fmodel_item][w]  == 'Double Peak':
                    p0 = [self.waypoints[self.WPfile_m1_item][w],self.waypoints[self.WPfile_m2_item][w],self.waypoints[self.WPfile_s1_item][w],
                          self.waypoints[self.WPfile_s2_item][w],self.waypoints[self.WPfile_f1_item][w],self.waypoints[self.WPfile_f2_item][w]]
                    popt,pcov = curve_fit(peakmodel2, vax, subdata_m, p0=p0)
                    fdata = peakmodel2(vax,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
                    if self.waypoints[self.WPfile_Direction_item][w] == 1: # upstream looking
                        FW_Velocity_flow =  abs(min(popt[0], popt[1]))
                        FW_Velocity_drone = abs(popt[0] - popt[1])
                    else: # downstream looking
                        FW_Velocity_flow = abs(min(popt[0], popt[1]))
                        FW_Velocity_drone = abs(popt[0] - popt[1])
                    self.waypoints.loc[w,self.WPfile_FWvel_item] = FW_Velocity_flow
                    self.waypoints.loc[w, self.WPfile_downwashvel_item] = FW_Velocity_drone
                plt.figure()
                plt.title('Waypoint' + str(wp_idx+1))
                plt.plot(vax,subdata_m,'k.')
                plt.plot(vax,fdata,'r-')
                plt.xlim(vlower,vupper)
                plt.xlabel(self.vel_ax_label)
                plt.ylabel(self.return_power_label)
                if  self.waypoints[self.WPfile_fmodel_item][w] == 'Double Peak':
                    plt.plot([popt[0],popt[0]],plt.gca().get_ylim(),color = 'r')
                    plt.plot([popt[1],popt[1]],plt.gca().get_ylim(),color = 'b')
                else:
                    plt.plot([popt[0],popt[0]],plt.gca().get_ylim())
                plt.show()
                phi = sum((fdata-subdata_m)**2/len(fdata))**0.5
                print('------------ Waypoint '+str(wp_idx+1)+'------------')
                print('Phi='+str(phi))
                if  self.waypoints[self.WPfile_fmodel_item][w] == 'Double Peak':
                    print('Velocity 1='+str(popt[0]))
                    print('Velocity 2='+str(popt[1]))
                else:
                    print('Vel='+str(popt[0]))           
        return self
    
    def plot_RSS_res(self):
        labels = [];
        if os.path.isfile(self.in_situ_L3):
            is_data = pd.read_csv(self.in_situ_L3)
            l1, = plt.plot(is_data[self.out_Chainage_item],is_data[self.isdata_surfvel_item],'r*',label=self.isdata_surfvel_label_item)
            labels.append(l1)
        l2,= plt.plot(self.waypoints[self.out_Chainage_item],self.waypoints[self.WPfile_FWvel_item],linestyle='',marker='+', color='black',label='RSS')
        labels.append(l2)
        plt.legend(handles = labels)
        plt.xlabel(self.out_Chainage_item)
        plt.ylabel(self.vel_ax_label)
        plt.title(self.name_segyfile.split('\\')[-1] )
        
    def output_generation(self):
        XS_L3 = self.waypoints
        XS_L3=XS_L3.drop(columns=[self.WPfile_StartTrace_item, self.WPfile_EndTrace_item, self.WPfile_m1_item, self.WPfile_m2_item,
                                  self.WPfile_s1_item, self.WPfile_s2_item, self.WPfile_f1_item, self.WPfile_f2_item, self.WPfile_f2_item, self.WPfile_flightalt_item,
                                  self.WPfile_popt_item, self.WPfile_pcov_item, self.WPfile_fit_vax_item, self.WPfile_fit_data_item])
        XS_L3.to_csv(self.outfile )
