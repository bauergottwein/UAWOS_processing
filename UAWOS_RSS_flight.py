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
        
    def visualize_radargram(self):
        data,headers, sample_rate, ntrace, nsamples = readsegy(self.name_segyfile)
        data = np.concatenate((data[2048:,:],data[0:2048,:]))
        vaxis_sort = np.concatenate((np.linspace(-2048,-1,2048)*self.vres, np.linspace(1,2048,2048)*self.vres))/10/ np.cos(self.angle/360*2*np.pi)
        plt.figure()
        plt.pcolor(np.linspace(1,data.shape[1],data.shape[1]),vaxis_sort,data,vmin = 0, vmax = 500)
        plt.ylim([-300, 300])
        plt.xlabel('Traces')
        plt.ylabel('Flow velocity, cm/s')
        plt.show()
    
    def process_RSS_flight(self):
        cl = gpd.read_file(self.refline_file)
        cl = cl.to_crs(crs = self.xy_EPSG)
        dfref = pd.DataFrame(np.array([cl['geometry'].x,cl['geometry'].y,cl[self.chainage_field]]).transpose(), columns = ['Easting', 'Northing', 'Chainage'])
        lat_name = 'Latitude'
        lon_name = 'Longitude'
        alt_name = 'Altitude RTK'
        pos_data = pd.read_csv(self.name_posfile)
        RSS_data = pd.read_csv(self.name_RSSfile)
        if not self.RTK_on:
            n_ppkfile = self.name_ppkfile
            ppk_data = pd.read_csv(n_ppkfile,skiprows = self.ppk_skiprows,delimiter=r"\s+")
            RSS_data = posinterp(RSS_data,ppk_data,'%','UTC','latitude(deg)','longitude(deg)', 'height(m)', False)
            lat_name = 'Latitude_interpolated'
            lon_name = 'Longitude_interpolated'
            alt_name = 'Altitude_interpolated'
        RSS_time = skyhubtime(RSS_data,'Date','Time')
        pos_time = skyhubtime(pos_data,'Date','Time')
        nwp_interp = np.interp(RSS_time.astype('float'),pos_time.astype('float'),pos_data['Next WP'])
        df = pd.DataFrame(np.array([RSS_data['Date'], RSS_data['Time'], RSS_data[lon_name], RSS_data[lat_name], RSS_data[alt_name], 
                                    RSS_data['RSS:Velocity raw'], RSS_data['RSS:Velocity filtered'], RSS_data['RSS:Tilt'], 
                                    RSS_data['RSS:Trace'], RSS_data['RSS:Direction'], nwp_interp]).transpose(), columns = ['Date','Time','Longitude',
                                    'Latitude','Altitude WGS84','Velocity raw', 'Velocity filtered', 'Tilt', 'Trace', 'Direction', 'Next_WP']) 
        df['geometry'] = df.apply(lambda x: Point((float(x.Longitude), float(x.Latitude))), axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs = 4326)
        df = df.to_crs(self.xy_EPSG)
        Geoid = raster_interp(df['geometry'].x.values,df['geometry'].y.values,self.geoid_tiff)
        df[self.z_EPSG] = Geoid
        h_geoid = df['Altitude WGS84'] - df[self.z_EPSG]
        df['Altitude.' + self.z_EPSG + ' (m)'] = h_geoid
        df['Easting.' + self.xy_EPSG + ' (m)'] = df['geometry'].x
        df['Northing.' + self.xy_EPSG + ' (m)'] = df['geometry'].y
        df = hubdist(df,dfref,self.CL_dist_threshold,'Chainage',Easting = 'Easting.' + self.xy_EPSG + ' (m)', Northing = 'Northing.' + self.xy_EPSG + ' (m)')
        self.RSS_data = df
        return self
    
    def plot_Alt_XScoord_vs_Tnum(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.RSS_data['Trace'],self.RSS_data['Altitude.' + self.z_EPSG + ' (m)'],'r')
        ax1.plot(self.RSS_data['Trace'],self.RSS_data['Hubs'])
        ax1.set_xlabel('RSS Trace Number')
        ax1.set_ylabel('Cross section coordinate (m)')
        ax2.set_ylabel('Altitude.' + self.z_EPSG + ' (m)')

    def plot_WP_XScoord_vs_Tnum(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(self.RSS_data['Trace'],self.RSS_data['Next_WP'],'r')
        ax1.plot(self.RSS_data['Trace'],self.RSS_data['Hubs'])
        ax1.set_xlabel('RSS Trace Number')
        ax1.set_ylabel('Cross section coordinate (m)')
        ax2.set_ylabel('Next WP')
        
    def make_WPfile(self,**kwargs):
        nwp_interp = self.RSS_data['Next_WP']
        numwp = int(np.nanmax(nwp_interp))
        wpdict = dict()
        lt_cutoff = kwargs.get('lt_cutoff',5)# trace cutoff at lower end of the automatically identified trace window
        ut_cutoff = kwargs.get('ut_cutoff',10)# trace cutoff at upper end of the automatically identified trace window 
        flight_alt = kwargs.get('flight_alt',4) # Flight altitude (is 4 m above water in the standard scenario)
        for i in range(1,numwp+1,1):
            wpdict[i] = dict()
            wpdict[i]['Waypoint'] = i
            wpdict[i]['Start Trace'] = self.RSS_data['Trace'][np.where(nwp_interp==i)[0][0]] + lt_cutoff
            wpdict[i]['End Trace'] = self.RSS_data['Trace'][np.where(nwp_interp==i)[0][-1]] - ut_cutoff
            wpdict[i]['Flight Altitude'] = flight_alt
            values, counts = np.unique(self.RSS_data['Direction'][np.where(nwp_interp==i)[0]], return_counts = True)
            ind = np.argmax(counts)
            wpdict[i]['Direction'] = values[ind]
            if wpdict[i]['Direction'] == 1:
                wpdict[i]['v upper'] = -30 #upper bound of fitting window, can be adjusted per waypoint
                wpdict[i]['v lower'] = -200 #lower bound of fitting window, can be adjusted per waypoint
                wpdict[i]['v mask'] = -15 # Any values inside the blind zone will be ignored
            else:
                wpdict[i]['v upper'] = 200
                wpdict[i]['v lower'] = 30
                wpdict[i]['v mask'] = 15
            wpdict[i]['Fitting Model'] = 'Single Peak' # Single or Double peak
            # Setting starting parameters for the fitting model
            if wpdict[i]['Direction'] == 1: 
                wpdict[i]['m1'] = -50
                wpdict[i]['m2'] = -50
            else:
                wpdict[i]['m1'] = 50
                wpdict[i]['m2'] = 50
            wpdict[i]['s1'] = 10
            wpdict[i]['s2'] = 10
            wpdict[i]['f1'] = 1000
            wpdict[i]['f2'] = 1000

        wpframe = pd.DataFrame.from_dict(wpdict).transpose()
        wpframe.to_excel(self.name_wpfile)
        
    def plot_WPdata(self):
        wpdata = pd.read_excel(self.name_wpfile,index_col = 0)
        wpdata["Cross section coordinate (m)"] = ""
        for w in wpdata.index:
            subflight = self.RSS_data[np.logical_and(self.RSS_data['Trace'] > wpdata['Start Trace'][w], self.RSS_data['Trace'] < wpdata['End Trace'][w])]
            wpdata.loc[w,"Cross section coordinate (m)"] = subflight['Hubs'].values.mean()

        plt.figure()
        plt.plot(self.RSS_data['Trace'],self.RSS_data['Hubs'])
        plt.plot(wpdata['Start Trace'],wpdata["Cross section coordinate (m)"], 'k.',markersize = 10)
        plt.plot(wpdata['End Trace'],wpdata["Cross section coordinate (m)"], 'g.', markersize = 10)
        plt.xlabel('RSS Trace Number')
        plt.ylabel('Cross section coordinate (m)')
        plt.show()
        
    def process_RSS_data(self):
        n_segyfile = self.name_segyfile
        data,headers, sample_rate, ntrace, nsamples = readsegy(n_segyfile)
        self.fwf_data =np.concatenate((data[2048:,:],data[0:2048,:]))
        self.vaxis_sort = np.concatenate((np.linspace(-2048,-1,2048)*self.vres, np.linspace(1,2048,2048)*self.vres))/10/ np.cos(self.angle/360*2*np.pi)
        self.waypoints = pd.read_excel(self.name_wpfile)
        self.waypoints["Cross section coordinate (m)"] = ""
        self.waypoints["RT Velocity"] = ""
        self.waypoints["Time stamp"] = ""
        self.waypoints["Easting"] = ""
        self.waypoints["Northing"] = ""
        self.waypoints['Altitude.' + self.z_EPSG + ' (m)'] = ""
        self.waypoints['fit_vax'] = ""
        self.waypoints['fit_data'] = ""
        self.waypoints['FW Velocity'] = np.nan
        self.waypoints['popt'] = dict()
        self.waypoints['pcov'] = dict()
        self.waypoints['fit_vax'] = dict()
        self.waypoints['fit_data'] = dict()
        self.popt = dict()
        self.pcov = dict()
        self.fit_vax = dict()
        self.fit_data = dict()
        for wp_idx , w in enumerate(self.waypoints.index):
            subflight = self.RSS_data[np.logical_and(self.RSS_data['Trace'] > self.waypoints['Start Trace'][w], self.RSS_data['Trace'] < self.waypoints['End Trace'][w])]
            subdata = self.fwf_data[: , self.waypoints['Start Trace'][w] : self.waypoints['End Trace'][w] + 1]
            subdata_m = subdata.mean(1)
            vax = self.vaxis_sort
            vlower = self.waypoints['v lower'][w]
            vupper = self.waypoints['v upper'][w]
            ind = np.logical_and(vax >vlower,vax <vupper)
            vax = vax[ind]
            subdata_m = subdata_m[ind]
            vmask = self.waypoints['v mask'][w]
            ind = np.logical_or(vax <-vmask,vax >vmask)
            vax = vax[ind]
            subdata_m = subdata_m[ind]
            plt.figure()
            plt.title('Waypoint' + str(wp_idx+1))
            plt.pcolor(np.linspace(1,subdata.shape[1],subdata.shape[1]),self.vaxis_sort,subdata,vmin = 0, vmax = 500)
            plt.ylim([-200, 200])
            plt.xlabel('Traces')
            plt.ylabel('Flow velocity, cm/s')
            self.waypoints.loc[w,'Cross section coordinate (m)'] = subflight['Hubs'].values.mean()
            self.waypoints.loc[w,'Easting.' + self.xy_EPSG + ' (m)'] = subflight['Easting.' + self.xy_EPSG + ' (m)'].values.mean()
            self.waypoints.loc[w,'Northing.' + self.xy_EPSG + ' (m)'] = subflight['Northing.' + self.xy_EPSG + ' (m)'].values.mean()
            self.waypoints.loc[w,'Altitude.' + self.z_EPSG + ' (m)'] = subflight['Altitude.' + self.z_EPSG + ' (m)'].values.mean()
            self.waypoints.loc[w,'RT Velocity'] = subflight['Velocity raw'].values.mean()*100
            self.waypoints.loc[w,'Time stamp'] = skyhubtime(subflight,'Date','Time').astype('float').mean().astype('datetime64[us]')
            try:
                if  self.waypoints['Fitting Model'][w] == 'Single Peak':
                    p0 = [self.waypoints['m1'][w],self.waypoints['s1'][w],self.waypoints['f1'][w]]
                    popt,pcov = curve_fit(peakmodel1, vax, subdata_m, p0=p0)#,bounds=flight_dict ['Aux data']['pm2 bounds'])
                elif self.waypoints['Fitting Model'][w]  == 'Double Peak':
                    p0 = [self.waypoints['m1'][w],self.waypoints['m2'][w],self.waypoints['s1'][w],self.waypoints['s2'][w],self.waypoints['f1'][w],self.waypoints['f2'][w]]
                    popt,pcov = curve_fit(peakmodel2, vax, subdata_m, p0=p0)
            except RuntimeError:
                print('an error occured in the fitting')
                vpick = np.nan
                plt.plot(vax,subdata_m,'k.')
                print(' Waypoint ' + str(w) +  ' could not be fitted')
                plt.title (' Waypoint ' + str(w))
                plt.show()
                popt = np.nan
                pcov = np.nan
            else:
                if  self.waypoints['Fitting Model'][w] == 'Single Peak':
                    p0 = [self.waypoints['m1'][w],self.waypoints['s1'][w],self.waypoints['f1'][w]]
                    popt,pcov = curve_fit(peakmodel1, vax, subdata_m, p0=p0)#,bounds=flight_dict ['Aux data']['pm2 bounds'])
                    fdata = peakmodel1(vax,popt[0],popt[1],popt[2])
                    self.waypoints.loc[w,'FW Velocity'] = popt[0]

                elif self.waypoints['Fitting Model'][w]  == 'Double Peak':
                    p0 = [self.waypoints['m1'][w],self.waypoints['m2'][w],self.waypoints['s1'][w],
                          self.waypoints['s2'][w],self.waypoints['f1'][w],self.waypoints['f2'][w]]
                    popt,pcov = curve_fit(peakmodel2, vax, subdata_m, p0=p0)
                    fdata = peakmodel2(vax,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
                    self.waypoints.loc[w,'FW Velocity'] = popt[0]
                    self.waypoints.loc[w,'Two Velocity together'] = np.nan
                    if self.waypoints['Direction'][w] == 1:
                        FW_Velocity_flow = min(abs(popt[0]) , abs(popt[1]))
                        FW_Velocity_drone = max(abs(popt[0]) , abs(popt[1]))
                    else:
                        FW_Velocity_flow = min(abs(popt[0]) , abs(popt[1]))
                        FW_Velocity_drone = max(abs(popt[0]) , abs(popt[1]))
                    self.waypoints.loc[w,'FW Velocity'] = FW_Velocity_flow
                    self.waypoints.loc[w,'Two Velocity together'] = FW_Velocity_drone
                plt.figure()
                plt.title('Waypoint' + str(wp_idx+1))
                plt.plot(vax,subdata_m,'k.')
                plt.plot(vax,fdata,'r-')
                plt.xlim(vlower,vupper)
                plt.xlabel('Velocity, cm/s')
                plt.ylabel('Return power')
                if  self.waypoints['Fitting Model'][w] == 'Double Peak':
                    plt.plot([popt[0],popt[0]],plt.gca().get_ylim(),color = 'r')
                    plt.plot([popt[1],popt[1]],plt.gca().get_ylim(),color = 'b')
                else:
                    plt.plot([popt[0],popt[0]],plt.gca().get_ylim())
                plt.show()
                phi = sum((fdata-subdata_m)**2/len(fdata))**0.5
                print('------------ Waypoint '+str(wp_idx+1)+'------------')
                print('Phi='+str(phi))
                if  self.waypoints['Fitting Model'][w] == 'Double Peak':
                    print('Vel_1='+str(popt[0]))
                    print('Vel_2='+str(popt[1]))
                else:
                    print('Vel='+str(popt[0]))
            self.popt[w]= popt
            self.pcov[w]= pcov
            self.fit_vax[w] = vax
            self.fit_data[w] = subdata_m
            
        return self
    
    def plot_RSS_res(self):
        labels = [];
        if os.path.isfile(self.in_situ_L3):
            is_data = pd.read_csv(self.in_situ_L3)
            l1, = plt.plot(is_data['Cross section coordinate (m)'],is_data['Surface velocity (cm/s)'],'r*',label='OTT MFPro')
            labels.append(l1)
        XS_L3 = self.waypoints
        colcodes = ['k','b','g']
        fa = np.unique(XS_L3['Flight Altitude'])
        count = 0
        for faa in fa:
            plotpoints = XS_L3[XS_L3['Flight Altitude']==faa]
            #l2= plt.errorbar(plotpoints['Distance'],plotpoints['Velocity_fwd min'],yerr=(plotpoints['Velocity_fwd min_cov'].values)**0.5,linestyle='',marker='d',color=colcodes[count],label='Flight altitude ' + str(faa) +' m')
            l2,= plt.plot(plotpoints['Cross section coordinate (m)'],plotpoints['FW Velocity'],linestyle='',marker='d',color=colcodes[count],label='Flight altitude ' + str(faa) +' m')
            labels.append(l2)
            count = count+1
        plt.legend(handles = labels)
        plt.xlabel('Distance along cross section (m)')
        plt.ylabel('Surface velocity, cm/s')
        plt.title(self.name_segyfile.split('\\')[-1] )
        
    def L2_L3_generation(self):
        XS_L3 = self.waypoints
        XS_L3=XS_L3.drop(columns=['Start Trace', 'End Trace','m1','m2','s1','s2','f1','f2','Flight Altitude'])
        if  self.waypoints['Fitting Model'][1] == 'Double Peak':
            XS_L3 = XS_L3.rename(columns={
                'RT Velocity': 'Velocity raw cm/s', 
                "FW Velocity": "Velocity full waveform data cm/s",
                "Two Velocity together": "Two Velocity together cm/s",
                })
        else:
            XS_L3 = XS_L3.rename(columns={
                'RT Velocity': 'Velocity raw cm/s', 
                "FW Velocity": "Velocity full waveform data cm/s",
                })
        if  self.waypoints['Fitting Model'][1] == 'Double Peak':
            XS_L3 = XS_L3[['Time stamp','Easting.' + self.xy_EPSG + ' (m)', 'Northing.' + self.xy_EPSG + ' (m)', 'Altitude.' + self.z_EPSG + ' (m)', 
                       'Cross section coordinate (m)', 'Direction','Velocity raw cm/s',"Velocity full waveform data cm/s", "Velocity full waveform data cm/s","Two Velocity together cm/s"]]
        else:
            XS_L3 = XS_L3[['Time stamp','Easting.' + self.xy_EPSG + ' (m)', 'Northing.' + self.xy_EPSG + ' (m)', 'Altitude.' + self.z_EPSG + ' (m)', 
                           'Cross section coordinate (m)', 'Direction','Velocity raw cm/s',"Velocity full waveform data cm/s", "Velocity full waveform data cm/s"]]
        XS_L3 = XS_L3.rename(columns={
            'Time stamp': 'Time stamp (UTC)',
            })
        XS_L3.to_csv(self.L3_output )
        XS_L2=XS_L3.drop(columns=[ 'Cross section coordinate (m)'])
        XS_L2.to_csv(self.L2_output)