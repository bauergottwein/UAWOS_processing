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
        eps = 81 #relative permittivity of water
        self.vwat = 3E8/eps**0.5
        
            
    def process_WPR_flight(self,**kwargs):
        geoid_exists = kwargs.get('geoid_exists',True)
        cl = gpd.read_file(self.refline_file)
        cl = cl.to_crs(crs = self.xy_EPSG)
        dfref = pd.DataFrame(np.array([cl['geometry'].x,cl['geometry'].y,cl[self.chainage_field]]).transpose(), columns = ['Easting', 'Northing', 'Chainage'])
        n_posfile = self.name_posfile
        pos_data = pd.read_csv(n_posfile)
        pos_data = pos_data.loc[~np.isnan(pos_data['zGPR:Trace'])]
        lat_name = 'Latitude RTK'
        lon_name = 'Longitude RTK'
        alt_name = 'Altitude RTK'
        if not self.RTK_on:
            n_ppkfile = self.name_ppkfile
            ppk_data = pd.read_csv(n_ppkfile,skiprows = 9,delimiter=r"\s+")
            pos_data = posinterp(pos_data,ppk_data,'%','UTC','latitude(deg)','longitude(deg)', 'height(m)', False)
            lat_name = 'Latitude_interpolated'
            lon_name = 'Longitude_interpolated'
            alt_name = 'Altitude_interpolated'
        df = pd.DataFrame(np.array([pos_data['Date'], pos_data['Time'], pos_data[lon_name], pos_data[lat_name], pos_data[alt_name], pos_data['ALT:Altitude'], pos_data['zGPR:Trace']]).transpose(), columns = ['Date','Time','Longitude','Latitude','Altitude WGS84','Altitude TTF','WPR Trace']) 
        df['geometry'] = df.apply(lambda x: Point((float(x.Longitude), float(x.Latitude))), axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry', crs = 4326)
        df = df.to_crs(self.xy_EPSG)
        if geoid_exists:
            Geoid = raster_interp(df['geometry'].x.values,df['geometry'].y.values,self.geoid_tiff)
            df[self.z_EPSG] = Geoid
            h_geoid = df['Altitude WGS84'] - df[self.z_EPSG]
            df['Altitude.' + self.z_EPSG + ' (m)'] = h_geoid
        df['Easting.' + self.xy_EPSG + ' (m)'] = df['geometry'].x
        df['Northing.' + self.xy_EPSG + ' (m)'] = df['geometry'].y
        df = hubdist(df,dfref,self.CL_dist_threshold,'Chainage',Easting = 'Easting.' + self.xy_EPSG + ' (m)', Northing = 'Northing.' + self.xy_EPSG + ' (m)')
        self.pos_data = df
        return self
    
    def plot_Tnum_Alt_vs_XScoord(self,**kwargs):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        pos_data = self.pos_data
        Alt_field = kwargs.get('Alt_field','Altitude.' + self.z_EPSG + ' (m)')
        ax2.plot(pos_data['Hubs'],pos_data[Alt_field],'r',label='Altitude')
        ax1.plot(pos_data['Hubs'],pos_data['WPR Trace'], label = 'Trace Number')
        ax1.set_xlabel('Cross section coordinate (m)')
        ax1.set_ylabel('WPR Trace Number')
        ax2.set_ylabel('Altitude.' + self.z_EPSG + ' (m)')
        
    def plot_XScoord_Alt_vs_Tnum(self,**kwargs):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        pos_data = self.pos_data
        Alt_field = kwargs.get('Alt_field','Altitude.' + self.z_EPSG + ' (m)')
        ax2.plot(pos_data['WPR Trace'],pos_data[Alt_field],'r')
        ax1.plot(pos_data['WPR Trace'],pos_data['Hubs'])
        ax1.set_xlabel('WPR Trace Number')
        ax1.set_ylabel('Cross section coordinate (m)')
        ax2.set_ylabel('Altitude.' + self.z_EPSG + ' (m)')
        
    def plot_sections(self):
        sections = pd.read_excel(self.name_secfile,index_col = 'Section')
        pos_data = self.pos_data
        plt.figure()
        for s in sections.index:
                plt.plot(pos_data['WPR Trace'],pos_data['Hubs'],'b')
                plt.plot([sections['Start Trace'][s],sections['Start Trace'][s]],plt.gca().get_ylim(), 'k',markersize = 10)
                plt.plot([sections['End Trace'][s],sections['End Trace'][s]],plt.gca().get_ylim(), 'g',markersize = 10)
        plt.xlabel('WPR Trace number')
        plt.ylabel('Cross section coordinate (m)')
        plt.show()
        
    def find_WSE(self):
        offset_TTF0_true_ground = 0.21 #estimated empirically from in-situ WSE
        sections = pd.read_excel(self.name_secfile,index_col = 'Section')
        pos_data = self.pos_data
        for s in sections.index:
            startt = sections['Start Trace'][s]
            endt = sections['End Trace'][s]
            meanalt = pos_data['Altitude.' + self.z_EPSG + ' (m)'][startt:endt].mean()
            print('Mean altitude, section ', s, meanalt)
            meanrange = pos_data['Altitude TTF'][startt:endt].mean()
            print('Mean range, section ', s, meanrange)
            print('WSE section ', s, meanalt-meanrange-offset_TTF0_true_ground)
            
    def process_WPR_data(self):
        fsize = 18
        sections = pd.read_excel(self.name_secfile,index_col = 'Section')
        pos_data = self.pos_data
        data,headers, sample_rate, ntrace, nsamples = readsegy(self.name_segyfile)
        ttorange = sample_rate*1E-9*self.vwat/2 #sampling rate in ns
        self.automatic_bathymetry_points = dict()
        for s in sections.index:
            starti = pos_data.loc[pos_data['WPR Trace'] == sections['Start Trace'][s]].index -1
            endi = pos_data.loc[pos_data['WPR Trace'] == sections['End Trace'][s]].index +1
            indeces = range(starti[0],endi[0])
            spos_data = pos_data.iloc[np.array(indeces)]
            stime = skyhubtime(spos_data,'Date','Time').astype('float').astype('datetime64[us]')
            feast = interpolate.interp1d(spos_data['Hubs'],spos_data['Easting.' + self.xy_EPSG + ' (m)'])
            fnorth = interpolate.interp1d(spos_data['Hubs'],spos_data['Northing.' + self.xy_EPSG + ' (m)'])
            falt = interpolate.interp1d(spos_data['Hubs'],spos_data['Altitude.' + self.z_EPSG + ' (m)'])
            fts = interpolate.interp1d(spos_data['Hubs'],stime)
            subdata = data[:,sections['Start Trace'][s]:sections['End Trace'][s]+1]
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
                traceindex[i] = sections['Start Trace'][s] + i
                distindex[i] = pos_data.loc[pos_data['WPR Trace'] == sections['Start Trace'][s]+ i]['Hubs'].values
            rangeax = np.linspace(1,nsamples,nsamples)*ttorange
            gain = ((rangeax-self.gain_lt)/(self.gain_ut-self.gain_lt))**self.gain_power
            gain[rangeax<self.gain_lt]=0
            gain[rangeax>self.gain_ut]=1
            mgain = np.tile(gain,(subdata.shape[1],1)).transpose()
            gndata = ndata*mgain
            
            XXv, YYv = np.meshgrid(distindex, rangeax)
            #%% Figure 1
            plt.figure()
            plt.pcolor(XXv, YYv, gndata, vmin=np.percentile(gndata, 0.05), vmax=np.percentile(gndata, 99.9))
            idx = np.argmax(gndata, axis=0)
            plt.plot(distindex, idx*ttorange, 'r.')
            plt.xlabel('Distance along cross section, m', fontsize=fsize)
            plt.ylabel('GPR range, m', fontsize=fsize)
            plt.title(self.name_segyfile.split('\\')[-1] )
            plt.tick_params(labelsize=fsize)
            # %% Pick some points for constructing a filter, which was used to filter out signals with shorter time (values =0)
            print('Please pick a rough lower bound on the bathymetry range by right-clicking your mouse, and clicking the middle for the ending:')
            # Figure 2
            plt.figure()
            plt.pcolor(gndata, vmin=np.percentile(gndata, 0.05),vmax=np.percentile(gndata, 99.9))
            plt.plot(idx, 'r.')
            plt.xlim(-10 , gndata.shape[1]+10)
            points_prof = plt.ginput(-1, timeout=-1, show_clicks=True,mouse_add=3, mouse_pop=1, mouse_stop=2)
            points_prof = np.array(points_prof)
            points_prof[0, 0] = 0
            points_prof[-1, 0] = gndata.shape[1]
            f = interpolate.interp1d(points_prof[:, 0], points_prof[:, 1])
            trax = np.linspace(0, gndata.shape[1]-1, gndata.shape[1])
            rax = np.linspace(0, gndata.shape[0]-1, gndata.shape[0])
            thresh_range = f(trax)
            plt.plot(trax, thresh_range, 'w.')
            plt.xlabel('Traces', fontsize=fsize)
            plt.ylabel('GPR range, time', fontsize=fsize)
            plt.title(self.name_segyfile.split('\\')[-1] )
            plt.tick_params(labelsize=fsize)
            tthresh_range = np.tile(thresh_range, (subdata.shape[0], 1))
            thrx, thry = np.meshgrid(trax, rax)
            tgndata = gndata*(thry > tthresh_range)
# %%         To construct a Gaussian lowpass filter
            filtedData = gaussian_filter(tgndata, sigma=2)
            tgndata=filtedData
#%%         Figure 3 (after filtering by a lowpass filter)
            plt.figure()
            plt.pcolor( tgndata, vmin=np.percentile(tgndata, 0.1),vmax=np.percentile(tgndata, 99.9))
            plt.xlabel('Traces', fontsize=fsize)
            plt.ylabel('GPR range, time', fontsize=fsize)
            plt.title(self.name_segyfile.split('\\')[-1] )
            plt.tick_params(labelsize=fsize)
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
            plt.xlim(-10 , gndata.shape[1]+10)
            print('Please manual updating by right-clicking your mouse, and clicking the middle for the ending:')
            points_surf = plt.ginput(-1, timeout=-1, show_clicks=True,mouse_add=3, mouse_pop=1, mouse_stop=2)
            points_manu = np.array(points_surf)
            plt.plot(points_manu[:, 0], points_manu[:, 1], 'w*')
            for idx in range(points_manu.shape[0]):
                new_idx = int(np.around(points_manu[idx, 0]))
                points_prof[new_idx, 1] = points_manu[idx, 1]
# %%        Figure 4 for showing updated results
            plt.figure() 
            plt.pcolor(XXv, YYv, gndata, vmin=np.percentile(gndata, 0.05), vmax=np.percentile(gndata, 99.9))
            plt.plot(distindex, points_prof[:, 1]*ttorange, 'r.')
            plt.xlabel('Distance along cross section, m', fontsize=fsize)
            plt.ylabel('GPR range, m', fontsize=fsize)
            plt.title(self.name_segyfile.split('\\')[-1] )
            plt.tick_params(labelsize=fsize)
# %%        Caculating the bathymetry
            points_prof[:, 1] = points_prof[:, 1]*ttorange-surf
            wse = self.WSE
            points_prof[:, 1] = wse - points_prof[:, 1]
            dfbathyauto = pd.DataFrame({'Cross section coordinate (m)': distindex, 'z.' + self.z_EPSG + ' (m)': points_prof[:, 1]})
            dfbathyauto['x.' + self.xy_EPSG +' (m)'] = feast(dfbathyauto['Cross section coordinate (m)'])
            dfbathyauto['y.' + self.xy_EPSG +' (m)'] = fnorth(dfbathyauto['Cross section coordinate (m)'])
            dfbathyauto['Altitude.' + self.z_EPSG +' (m)'] = falt(dfbathyauto['Cross section coordinate (m)'])
            dfbathyauto['Time stamp (UTC)'] = fts(dfbathyauto['Cross section coordinate (m)']).astype('float').astype('datetime64[us]')
            self.automatic_bathymetry_points[s] = dfbathyauto
        return self
            
    def plot_results(self, xlim):
        fsize = 18
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
            l1, = plt.plot(is_data['XS coordinate (m)'],is_data['Elevation'],'k*',label='in situ')
            labels.append(l1)
        
        plt.plot(plt.gca().get_xlim(),[WSE,WSE],'b')
        plt.xlim(xlim)
        plt.xlabel('Distance along cross section, m',fontsize=fsize)
        plt.ylabel('River bottom elevation, m above ' + self.z_EPSG,fontsize=fsize)
        plt.title(self.name_segyfile.split('\\')[-1] )
        plt.tick_params(labelsize=fsize)
        plt.legend(handles = labels,fontsize=fsize-3)
        
    def L2_L3_generation(self):
        XS_L3_auto = self.automatic_bathymetry_points[1].head(0)
        XS_L3_auto['Frequency (MHz)'] = None
        XS_L3_auto['Antenna direction'] = None
        XS_L3_auto['WSE.' + self.z_EPSG + ' (m)'] = None
        for s in self.automatic_bathymetry_points:
            df = self.automatic_bathymetry_points[s]
            df['Frequency (MHz)'] = self.cfreq
            df['Antenna orientation'] = self.dir
            df['WSE.' + self.z_EPSG + ' (m)'] = self.WSE
            XS_L3_auto = pd.concat([XS_L3_auto, df], axis=0)
        XS_L3_auto = XS_L3_auto[XS_L3_auto['z.' + self.z_EPSG + ' (m)'].notna()]
        XS_L3_auto.to_csv(self.L3_output, index = False)
        XS_L2_auto=XS_L3_auto.drop(columns=['Cross section coordinate (m)'])
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
        plt.xlabel('Trace')
        plt.ylabel('WPR range, m')
        plt.title(self.name_segyfile.split('\\')[-1] )
        plt.show()
        
