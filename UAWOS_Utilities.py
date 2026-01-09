# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:30:36 2025

@author: vpk410
"""
import os
import struct
import numpy as np
from datetime import datetime, timedelta
from osgeo import gdal
from scipy import interpolate
import segyio
from scipy.spatial import cKDTree
from scipy.stats import norm

def skyhubtime(df,datename,timename,**kwargs): #turns time from skyhub into datetime objects
    #*****************************************************************************
    # Adjustable input
    #*****************************************************************************
    tformatstr = '%Y/%m/%d%H:%M:%S.%f' #default time format for skyhub files
    if 'tformatstr' in kwargs:
        tformatstr = kwargs.get('tformatstr') # hand over non-default time format with argument "tformatstr"
    #*****************************************************************************
    tt=df[datename] + df[timename]
    t=[]
    for i in tt:
        t.append(np.datetime64(datetime.strptime(i, tformatstr)))
    return np.array(t)

def posinterp(df,dfpos,datename,timename,latname,lonname,altname,anglesyn,**kwargs): #interpolates accurate positions by matching the time stamp in a position log and a ppk file when time is given as date and time columns
    #*****************************************************************************
    # Adjustable input
    #*****************************************************************************    
    UTC_GPST_offset = 18*1E6 # in microseconds, may have to be adjusted
    GPST_string = 'GPST'
    skyhub_datename = 'Date'
    if 'skyhub_datename' in kwargs:
        skyhub_datename = kwargs.get('skyhub_datename')
    skyhub_timename = 'Time' # Skyhub time column - assumed to be UTC
    if 'skyhub_timename' in kwargs:
        skyhub_timename = kwargs.get('skyhub_timename')
    skyhub_pitchname = 'Pitch'
    if 'skyhub_pitchname' in kwargs:
        skyhub_pitchname = kwargs.get('skyhub_pitchname')
    skyhub_rollname = 'Roll'
    if 'skyhub_rollname' in kwargs:
        skyhub_rollname = kwargs.get('skyhub_rollname')
    interp_pitchname = 'Pitch_interpolated'
    if 'interp_pitchname' in kwargs:
        interp_pitchname = kwargs.get('interp_pitchname')
    interp_rollname = 'Roll_interpolated'
    if 'interp_rollname' in kwargs:
        interp_rollname = kwargs.get('interp_rollname')
    interp_latname = 'Latitude_interpolated'
    if 'interp_latname' in kwargs:
        interp_latname = kwargs.get('interp_latname')
    interp_lonname = 'Longitude_interpolated'
    if 'interp_lonname' in kwargs:
        interp_lonname = kwargs.get('interp_lonname') 
    interp_altname = 'Altitude_interpolated'
    if 'interp_altname' in kwargs:
        interp_altname = kwargs.get('interp_altname') 
   #***************************************************************************** 
    df=df.copy(deep = True)
    tpos = skyhubtime(dfpos,datename,timename)
    tsens = skyhubtime(df,skyhub_datename,skyhub_timename)
    latpos = dfpos[latname]
    tpos = tpos.astype('float')
    if timename == GPST_string:
        tpos = tpos-UTC_GPST_offset
    latint = np.interp(tsens.astype('float'),tpos,latpos)
    lonpos = dfpos[lonname]
    lonint = np.interp(tsens.astype('float'),tpos,lonpos)
    altpos = dfpos[altname]
    altint = np.interp(tsens.astype('float'),tpos,altpos)
    if anglesyn:
        pitchpos = dfpos[skyhub_pitchname]
        pitchint = np.interp(tsens.astype('float'),tpos,pitchpos)
        rollpos = dfpos[skyhub_rollname]
        rollint = np.interp(tsens.astype('float'),tpos,rollpos)
        df[interp_pitchname] = pitchint
        df[interp_rollname] = rollint
    df[interp_latname] = latint
    df[interp_lonname] = lonint
    df[interp_altname] = altint
    df = df[np.logical_and(tsens.astype('float')>np.min(tpos),tsens.astype('float')<np.max(tpos))]
    return df

def raster_interp(x,y,n_raster,**kwargs): # interpolates any raster value to points, based on a raster input
    #*****************************************************************************
    # Adjustable input
    #*****************************************************************************    
    interp_method = 'linear'
    #*****************************************************************************     
    raster = gdal.Open(n_raster)
    img = raster.GetRasterBand(1)
    rasterv = img.ReadAsArray()
    nd_value = kwargs.get('nd_value')
    if nd_value:
        rasterv[np.where(rasterv==nd_value)]=np.nan
    width = raster.RasterXSize
    height = raster.RasterYSize
    gt = raster.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    xg = np.linspace(minx + gt[1]/2,maxx-gt[1]/2,width)
    yg = np.linspace(maxy + gt[5]/2,miny-gt[5]/2,height)
    interp = interpolate.RegularGridInterpolator((-yg, xg), rasterv,method=interp_method)
    rasterint = interp(np.array([-np.array(y),x]).transpose())
    return rasterint

def hubdist(points_in,hubs,tol,chainage_field,**kwargs): # matches data points to a reference line
    #*****************************************************************************
    # Adjustable input
    #*****************************************************************************  
    Easting_points = 'Easting'
    if 'Easting' in kwargs:
        Easting_points = kwargs.get('Easting', None)
    Northing_points = 'Northing'
    if 'Northing' in kwargs:
        Northing_points = kwargs.get('Northing', None)
    Easting_hubs = 'Easting'
    if 'Easting_hubs' in kwargs:
        Easting_hubs = kwargs.get('Easting_hubs', None)
    Northing_hubs = 'Northing'
    if 'Northing_hubs' in kwargs:
        Northing_hubs = kwargs.get('Northing_hubs', None)
    Hubs_out = 'Hubs'
    if 'Hubs_out' in kwargs:
        Hubs_out = kwargs.get('Hubs_out', None)
    Distance_Hubs_out = 'Distance_Hubs'
    if 'Distance_Hubs_out' in kwargs:
        Distance_Hubs_out = kwargs.get('Distance_Hubs_out', None)
    #*****************************************************************************    
    points = points_in.copy(deep=True)
    nA = np.array((points_in[Easting_points],points_in[Northing_points])).transpose()
    nB = np.array((hubs[Easting_hubs],hubs[Northing_hubs])).transpose()
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    points[Hubs_out] = hubs[chainage_field][idx].values
    points[Distance_Hubs_out] = dist
    points = points.loc[points[Distance_Hubs_out]<=tol]
    return points

def _rename_corrected_file(src_path ): # Internal function for readsegy
    #*****************************************************************************
    # Adjustable input
    #*****************************************************************************  
    suffix= "_corrected"
    #*****************************************************************************
    folder, name = os.path.split(src_path)
    stem, ext = os.path.splitext(name)
    return os.path.join(folder, f"{stem}{suffix}{ext}")

def _correct_segy_file( in_path, out_path): # Internal function for readsegy
    #*****************************************************************************
    # Adjustable input
    #***************************************************************************** 
    ns = 512
    bps = 4
    header_size = 3600
    trace_header_size = 240
    #*****************************************************************************
    size = os.path.getsize(in_path)
    step = trace_header_size + ns * bps
    n_traces = (size - header_size) // step
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(fin.read(header_size))
        for _ in range(int(n_traces)):
            th = bytearray(fin.read(trace_header_size))
            th[114:116] = struct.pack(">H", ns)
            fout.write(th)
            fout.write(fin.read(ns * bps))
    return out_path

def readsegy(filename): #areads full waveform data from WPR/Doppler
    #*****************************************************************************
    # Adjustable input
    #*****************************************************************************  
    skiptraceini = 0 # 
    skiptraceend = 0
    strict_yn = False
    endian_str = 'little'
    dt_unit_conv = 1000.
    #*****************************************************************************  
    try:
        with segyio.open(filename,strict = strict_yn,endian = endian_str) as f:
            traces = [np.copy(tr) for tr in f.trace[skiptraceini:-(skiptraceend +1)]]
            headers = [head for head in f.header[skiptraceini:-(skiptraceend +1)]]
            sample_rate = segyio.tools.dt(f) / dt_unit_conv
        print('Sampling rate: ', sample_rate)
        ntrace = len(traces)
        print('Number of traces in dataset: ', ntrace)
        nsamples = len(traces[0])
        print('Number of samples in each trace: ',nsamples)
        traces = np.transpose(np.array(traces))
        hdic = []
        for i in range(ntrace):
            hnow = dict((a.strip(), int(b.strip()))
                             for a, b in (element.split(':')  
                                          for element in str(headers[i])[2:-1].split(', ')))
            hdic.append(hnow)      
        return traces, hdic, sample_rate, ntrace, nsamples
    except Exception as e1:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Error while read sgy file：{e1}")
        print("Attempting internal correction...")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        corrected_path = _rename_corrected_file(filename, suffix="_corrected")
        try:
            _correct_segy_file( in_path=filename, out_path=corrected_path )
        except Exception as e2:
            print(f"Internal correction failed (file writing stage): {e2}")
            raise RuntimeError("Internal correction failed...") from e2
        try:
            with segyio.open(corrected_path ,strict = strict_yn,endian = endian_str) as f:
                traces = [np.copy(tr) for tr in f.trace[skiptraceini:-(skiptraceend +1)]]
                headers = [head for head in f.header[skiptraceini:-(skiptraceend +1)]]
                sample_rate = segyio.tools.dt(f) / dt_unit_conv
            print('Sampling rate: ', sample_rate)
            ntrace = len(traces)
            print('Number of traces in dataset: ', ntrace)
            nsamples = len(traces[0])
            print('Number of samples in each trace: ',nsamples)
            traces = np.transpose(np.array(traces))
            hdic = []
            for i in range(ntrace):
                hnow = dict((a.strip(), int(b.strip()))
                                 for a, b in (element.split(':')  
                                              for element in str(headers[i])[2:-1].split(', ')))
                hdic.append(hnow)      
            return traces, hdic, sample_rate, ntrace, nsamples
        except Exception as e3:
            print(f"Internal correction failed:{e3}")
            raise RuntimeError("Internal correction failed...") from e3

def peakmodel2(v,mu1,mu2,s1,s2,f1,f2):
    ret_pow = f1*norm.pdf((v-mu1)/s1) + f2*norm.pdf((v-mu2)/s2)
    return ret_pow

def peakmodel1(v,mu1,s1,f1):
    ret_pow = f1*norm.pdf((v-mu1)/s1)
    return ret_pow