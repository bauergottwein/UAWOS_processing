# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:30:36 2025

@author: vpk410
"""
import numpy as np
from datetime import datetime, timedelta
from osgeo import gdal
from scipy import interpolate
import segyio
from scipy.spatial import cKDTree
from scipy.stats import norm

def skyhubtime(df,datename,timename): #turns time from skyhub into datetime objects
    tt=df[datename] + df[timename]
    t=[]
    for i in tt:
        t.append(np.datetime64(datetime.strptime(i, '%Y/%m/%d%H:%M:%S.%f')))
    return np.array(t)

def posinterp(df,dfpos,datename,timename,latname,lonname,altname,anglesyn): #interpolates accurate positions by matching the time stamp in a position log and a ppk file when time is given as date and time columns
    UTC_GPST_offset = 18*1E6 # in microseconds, may have to be adjusted
    df=df.copy(deep = True)
    tpos = skyhubtime(dfpos,datename,timename)
    tsens = skyhubtime(df,'Date','Time')
    latpos = dfpos[latname]
    tpos = tpos.astype('float')
    if timename == 'GPST':
        tpos = tpos-UTC_GPST_offset
    latint = np.interp(tsens.astype('float'),tpos,latpos)
    lonpos = dfpos[lonname]
    lonint = np.interp(tsens.astype('float'),tpos,lonpos)
    altpos = dfpos[altname]
    altint = np.interp(tsens.astype('float'),tpos,altpos)
    if anglesyn:
        pitchpos = dfpos['Pitch']
        pitchint = np.interp(tsens.astype('float'),tpos,pitchpos)
        rollpos = dfpos['Roll']
        rollint = np.interp(tsens.astype('float'),tpos,rollpos)
        df['Pitch interpolated'] = pitchint
        df['Roll interpolated'] = rollint
    df['Latitude interpolated'] = latint
    df['Longitude interpolated'] = lonint
    df['Altitude interpolated'] = altint
    df = df[np.logical_and(tsens.astype('float')>np.min(tpos),tsens.astype('float')<np.max(tpos))]
    return df

def raster_interp(x,y,n_raster,**kwargs): # interpolates any raster value to points, based on a raster input
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
    interp = interpolate.RegularGridInterpolator((-yg, xg), rasterv,method='linear')
    rasterint = interp(np.array([-np.array(y),x]).transpose())
    return rasterint

def hubdist(points_in,hubs,tol,chainage_field,**kwargs): # matches data points to a reference line
    Easting = 'Easting'
    Northing = 'Northing'
    if 'Easting' in kwargs:
        Easting = kwargs.get('Easting', None)
    if 'Northing' in kwargs:
        Northing = kwargs.get('Northing', None)
    points = points_in.copy(deep=True)
    nA = np.array((points_in[Easting],points_in[Northing])).transpose()
    nB = np.array((hubs['Easting'],hubs['Northing'])).transpose()
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    points['Hubs'] = hubs[chainage_field][idx].values
    points['Distance_Hubs'] = dist
    points = points[points['Distance_Hubs']<=tol]
    return points

def readsegy(filename): #areads full waveform data from WPR/Doppler
    skiptraceini = 0 # 
    skiptraceend = 0
    with segyio.open(filename,strict = False,endian = 'little') as f:
        traces = [np.copy(tr) for tr in f.trace[skiptraceini:-(skiptraceend +1)]]
        headers = [head for head in f.header[skiptraceini:-(skiptraceend +1)]]
        sample_rate = segyio.tools.dt(f) / 1000
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

def peakmodel2(v,mu1,mu2,s1,s2,f1,f2):
    ret_pow = f1*norm.pdf((v-mu1)/s1) + f2*norm.pdf((v-mu2)/s2)
    return ret_pow

def peakmodel1(v,mu1,s1,f1):
    ret_pow = f1*norm.pdf((v-mu1)/s1)
    return ret_pow