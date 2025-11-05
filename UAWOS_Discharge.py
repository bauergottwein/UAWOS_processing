# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:13:43 2025

@author: ge86naf
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UAWOS_Utilities import skyhubtime, raster_interp, hubdist, posinterp
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit,minimize,root_scalar
import sys
import time
from dataclasses import dataclass

@dataclass
class ColumnData:
    vsurf: float
    md: float
    hl: float
    hu: float
    wss: float
    x: float
    
class UAWOS_Discharge:
    def __init__(self, config_dict):
        print("UAWOS Discharge class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
        # Some common global parameters
        self.g = 9.81
        self.size_T = 16 # Fontsize
        self.value_min = 1e-3 # minimum value for physical variables (avoid devided by 0 error)
        self.ks_x = np.linspace(1, 100.05, 50) # Range of stickler value, 1 - 100
        self.velocity_index = 0.85             # velocity index used for 0.85 method
        
        # WPR+DEM geometry file
        self.column_to_be_droped_wpr = ["Time stamp (UTC)", 'Antenna direction']
        self.cs_coordinate_column_wpr = 'Distance along reference line (m)'
        self.geometry_profile_column_wpr = 'z.' + self.z_EPSG + ' (m)'
        # -- bathymetry uncertainty parameters(optional)
        self.whether_geo_std_calculated_wpr = True
        self.geo_std_column_wpr = 'Std with 0.5 m interval (m)'
        
        # Doppler velocity file
        self.time_column_v = 'Time'
        self.rss_direction_column_v = 'RSS:Direction'
        self.cs_coordinate_column_v = 'Distance along reference line (m)'
        self.velocity_column_v = 'Two peak_full velocity (cm/s)'
        # -- Velocity uncertainty parameters(optional)
        self.whether_vel_std_calculated_v = True
        self.vel_std_column_v = 'Std with 5 traces (cm/s)_two peak'
        
        self.XS_info = {}
        self.XS_uncertainty = {}
        self.XS_uncertainty_L2 = {}
        if self.resultpath and not os.path.exists(self.resultpath):
            os.makedirs(self.resultpath)
   
    # --------- Main structure ---------- #    
    def process_Discharge(self):
        self.data_import()
        self.model_run()
        self.model_export()
        if self.if_calc_uncertainty == True:
            return self.XS_uncertainty_contribution
        else:
            return self.XS_info
    
    def data_import(self):
        self.full_geometry_import()
        self.velocity_import()
        
    def model_run(self):
        if self.if_calc_uncertainty == True:
            if not (self.whether_geo_std_calculated_wpr) or not (self.whether_geo_std_calculated_wpr):
                print(" Uncertainty information from Doppler or WPR is not provided!")
            else:
                self.processDischarge()
                self.uncertainty_sample()
                self.processDischarge_sigma()
        else:
            self.processDischarge()
            
    def model_export(self):
        self.generate_report()

#==============================================================================#
#=========== Model pool: include different alpha value algorithms =============#
#==============================================================================#
    
    def alpha_power(self, vsurf_i, md_i, Ks):
        m_factor = 1.37 * (8 * self.g / (md_i**(1/3) * Ks**2))**(-0.43)
        return m_factor / (m_factor + 1)

    def alpha_log(self, vsurf_i, md_i, Ks):
        z01 = (1 / max(0.06 * Ks, 1e-6))**7
        z02 = (1 / 2.705) * md_i * np.exp(-Ks * md_i**(1/6) / (2.03 * np.sqrt(8 * self.g)))
        z0 = min(z01, z02)
        md = md_i - z0
        return md_i / max(md, 1e-10) - 1 / (np.log(md_i / z0 + 1e-6))
    
    def alpha_entropy(self, vsurf_i, md_i, Ks):
        z01 = (1 / max(0.06 * Ks, 1e-6))**7
        z02 = (1 / 2.705) * md_i * np.exp(-Ks * md_i**(1/6) / (2.03 * np.sqrt(8 * self.g)))
        z0 = min(z01, z02)
        denominator = max((np.log(md_i / z0 + 1e-6)) / 0.41, 1e-10)
        alpha = (Ks * md_i**(1/6) / self.g**0.5) / denominator
        return min(alpha , 1)
        
#==============================================================================#
#==============================================================================#
#==============================================================================# 
    
     # --------- Model preprocessing ---------- #
    def full_geometry_import(self):
        bdata = pd.read_csv(self.full_geometry).groupby( self.cs_coordinate_column_wpr , as_index=False).mean()
        self.bcs = make_interp_spline(bdata[self.cs_coordinate_column_wpr], bdata[self.geometry_profile_column_wpr],k=1)
        bcs = self.bcs
        helpfunc = lambda bx: ( bcs(bx)-self.WSE )
        b_xaix = np.linspace(self.lbguess-5 , self.rbguess+5 , 1500)
        roots = []
        for i in range(len(b_xaix) - 1):
            try:
                sol = root_scalar(helpfunc, bracket=[b_xaix[i], b_xaix[i+1]], method='brentq')
                if sol.converged:
                    roots.append(sol.root)
            except ValueError:
                continue
        if len(roots) < 2:
            raise ValueError(f"Error: Expected at least 2 elements in 'roots', but got {len(roots)}, please check your lb and rb values")
        self.rb = roots[-1]
        self.lb = roots[0]
        self.npoints = int((self.rb-self.lb)/self.dx)+1
        
    def velocity_import(self):
        vdata = pd.read_csv(self.velocity)
        vdata =vdata.drop(columns=[ self.time_column_v ])
        vdata = vdata.loc[vdata[ self.rss_direction_column_v ] == -1]
        vdata['Surface velocity (cm/s)']=abs(vdata[ self.velocity_column_v ])
        vdata= vdata.groupby([ self.cs_coordinate_column_v ],as_index=False).mean()
        vdatax= np.array(vdata[ self.cs_coordinate_column_v ])
        vdatax = np.insert(vdatax , 0 , self.lb) # 0 velocity at the land-water boundary
        vdatax = np.insert(vdatax , len(vdatax) , self.rb) 
        vdatay= np.array(vdata['Surface velocity (cm/s)'])
        vdatay = np.insert(vdatay ,0,0)
        vdatay = np.insert(vdatay , len(vdatay) ,0)
        self.vcs = make_interp_spline( vdatax , vdatay , k=1)
        
        if self.whether_vel_std_calculated_v:
            vstdy = np.array(vdata[self.vel_std_column_v])
            vstdy = np.insert(vstdy ,0,0)
            vstdy = np.insert(vstdy , len(vstdy) ,0)
            self.vcs_std  = make_interp_spline(vdatax , vstdy , k=1)
        
    def uncertainty_sample(self):
        x_values = np.linspace(self.lb, self.rb, self.npoints)
        if self.whether_geo_std_calculated_wpr:
            bdata = pd.read_csv(self.full_geometry).groupby(self.cs_coordinate_column_wpr, as_index=False).mean()
            bdata_L3 = pd.read_csv(self.bathymetry).drop(columns=self.column_to_be_droped_wpr).groupby([self.cs_coordinate_column_wpr], as_index=False).mean()
            bdata['std'] = 0.0
            L3_std_map = dict(zip(bdata_L3[self.cs_coordinate_column_wpr], bdata_L3[self.geo_std_column_wpr]))
            bdata['std'] = bdata[self.cs_coordinate_column_wpr].map(L3_std_map).fillna(0)
            self.bcs_std  = make_interp_spline(bdata[self.cs_coordinate_column_wpr] , bdata['std'] ,k=1)
            bcs_samples = self.bcs(x_values)
            bcs_std_samples = self.bcs_std(x_values)
            nominal_wpr = bcs_samples
            wpr_err = np.array(bcs_std_samples)
            self.wpr_samples     = self._sample_field(nominal_wpr , wpr_err , self.sample_size)
        else:
            print("Uncertainty parameters are missing in the WPR Level3 input file, uncertainty calculation cannot be performed!")
            
        if self.whether_vel_std_calculated_v:
            vcs_samples = self.vcs(x_values)
            vcs_std_samples = self.vcs_std(x_values)
            nominal_doppler = vcs_samples
            doppler_err = np.array( vcs_std_samples )
            self.doppler_samples = self._sample_field(nominal_doppler , doppler_err , self.sample_size)
        else:
            print("Uncertainty parameters are missing in the Doppler Level3 input file, uncertainty calculation cannot be performed!")
        
        nominal_wse = self.WSE
        nominal_wse = np.full_like(np.array(nominal_wpr), nominal_wse)
        nominal_wss = self.WSS
        nominal_wss = np.full_like(np.array(nominal_wpr), nominal_wss)
        self.wse_samples     = self._sample_field(nominal_wse , self.WSE_sigma , self.sample_size)
        self.wss_samples     = self._sample_field(nominal_wss , self.WSS_sigma , self.sample_size)
        
    def _sample_field(self, nominal, err, size):
        nom = np.asarray(nominal)
        sd = np.abs(np.asarray(err))
        if nom.ndim == 0:
            nom = np.full(1, nom)
        if sd.ndim == 0:
            sd = np.full_like(nom, sd)
        return np.random.normal(loc=nom[:, None], scale=sd[:, None], size=(len(nom), size))

    # --------- Model simulation ---------- #
    def processDischarge(self):
        self.cs_column_split()
        self.XS_info = {}
        self.m_constant() # 0.85 method, can also applied to other velocity index
        self.solve_ks(self.ks_x, self.m_power, 'power')
        self.solve_ks(self.ks_x, self.m_log, 'log')
        if self.XS_info['WD_ratio'] < 5:
            print('Width depth ratio less than 5 -> dip activated')
            self.m_dip()
        else:
            print('Width depth ratio larger than 5 -> dip inactivated')
            self.solve_ks(self.ks_x, self.m_entropy, 'entropy')
    
    def processDischarge_sigma(self):        
        bcs_ini = self.bcs
        vcs_ini = self.vcs
        WSE_ini = self.WSE
        WSS_ini = self.WSS
        x_values = np.linspace(self.lb, self.rb, self.npoints)
        bcs = self.wpr_samples
        vcs = self.doppler_samples
        WSE = self.wse_samples
        WSS = self.wss_samples
        vars = ['bcs','vcs','WSS','WSE','all']
        var_names = ['bathymetry','velocity','WSS','WSE','all']
        metrics = ['Q_085','Ks_085','Q_log','Ks_log','Q_power','Ks_power','Q_entropy','Ks_entropy']
        results = {var: {m: [] for m in metrics} for var in vars} # raw results
        result_L2 = {var: {f"{m}_{stat}": [] for m in metrics for stat in ['mean', 'std']} for var in vars} # statistical results
        result_contribution = {var: {f"{m}_var": [] for m in metrics} for var in vars} # contribution to total uncertainty
        if self.XS_info['WD_ratio'] < 5:
            print('Width depth ratio less than 5 -> dip activated')
        else:
            print('Width depth ratio larger than 5 -> dip inactivated')
        for var , var_name in zip(vars , var_names):
            print(f"--------Processing variable: {var_name}--------")
            for i in range(self.sample_size):
                progress = (i + 1) / self.sample_size
                bar_length = 50
                bar = '#' * int(progress * bar_length) + '-' * (bar_length - int(progress * bar_length))
                sys.stdout.write(f"\r[{bar}] {progress * 100:.1f}% ({i+1}/{self.sample_size})")
                sys.stdout.flush()
                self.bcs = bcs_ini
                self.vcs = vcs_ini
                self.WSE = WSE_ini
                self.WSS = WSS_ini
                if var in ('bcs','all'):
                    self.bcs = make_interp_spline(x_values, bcs[:, i], k=1)
                if var in ('vcs','all'):
                    self.vcs = make_interp_spline(x_values, vcs[:, i], k=1)
                if var in ('WSE','all'):
                    self.WSE = make_interp_spline(x_values, WSE[:, i], k=1)
                if var in ('WSS','all'):
                    self.WSS = make_interp_spline(x_values, WSS[:, i], k=1)
                self.cs_column_split()
                self.XS_info = {}
                self.m_constant()
                self.solve_ks(self.ks_x, self.m_power,   'power')
                self.solve_ks(self.ks_x, self.m_log,     'log')
                if self.XS_info['WD_ratio'] < 5:
                    self.m_dip()
                else:
                    self.solve_ks(self.ks_x, self.m_entropy, 'entropy')
                res = results[var]
                for m in metrics:
                    res[m].append(self.XS_info[m])
            res_L2 = result_L2[var]
            for key in results[var].keys():
                res_L2[f'{key}_mean'] = np.nanmean(res[key])
                res_L2[f'{key}_std']  = np.nanstd(res[key])
            res_contribution = result_contribution[var]
            for key in results[var].keys():
                res_contribution[f'{key}_var'] = np.nanvar(res[key])
        self.XS_uncertainty = results
        self.XS_uncertainty_L2 = result_L2
        self.XS_uncertainty_contribution = result_contribution
        
    def solve_ks(self, ks_x, method_func, method_key):
        helpfunc = lambda Ks: self.m_manning(Ks) - method_func(Ks)
        ks_result = self.q_root(helpfunc, self.ks_x)
        if ks_result:
            Ks_solved = ks_result[0]
            Q_solved = method_func(Ks_solved)
            self.XS_info[f'Ks_{method_key}'] = Ks_solved
            self.XS_info[f'Q_{method_key}'] = Q_solved
        else:
            self.XS_info[f'Ks_{method_key}'] = np.nan
            self.XS_info[f'Q_{method_key}'] = np.nan

    def m_constant(self):
        Q = 0
        Area = 0
        WP = 0
        h = []
        v_bulk = []
        for col in self._column_iter():
            vbulk_i = self.velocity_index * col.vsurf
            v_bulk.append(vbulk_i)
            dQ = vbulk_i / 100 * self.dx * col.md
            dWP = np.sqrt((self.dx) ** 2 + (col.hl - col.hu) ** 2)
            Q = Q + dQ
            WP = WP + dWP
            Area += self.dx * col.md
            h.append(col.md)
        h_ave = np.nanmean(np.array(h))
        vbulk_ave = np.nanmean(np.array(v_bulk))/100
        hydraulic_radius = Area / WP
        width = abs(self.rb - self.lb)
        x_values = np.linspace(self.lb, self.rb, self.npoints)
        if callable(self.WSS):
            wss_values = self.WSS(x_values)
        else:
            wss_values = np.full_like(x_values, self.WSS)
        wss_mean = np.nanmean(wss_values)
        Ks = Q / (Area * hydraulic_radius**(2/3) * (wss_mean*1e-5)**(0.5))
        self.XS_info.update({
            "XS name" : self.XSid,
            "XS area" : Area,
            "Average depth" : h_ave,
            "Width" : width,
            "WD_ratio": width/h_ave, 
            "Hydraulic_radius" : hydraulic_radius,
            "Average bulk velocity" : vbulk_ave,
            "Froude" : vbulk_ave / np.sqrt(9.81 * h_ave),
            "Ks_085" : Ks,
            "Q_085" : Q
        }) 

    def m_power(self, Ks):
        return self.joint_calculator(Ks, self.alpha_power, 'alpha_power')

    def m_log(self, Ks):
        return self.joint_calculator(Ks, self.alpha_log, 'alpha_log')
    
    def m_entropy(self, Ks):
        return self.joint_calculator(Ks, self.alpha_entropy, 'alpha_entropy')
    
    def m_manning(self , Ks):
        Q=0
        for col in self._column_iter():
            wp_i = np.sqrt( (self.dx)**2 + (col.hl - col.hu)**2 )
            area_i = self.dx * col.md
            hr_i = area_i / wp_i # hydraulic radius
            dQ = Ks * area_i * (hr_i**(2/3)) * ((col.wss * 1e-5)**0.5)
            Q = Q + dQ
        return Q
    
    def m_dip(self):
        x_values = np.linspace(self.lb, self.rb, self.npoints)
        def phi_m(m):
            return (np.exp(m) / (np.exp(m) - 1)) - 1 / m
        def dip_h(m , H):
            return 0.2 * H * np.log(58.3 * m * phi_m(m) / (np.exp(m) - 1))
        def I_entropy(m, r): # eq8 in Farina 2014
            from scipy.integrate import quad
            func = lambda y: np.log(1 + (np.exp(m) - 1) * (y / r) * np.exp(1 - y / r))
            val, _ = quad(func, 0, 1)
            return val
        def mean_velocity_entropy(u_D , m , h , D): # eq7 in Farina 2014
            r = h / D
            I_val = I_entropy(m, r)
            denominator = np.log(1 + (np.exp(m) - 1) * (1 / (1 - r)) * np.exp(1 - 1 / (1 - r)))
            return u_D * I_val / denominator
        def v_surfmax_mean(m):
            v_surfmax = max(self.vsurf)
            md_vsurfmax = self.md[self.vsurf.index(v_surfmax)]
            r = dip_h(m, md_vsurfmax) / md_vsurfmax
            denominator = np.log(1 + (np.exp(m) - 1) * (1 / (1 - r)) * np.exp(1 - 1 / (1 - r)))
            v_max = v_surfmax * m / denominator
            return v_max * phi_m(m)
        def v_columns(m):
            Q = 0.0
            area_sum = 0.0
            for col in self._column_iter():
                h_dip = dip_h(m, col.md)
                u_mean = mean_velocity_entropy(col.vsurf, m, h_dip, col.md)
                dQ = u_mean * self.dx * col.md
                Q += np.nan_to_num(dQ, nan=0.0)
                area_sum += self.dx * col.md
            return Q / area_sum if area_sum > 0 else 0.0
        def diff_mean_velocity(m):
            try:
                return v_surfmax_mean(m) - v_columns(m)
            except (ValueError, ZeroDivisionError, OverflowError):
                return np.nan
        result = root_scalar(diff_mean_velocity, bracket=[0.5, 10.0], method='brentq')
        if result.converged:
            m_opt = result.root
            Q = 0.0
            Area = 0.0
            for col in self._column_iter():
                h_dip = dip_h(m_opt, col.md)
                u_mean = mean_velocity_entropy(col.vsurf, m_opt, h_dip, col.md)
                dQ = u_mean * self.dx * col.md
                Q += np.nan_to_num(dQ, nan=0.0)
                Area += self.dx * col.md
            self.XS_info[f'Q_entropy_dip'] = Q
            self.XS_info[f'alpha_entropy_dip_CS'] = phi_m(m_opt)
            if callable(self.WSS):
                wss_values = self.WSS(x_values)
            else:
                wss_values = np.full_like(x_values, self.WSS)
            wss_mean = np.nanmean(wss_values)
            self.XS_info[f'Ks_entropy_dip'] = Q / (Area * self.XS_info['Hydraulic_radius']**(2/3) * (wss_mean*1e-5)**(0.5))
        else:
            self.XS_info[f'Q_entropy_dip'] = None
            self.XS_info[f'alpha_entropy_dip_CS'] = None
            self.XS_info[f'Ks_entropy_dip'] = None
        
    def joint_calculator(self, Ks, alpha_func, alpha_key):
        Q = 0
        alphas = []
        v_surf_sum = 0
        area_sum = 0
        for col in self._column_iter():
            alpha = alpha_func(col.vsurf , col.md , Ks)
            alphas.append(alpha)
            vbulk_i = (alpha * col.vsurf / 100)
            dQ = vbulk_i * self.dx * col.md
            dQ = np.nan_to_num(dQ, nan=0)
            Q += dQ
            v_surf_sum += col.vsurf * self.dx * col.md / 100
            area_sum += self.dx * col.md
        v_bulk_avg = Q / area_sum if area_sum != 0 else np.nan
        v_surf_avg = v_surf_sum / area_sum if area_sum != 0 else np.nan
        alpha_total = v_bulk_avg / v_surf_avg if v_surf_avg != 0 else np.nan
        self.XS_info[alpha_key] = alphas
        self.XS_info[f'{alpha_key}_CS'] = alpha_total
        return Q
   
    def cs_column_split(self):
        self.vsurf = []
        self.md = []
        self.hl = [] # HL
        self.hu = [] # HU
        self.wss_values = []
        self.x = []
        for i in range(0 , self.npoints , 1):
            xmid = self.lb + self.dx*i + self.dx/2
            self.x.append(xmid)
            if callable(self.WSE):
                wse_l = self.WSE(xmid - self.dx/2)
                wse_u = self.WSE(xmid + self.dx/2)
            else:
                wse_l = wse_u = self.WSE
            dl = wse_l - self.bcs(xmid - self.dx/2)
            du = wse_u - self.bcs(xmid + self.dx/2)
            dl = np.maximum(dl , self.value_min)
            du = np.maximum(du , self.value_min)
            self.hl.append(dl)
            self.hu.append(du)
            self.md.append((dl + du)/2)
            if callable(self.vcs):
                vl = self.vcs(xmid - self.dx/2)
                vu = self.vcs(xmid + self.dx/2)
            else:
                vl = vu = self.vcs
            vl = np.maximum(vl , 0)
            vu = np.maximum(vu , 0)
            self.vsurf.append((vl + vu) / 2)
            if callable(self.WSS):
                self.wss_values.append( self.WSS(xmid) )
            else:
                self.wss_values.append( self.WSS )
    
    def q_root(self , helpfunc , xaix):
        roots = []
        for i in range(len(xaix) - 1):
            try:
                sol = root_scalar(helpfunc, bracket=[xaix[i], xaix[i+1]], method='brentq',rtol = 0.001)
                if sol.converged:
                    roots.append(sol.root)
            except ValueError:
                continue
        return roots
    
    def _column_iter(self):
        for vsurf, md, hl, hu, wss, x in zip(self.vsurf, self.md, self.hl, self.hu, self.wss_values, self.x):
            yield ColumnData(vsurf, md, hl, hu, wss, x)
    
    # --------- Model report ---------- #
    
    def generate_report(self):
        output_dir = self.resultpath
        if hasattr(self, 'XS_info'):
            raw = pd.DataFrame([self.XS_info])
            drop_cols = [c for c in ['alpha_entropy', 'alpha_power', 'alpha_log'] if c in raw.columns]
            if drop_cols:
                raw.drop(columns=drop_cols, inplace=True)
            raw.to_csv(self.resultpath + os.sep + self.XSid + '_result.csv', index=False)
            def _g(col):
                return raw[col].iloc[0] if col in raw.columns else np.nan
            df_properties = pd.DataFrame({
                'CS name'             : [_g('XS name')],
                'CS area [m2]'        : [_g('XS area')],
                'Width [m]'           : [_g('Width')],
                'CS depth [m]'        : [_g('Average depth')],
                'Hydraulic_radius [m]': [_g('Hydraulic_radius')],
            }).round(3)
            df_alpha = pd.DataFrame({
                '085 method'       : [0.85],
                'log law method'   : [_g('alpha_log_CS')],
                'power law method' : [_g('alpha_power_CS')],
                'entropy method'   : [_g('alpha_entropy_CS') if 'alpha_entropy_CS' in raw.columns else _g('alpha_entropy_dip_CS')],
            }).round(3)
            df_result = pd.DataFrame({
                'Q_085'      : [_g('Q_085')],
                'Ks_085'     : [_g('Ks_085')],
                'Q_log'      : [_g('Q_log')],
                'Ks_log'     : [_g('Ks_log')],
                'Q_power'    : [_g('Q_power')],
                'Ks_power'   : [_g('Ks_power')],
                'Q_entropy'  : [_g('Q_entropy') if 'Q_entropy' in raw.columns else _g('Q_entropy_dip')],
                'Ks_entropy' : [_g('Ks_entropy') if 'Ks_entropy' in raw.columns else _g('Ks_entropy_dip')],
            }).round(2)
            
        if self.if_calc_uncertainty and hasattr(self, 'XS_uncertainty_L2') and 'all' in self.XS_uncertainty_L2:
            df_full = pd.DataFrame(self.XS_uncertainty_L2['all'], index=[0]).T.reset_index()
            df_full.columns = ['Metric', 'Value']
            df_full.to_csv(self.resultpath + os.sep + self.XSid + '_ensemble.csv', index=False)
            
        print("-----------------------------------")
        print(f"Results generated: {output_dir}")
        print("-----------------------------------")