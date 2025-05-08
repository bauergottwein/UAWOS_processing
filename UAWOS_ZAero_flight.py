# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:55:27 2025

@author: vpk410
"""
        
class UAWOS_ZAreo_flight:
    def __init__(self, config_dict):
        print("UAWOS Zond Areo LF flight class constructor")
        for key, value in config_dict.items():
            setattr(self, key, value)
        
