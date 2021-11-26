import pandas as pd
import numpy as np

def extract_region_features(regions, regions_hcps):
    '''Renames static features about the regions'''
    
    # Rename columns
    regions_hcps.columns = ['region', 'region_area', 'region_pci16', 'region_pci18', 'region_n_intmedicine',
                            'region_n_intpneu', 'region_n_genpract', 'region_n_intmedandgen', 'region_n_pediat']
    regions_hcps['region_population'] = regions_hcps.region.map(dict(zip(regions.region, regions.population)))
    
    return regions_hcps