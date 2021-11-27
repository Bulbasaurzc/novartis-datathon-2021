import pandas as pd
import numpy as np

def extract_region_features(regions, regions_hcps, hcps):
    '''Extracts region level static features from specialty numbers and HCP'''
    
    # Rename columns
    regions_hcps.columns = ['region', 'region_area', 'region_pci16', 'region_pci18', 'region_n_intmedicine',
                            'region_n_intpneu', 'region_n_genpract', 'region_n_intmedandgen', 'region_n_pediat']
    regions_hcps['region_population'] = regions_hcps.region.map(dict(zip(regions.region, regions.population)))
    
    regions_hcps['region_totaltypes'] = regions_hcps[['region_n_intmedicine', 'region_n_intpneu', 'region_n_genpract', 
                                                      'region_n_intmedandgen', 'region_n_pediat']].sum(axis = 1)
    
    regions_hcps['region_density'] = regions_hcps['region_population'] / (regions_hcps['region_area'] + 1)
    
    cols = ['region_pci16', 'region_pci18', 'region_n_intmedicine',
            'region_n_intpneu', 'region_n_genpract', 'region_n_intmedandgen', 'region_n_pediat', 'region_totaltypes']
    
    for col in cols:
        regions_hcps[col + '_perperson'] = regions_hcps[col] / (regions_hcps['region_population'] + 1)
        
        
        
    hcps_features = hcps.groupby('region').specialty.nunique().reset_index()
    hcps_features.columns = ['region', 'region_uniquespecialties']
    hcps_features['region_avgtier'] = hcps_features.region.map(hcps.groupby('region').tier.mean())
    hcps_features['region_mintier'] = hcps_features.region.map(hcps.groupby('region').tier.min())
    hcps_features['region_maxtier'] = hcps_features.region.map(hcps.groupby('region').tier.max())
    hcps_features['region_stdtier'] = hcps_features.region.map(hcps.groupby('region').tier.std())

    n_hcp = hcps.groupby(['region', 'specialty']).hcp.sum().unstack().reset_index()
    n_hcp.columns = ['region', 'region_nhcp_genprac', 'region_nhcp_intmed', 'region_nhcp_pneumol', 
                     'region_nhcp_intgen','region_nhcp_pediat']

    n_hcp2 = hcps[hcps.tier == 2].groupby(['region', 'specialty']).hcp.sum().unstack().reset_index()
    n_hcp2.columns = ['region', 'region_nhcp_genprac2', 'region_nhcp_intmed2', 'region_nhcp_pneumol2', 
                     'region_nhcp_intgen2','region_nhcp_pediat2']

    n_hcp3 = hcps[hcps.tier == 3].groupby(['region', 'specialty']).hcp.sum().unstack().reset_index()
    n_hcp3.columns = ['region', 'region_nhcp_genprac3', 'region_nhcp_intmed3', 'region_nhcp_pneumol3', 'region_nhcp_pediat3']

    hcps_features = hcps_features.merge(n_hcp, how = 'left', on = 'region')
    hcps_features = hcps_features.merge(n_hcp2, how = 'left', on = 'region')
    hcps_features = hcps_features.merge(n_hcp3, how = 'left', on = 'region')

    hcps_features.fillna(0, inplace = True)
    
    regions_hcps = regions_hcps.merge(hcps_features, how = 'left', on = 'region')

    return regions_hcps