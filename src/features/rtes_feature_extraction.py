import pandas as pd
import numpy as np

def add_month(month):
    '''Given a string with month format of YYYY-MM add one month'''

    if (int(month[5:])<12):
        if (int(month[5:]) + 1 >= 10):
            return month[:5] + str(int(month[5:]) + 1)
        return month[:5] + '0' + str(int(month[5:]) + 1)
    return str(int(month[:4]) + 1) + '-01'

def extract_rtes_features(raw_rtes: pd.DataFrame) -> pd.DataFrame:
    raw_rtes=raw_rtes.fillna(0)
    raw_rtes['time_sent'] = pd.to_datetime(raw_rtes.time_sent, format='%Y-%m-%d')
    raw_rtes['month'] = raw_rtes['time_sent'].dt.strftime('%Y-%m')
    # 1 - total counts at a month-region-brand-level
    rtes_total = raw_rtes.groupby(['month', 'region', 
                               'brand']).agg({'hcp': 'count', 
                                              'no. openings': 'sum',
                                              'no. clicks': 'sum'
                                             }).reset_index()
    rtes_total.columns = ['month', 'region', 'brand', 'rtes_totalemails_1mo', 'rtes_totalopenings_1mo', 'rtes_totalclicks_1mo']
    

    # 2 - Breaking down the counts by type of email
    rtes_feature_type = raw_rtes.groupby(['month', 'region', 'brand',
                                                  'email_type']).size().unstack().reset_index().fillna(0)
    rtes_feature_type.columns = ['month', 'region', 'brand', 'rtes_typegeneralcount_brand_1mo', 'rtes_typeprodcount_brand_1mo', 'rtes_typeevent_brand_1mo']

    # 3 - Type of emails per month per region
    rtes_feature_type_month_region = raw_rtes.groupby(['month', 'region','email_type']).size().unstack().reset_index().fillna(0)
    rtes_feature_type_month_region.columns = ['month', 'region', 'rtes_typegeneralcount_1mo', 'rtes_typeprodcount_1mo', 'rtes_typeevent_1mo']

    
    # 4 - Types of emails per month per specialty
    raw_rtes['type_specialty'] = raw_rtes['email_type']+raw_rtes['specialty']
    rtes_feature_type_specialty = raw_rtes.groupby(['month', 'region', 'brand',
                                                  'type_specialty']).size().unstack().reset_index().fillna(0)
    rtes_feature_type_specialty.columns = ['month','region','brand','generalGP_region_brand_1mo','generalIM_region_brand_1mo','generalIMP_region_brand_1mo','generalIMGP_region_brand_1mo','generalPE_region_brand_1mo',
                                          'productGP_region_brand_1mo','productIM_region_brand_1mo','productIMP_region_brand_1mo','productIMGP_region_brand_1mo','productPE_region_brand_1mo','eventGP_region_brand_1mo','eventIM_region_brand_1mo']
    rtes_feature_type_specialty_month = rtes_feature_type_specialty.groupby(['month'])['generalGP_region_brand_1mo','generalIM_region_brand_1mo','generalIMP_region_brand_1mo','generalIMGP_region_brand_1mo','generalPE_region_brand_1mo',
                                      'productGP_region_brand_1mo','productIM_region_brand_1mo','productIMP_region_brand_1mo','productIMGP_region_brand_1mo','productPE_region_brand_1mo','eventGP_region_brand_1mo','eventIM_region_brand_1mo'].sum().reset_index()
    rtes_feature_type_specialty_month.columns = ['month','generalGP_1mo','generalIM_1mo','generalIMP_1mo','generalIMGP_1mo','generalPE_1mo',
                                          'productGP_1mo','productIM_1mo','productIMP_1mo','productIMGP_1mo','productPE_1mo','eventGP_1mo','eventIM_1mo']

    
    # 5 - Number of opening per month per region
    rtes_feature_nopen_month_region = raw_rtes.groupby(['month', 'region']).agg({'no. openings': 'sum'}).reset_index()
    rtes_feature_nopen_month_region.columns = ['month', 'region', 'rtes_nopening_region_1mo']

    # 6 - Number of opening per region
    rtes_feature_nopen_region = raw_rtes.groupby(['region']).agg({'no. openings': 'sum'}).reset_index()
    rtes_feature_nopen_region.columns = ['region','rtes_nopening_region']

    
    # 7 - Number of opening per month per region per specialty
    rtes_feature_nopen_specialty_region_month = raw_rtes.groupby(['month', 'region','specialty']).agg({'no. openings': 'sum'}).unstack().reset_index().fillna(0)
    rtes_feature_nopen_specialty_region_month.columns = ['month','region','rtes_nopen_GP_region_1mo','rtes_nopen_IM_region_1mo',
                                                       'rtes_nopen_IMP_region_1mo','rtes_nopen_IMGP_region_1mo','rtes_nopen_PE_region_1mo']
    final_rtes_features = rtes_total.copy()
    final_rtes_features = final_rtes_features.merge(rtes_feature_type, 
                                                    how = 'left', on = ['month', 'region', 'brand'])
    final_rtes_features = final_rtes_features.merge(rtes_feature_type_month_region, how = 'left', 
                                                   on = ['month', 'region'])
    final_rtes_features = final_rtes_features.merge(rtes_feature_type_specialty, how= 'left',
                                                   on = ['month', 'region', 'brand'])
    final_rtes_features = final_rtes_features.merge(rtes_feature_nopen_month_region, how = 'left',
                                                   on = ['month', 'region'])
    final_rtes_features = final_rtes_features.merge(rtes_feature_nopen_specialty_region_month, how= 'left',
                                                   on = ['month', 'region'])
    final_rtes_features.month = final_rtes_features.month.apply(lambda x: add_month(x))

    # Add rolling aggregations of the previously created features
    cols = [c for c in final_rtes_features.columns if c not in ['month', 'region', 'brand']]
    
    for c in cols:
        final_rtes_features[c[:-3] + '2mo'] = final_rtes_features.groupby(['region', 'brand'])[c].shift()
        final_rtes_features[c[:-3] + '3mo'] = final_rtes_features.groupby(['region', 'brand'])[c].shift(2)
        final_rtes_features[c[:-3] + '4mo'] = final_rtes_features.groupby(['region', 'brand'])[c].shift(3)
        
        final_rtes_features[ c[:-3] + 'trend_2mo'] = final_rtes_features[c[:-3] + '2mo'] / (final_rtes_features[c[:-3] + '1mo']+1)
        final_rtes_features[c[:-3] + 'trend_3mo'] = final_rtes_features[c[:-3] + '3mo'] / (final_rtes_features[c[:-3] + '1mo']+1)
        final_rtes_features[ c[:-3] + 'trend_4mo'] = final_rtes_features[c[:-3] + '4mo'] / (final_rtes_features[c[:-3] + '1mo']+1)

    months = pd.DataFrame({'month': raw_rtes.month.unique()})
    regions = pd.DataFrame({'region': raw_rtes.region.unique()})
    brands = pd.DataFrame({'brand': ['brand_1', 'brand_2']})

    months['dummy_col'] = 0
    regions['dummy_col'] = 0
    brands['dummy_col'] = 0

    final_rtes_features_full = months.merge(regions, how = 'outer', on = 'dummy_col')
    final_rtes_features_full= final_rtes_features_full.merge(brands, how = 'outer', on = 'dummy_col')
    final_rtes_features_full.drop(columns = 'dummy_col', inplace = True)
    final_rtes_features_full.sort_values('month', inplace = True)
    final_rtes_features = final_rtes_features.merge(final_rtes_features_full,how='outer',on=['month','region','brand'])
    final_rtes_features = final_rtes_features.fillna(0)
    return final_rtes_features
