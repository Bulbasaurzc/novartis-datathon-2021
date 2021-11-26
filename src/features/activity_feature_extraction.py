import pandas as pd
import numpy as np

def add_month(month):
    '''Given a string with month format of YYYY-MM add one month'''

    if (int(month[5:])<12):
        if (int(month[5:]) + 1 >= 10):
            return month[:5] + str(int(month[5:]) + 1)
        return month[:5] + '0' + str(int(month[5:]) + 1)
    return str(int(month[:4]) + 1) + '-01'

def extract_activity_features(raw_activity: pd.DataFrame) -> pd.DataFrame:
    raw_activity = raw_activity.rename(columns = {'count':'counts'})
    raw_activity.sort_values('month', inplace = True)
    
    #activities to different specialty per month
    activity_features_specialty = raw_activity.groupby(['month', 'region', 'brand', 
                                              'specialty']).counts.sum().unstack().reset_index().fillna(0)
    activity_features_specialty['act_total_1mo'] = activity_features_specialty.sum(axis = 1)
    activity_features_specialty['act_GP_1mo'] = activity_features_specialty.month.map(
        activity_features_specialty.groupby('month')['General practicioner'].sum())
    activity_features_specialty['act_IM_1mo'] = activity_features_specialty.month.map(
        activity_features_specialty.groupby('month')['Internal medicine'].sum())
    activity_features_specialty['act_IMP_1mo'] = activity_features_specialty.month.map(
        activity_features_specialty.groupby('month')['Internal medicine / pneumology'].sum())
    activity_features_specialty['act_IMGP_1mo'] = activity_features_specialty.month.map(
        activity_features_specialty.groupby('month')['Internal medicine and general practicioner'].sum())
    activity_features_specialty['act_PE_1mo'] = activity_features_specialty.month.map(
        activity_features_specialty.groupby('month')['Pediatrician'].sum())
    
    #activities via different channel per month
    activity_features_channel = raw_activity.groupby(['month', 'region', 'brand', 
                                              'channel']).counts.sum().unstack().reset_index().fillna(0)
    activity_features_channel['act_f2f_1mo'] = activity_features_channel.month.map(
        activity_features_channel.groupby('month')['f2f'].sum())
    activity_features_channel['act_other_1mo'] = activity_features_channel.month.map(
        activity_features_channel.groupby('month')['other'].sum())
    activity_features_channel['act_phone_1mo'] = activity_features_channel.month.map(
        activity_features_channel.groupby('month')['phone'].sum())
    activity_features_channel['act_video_1mo'] = activity_features_channel.month.map(
        activity_features_channel.groupby('month')['video'].sum())
    activity_features_channel

    #activity to different specialty per month per region
    GP_monthly = activity_features_specialty.groupby(['month','region'])['General practicioner'].sum().reset_index()
    GP_monthly.columns = ['month', 'region', 'act_GP_region_1mo']
    IM_monthly = activity_features_specialty.groupby(['month','region'])['Internal medicine'].sum().reset_index()
    IM_monthly.columns = ['month', 'region', 'act_IM_region_1mo']
    IMP_monthly = activity_features_specialty.groupby(['month','region'])['Internal medicine / pneumology'].sum().reset_index()
    IMP_monthly.columns = ['month', 'region', 'act_IMP_region_1mo']
    IMGP_monthly = activity_features_specialty.groupby(['month','region'])['Internal medicine and general practicioner'].sum().reset_index()
    IMGP_monthly.columns = ['month', 'region', 'act_IMGP_region_1mo']
    PE_monthly = activity_features_specialty.groupby(['month','region'])['Pediatrician'].sum().reset_index()
    PE_monthly.columns = ['month', 'region', 'act_PE_region_1mo']

    activity_features_specialty = activity_features_specialty.merge(GP_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_specialty = activity_features_specialty.merge(IM_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_specialty = activity_features_specialty.merge(IMP_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_specialty = activity_features_specialty.merge(IMGP_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_specialty = activity_features_specialty.merge(PE_monthly, how = 'outer', on = ['month', 'region'])

    #activity to different specialty per month per brand
    GP_monthly_brand = activity_features_specialty.groupby(['month','brand'])['General practicioner'].sum().reset_index()
    GP_monthly_brand.columns = ['month', 'brand', 'act_GP_brand_1mo']
    IM_monthly_brand = activity_features_specialty.groupby(['month','brand'])['Internal medicine'].sum().reset_index()
    IM_monthly_brand.columns = ['month', 'brand', 'act_IM_brand_1mo']
    IMP_monthly_brand = activity_features_specialty.groupby(['month','brand'])['Internal medicine / pneumology'].sum().reset_index()
    IMP_monthly_brand.columns = ['month', 'brand', 'act_IMP_brand_1mo']
    IMGP_monthly_brand = activity_features_specialty.groupby(['month','brand'])['Internal medicine and general practicioner'].sum().reset_index()
    IMGP_monthly_brand.columns = ['month', 'brand', 'act_IMGP_brand_1mo']
    PE_monthly_brand = activity_features_specialty.groupby(['month','brand'])['Pediatrician'].sum().reset_index()
    PE_monthly_brand.columns = ['month', 'brand', 'act_PE_brand_1mo']

    activity_features_specialty = activity_features_specialty.merge(GP_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_specialty = activity_features_specialty.merge(IM_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_specialty = activity_features_specialty.merge(IMP_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_specialty = activity_features_specialty.merge(IMGP_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_specialty = activity_features_specialty.merge(PE_monthly_brand, how = 'outer', on = ['month', 'brand'])

    #activity via different channel per month per region
    f2f_monthly = activity_features_channel.groupby(['month','region'])['f2f'].sum().reset_index()
    f2f_monthly.columns = ['month', 'region', 'act_f2f_region_1mo']
    phone_monthly = activity_features_channel.groupby(['month','region'])['phone'].sum().reset_index()
    phone_monthly.columns = ['month', 'region', 'act_phone_region_1mo']
    video_monthly = activity_features_channel.groupby(['month','region'])['video'].sum().reset_index()
    video_monthly.columns = ['month', 'region', 'act_video_region_1mo']
    other_monthly = activity_features_channel.groupby(['month','region'])['other'].sum().reset_index()
    other_monthly.columns = ['month', 'region', 'act_other_region_1mo']

    activity_features_channel = activity_features_channel.merge(f2f_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_channel = activity_features_channel.merge(phone_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_channel = activity_features_channel.merge(video_monthly, how = 'outer', on = ['month', 'region'])
    activity_features_channel = activity_features_channel.merge(other_monthly, how = 'outer', on = ['month', 'region'])



    #activity via different channel per month per brand
    f2f_monthly_brand = activity_features_channel.groupby(['month','brand'])['f2f'].sum().reset_index()
    f2f_monthly_brand.columns = ['month', 'brand', 'act_f2f_brand_1mo']
    phone_monthly_brand = activity_features_channel.groupby(['month','brand'])['phone'].sum().reset_index()
    phone_monthly_brand.columns = ['month', 'brand', 'act_phone_brand_1mo']
    video_monthly_brand = activity_features_channel.groupby(['month','brand'])['video'].sum().reset_index()
    video_monthly_brand.columns = ['month', 'brand', 'act_video_brand_1mo']
    other_monthly_brand = activity_features_channel.groupby(['month','brand'])['other'].sum().reset_index()
    other_monthly_brand.columns = ['month', 'brand', 'act_other_brand_1mo']

    activity_features_channel = activity_features_channel.merge(f2f_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_channel = activity_features_channel.merge(phone_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_channel = activity_features_channel.merge(video_monthly_brand, how = 'outer', on = ['month', 'brand'])
    activity_features_channel = activity_features_channel.merge(other_monthly_brand, how = 'outer', on = ['month', 'brand'])

    
    activity_features = activity_features_channel.merge(activity_features_specialty, 
                                                    how = 'outer', on = ['month', 'region','brand'])


    activity_features=activity_features.rename(columns = {'f2f':'act_f2f_region_brand_1mo', 'other':'act_other_region_brand_1mo',
                                                     'phone':'act_phone_region_brand_1mo','video':'act_video_region_brand_1mo',
                                                      'General practicioner':'act_GP_region_brand_1mo', 'Internal medicine':'act_IM_region_brand_1mo',
                                                     'Internal medicine / pneumology':'act_IMP_region_brand_1mo',
                                                     'Internal medicine and general practicioner':'act_IMPG_region_brand_1mo',
                                                     'Pediatrician':'act_PE_region_brand_1mo'})
    
    activity_features.month = activity_features.month.apply(lambda x: add_month(x))
    # Add rolling aggregations of the previously created features
    
    cols = [c for c in activity_features.columns if c not in ['month', 'region', 'brand']]
    
    for c in cols:
        activity_features[c[:-3] + '2mo'] = activity_features.groupby(['region', 'brand'])[c].shift()
        activity_features[c[:-3] + '3mo'] = activity_features.groupby(['region', 'brand'])[c].shift(2)
        activity_features[c[:-3] + '4mo'] = activity_features.groupby(['region', 'brand'])[c].shift(3)
        
        activity_features[ c[:-3] + 'trend_2mo'] = activity_features[c[:-3] + '2mo'] / (activity_features[c[:-3] + '1mo']+1)
        activity_features[c[:-3] + 'trend_3mo'] = activity_features[c[:-3] + '3mo'] / (activity_features[c[:-3] + '1mo']+1)
        activity_features[ c[:-3] + 'trend_4mo'] = activity_features[c[:-3] + '4mo'] / (activity_features[c[:-3] + '1mo']+1)
    
    return activity_features