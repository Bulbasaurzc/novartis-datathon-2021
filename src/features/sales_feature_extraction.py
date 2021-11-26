import pandas as pd
import numpy as np

def add_month(month):
    '''Given a string with month format of YYYY-MM add one month'''

    if (int(month[5:])<12):
        if (int(month[5:]) + 1 >= 10):
            return month[:5] + str(int(month[5:]) + 1)
        return month[:5] + '0' + str(int(month[5:]) + 1)
    return str(int(month[:4]) + 1) + '-01'


def extract_sales_features(raw_sales: pd.DataFrame) -> pd.DataFrame:
    
    '''Extracts lagged and window variables from the raw sales data at 
    different aggregation levels'''
    
    # Define all possible combinations of month - region - brand
    
    months = pd.DataFrame({'month': raw_sales.month.unique()})
    regions = pd.DataFrame({'region': raw_sales.region.unique()})
    brands = pd.DataFrame({'brand': ['brand_1', 'brand_2']})

    months['dummy_col'] = 0
    regions['dummy_col'] = 0
    brands['dummy_col'] = 0

    sales_features = months.merge(regions, how = 'outer', on = 'dummy_col')
    sales_features = sales_features.merge(brands, how = 'outer', on = 'dummy_col')
    sales_features.drop(columns = 'dummy_col', inplace = True)
    sales_features.sort_values('month', inplace = True)
    
    ### Universe sales of previous periods for the 150 regions in the train set at a month level
    
    train_regions = ['region_'+str(i) for i in range(151)]
    universe_features = raw_sales[raw_sales.region.isin(train_regions)].groupby(['month', 'brand']).sales.sum().unstack().reset_index()
    universe_features.month = universe_features.month.apply(lambda x: add_month(x))
    universe_features.columns = ['month', 'sales_univ_b1_1mo', 
                                 'sales_univ_b12_market_1mo',
                                 'sales_univ_b2_1mo', 'sales_univ_b3_1mo',
                                 'sales_univ_b3_market_1mo']
    universe_features['sales_univ_1mo'] = universe_features.sum(axis = 1)
    universe_features['sales_univ_market_1mo'] = (universe_features['sales_univ_b12_market_1mo'] + 
                                                  universe_features['sales_univ_b3_market_1mo'] )
    
    
    ### Universe sales of previous periods for the 150 regions in the train set at a month-brand level
    
    universe_brandsales = raw_sales[raw_sales.region.isin(train_regions)].groupby(['month', 'brand']).sales.sum().reset_index()
    universe_brandsales.month = universe_brandsales.month.apply(lambda x: add_month(x))
    universe_brandsales.columns = ['month', 'brand', 'sales_univ_brand_1mo']
    
    ### Brand 3 sales at the region
    sales3 = raw_sales.groupby(['month', 'region', 'brand']).sales.sum().unstack().reset_index()
    sales3 = sales3[['month', 'region', 'brand_3', 'brand_3_market', 'brand_12_market']]
    sales3.month = sales3.month.apply(lambda x: add_month(x))
    sales3.columns = ['month', 'region', 'sales_region_b3_1mo', 'sales_region_b3market_1mo', 'sales_region_b12market_1mo']

    ### 
    
    
    
    # Features scaled by overall market increase / decrease
    
    
    # Combine all sales features in a unique dataset
    
    sales_features = sales_features.merge(universe_features, how = 'left', on = 'month')
    sales_features = sales_features.merge(universe_brandsales, how = 'left', on = ['month', 'brand'])
    sales_features = sales_features.merge(sales3, how = 'left', on = ['month', 'region'])
    
    # Add rolling aggregations of the previously created features
    
    cols = [c for c in sales_features.columns if c not in ['month', 'region', 'brand']]
    
    for c in cols:
        sales_features[c[:-3] + '2mo'] = sales_features.groupby(['region', 'brand'])[c].shift()
        sales_features[c[:-3] + '3mo'] = sales_features.groupby(['region', 'brand'])[c].shift(2)
        sales_features[c[:-3] + '4mo'] = sales_features.groupby(['region', 'brand'])[c].shift(3)
        
        sales_features[ c[:-3] + 'trend_2mo'] = sales_features[c[:-3] + '2mo'] / (sales_features[c[:-3] + '1mo']+1)
        sales_features[c[:-3] + 'trend_3mo'] = sales_features[c[:-3] + '3mo'] / (sales_features[c[:-3] + '1mo']+1)
        sales_features[ c[:-3] + 'trend_4mo'] = sales_features[c[:-3] + '4mo'] / (sales_features[c[:-3] + '1mo']+1)
    
    return sales_features
    
    