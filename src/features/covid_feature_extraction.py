import numpy as np
import pandas as pd

def add_month(month):
    '''Given a string with month format of YYYY-MM add one month'''

    if (int(month[5:])<12):
        if (int(month[5:]) + 1 >= 10):
            return month[:5] + str(int(month[5:]) + 1)
        return month[:5] + '0' + str(int(month[5:]) + 1)
    return str(int(month[:4]) + 1) + '-01'

def extract_covid_features(covid_data: pd.DataFrame) -> pd.DataFrame:
    
    covid_data['month'] = covid_data.date.apply(lambda x: x[:7])
    covid_features = covid_data.groupby('month').agg({'total_cases': 'sum', 'new_cases': 'sum', 'total_deaths': 'sum', 
                                     'new_deaths': 'sum', 'new_tests': 'sum', 'total_tests': 'sum',
                                     'people_vaccinated': 'sum', 'new_vaccinations': 'sum'
                                    }).reset_index()
    covid_features.columns = ['month', 'covid_cases_1mo', 'covid_newcases_1mo', 'covid_deaths_1mo', 'covid_newdeaths_1mo',
                         'covid_newtests_1mo', 'covid_tests_1mo', 'covid_peoplevacc_1mo', 'covid_newvacc_1mo']
    
    covid_features.month = covid_features.month.apply(lambda x: add_month(x))

    return covid_features