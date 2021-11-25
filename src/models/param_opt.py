import os
import lightgbm as lgb
import pandas as pd
from bayes_opt import BayesianOptimization

def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=3, n_folds=5, random_seed=6, n_estimators=10000, 
                            learning_rate=0.01, save_path = ''):

    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False, categorical_feature=[c for c in X if 'categorical' in c])

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, 
                 min_child_weight):
        
        params = {'application':'regression', 
                  'num_iterations': n_estimators, 
                  'learning_rate':learning_rate,
                  'early_stopping_round': 100, 
                  'metric':'rmse',
                  'verbose': -1,
                  'random_state': random_seed}
        
        params['num_leaves'] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, verbose_eval = 200, 
                           metrics=['rmse'], stratified = False)
        return -cv_result['rmse-mean'][-1]

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (15, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 18),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=47)

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    if (save_path != ''): 
        
        params_table = pd.DataFrame({'target': [lgbBO.res[c]['target'] for c in range(len(lgbBO.res))],
                                     'bagging_fraction': [lgbBO.res[c]['params']['bagging_fraction'] for c in range(len(lgbBO.res))],
                                     'feature_fraction': [lgbBO.res[c]['params']['feature_fraction'] for c in range(len(lgbBO.res))],
                                     'lambda_l1': [lgbBO.res[c]['params']['lambda_l1'] for c in range(len(lgbBO.res))],
                                     'lambda_l2': [lgbBO.res[c]['params']['lambda_l2'] for c in range(len(lgbBO.res))],
                                     'max_depth': [lgbBO.res[c]['params']['max_depth'] for c in range(len(lgbBO.res))],
                                     'min_child_weight': [lgbBO.res[c]['params']['min_child_weight'] for c in range(len(lgbBO.res))],
                                     'min_split_gain': [lgbBO.res[c]['params']['min_split_gain'] for c in range(len(lgbBO.res))],
                                     'num_leaves': [lgbBO.res[c]['params']['num_leaves'] for c in range(len(lgbBO.res))]}
                                   ).sort_values('target', ascending = False)
        
        params_table.to_csv(os.path.join(save_path, 'lgbBOparameters.csv'), index = False, sep = ';', decimal = ',')

    return lgbBO