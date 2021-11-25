import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_model(X, target_name = 'y', split = 'in_sample', model_type = 'lgb', params = 'default', 
                objective = 'binary', metric = 'auc', save_path = ''):
    '''returns model and saves log parameters'''
    
    if (split == 'in_sample'):
        X_tr, X_val, y_train, y_val = train_test_split(X.drop(columns = target_name), 
                                                       X[target_name], 
                                                       test_size = 0.1, random_state = 1926)
        
    if (split != 'in_sample'):
        X_tr = X[X['train'] == 1].drop(columns = [target_name, 'train'])
        X_val = X[X['train'] != 1].drop(columns = [target_name, 'train'])
        y_train = X[X['train'] == 1][target_name]
        y_val = X[X['train'] != 1][target_name]
    
    if (params == 'default'):
        params = {'num_leaves': 54, 'min_data_in_leaf': 79, 'objective': objective,
                  'max_depth': 3, 'learning_rate': 0.01, 'boosting': 'gbdt', 'feature_fraction': 1,
                  'bagging_freq': 5, 'bagging_fraction': 0.9, 'bagging_seed': 11, 'metric': metric, 'lambda_l1': 0.1,
                  'verbosity': -1, 'min_child_weight': 5, 'reg_alpha': 3, 'reg_lambda': 2, 'subsample': 0.8,'seed': 1926}
        
    if (model_type == 'lgb'):
        X_train_lgb = lgb.Dataset(X_tr, label = y_train)
        X_val_lgb = lgb.Dataset(X_val, label = y_val)
        
        print('#'*20 + ' '*5 + 'training with ', X_tr.shape[0], ' '*5 + '#'*20)
        print('#'*20 + ' '*5 + 'validating with ', X_val.shape[0], ' '*5 + '#'*20)
        
        model = lgb.train(params, 
                          X_train_lgb,
                          num_boost_round = 1000,
                          valid_sets = [X_train_lgb, X_val_lgb],
                          early_stopping_rounds = 20)
    if (save_path != ''):
        with open(save_path + '/_model.pickle', 'wb') as pfile:
            pickle.dump(model, pfile, protocol = pickle.HIGHEST_PROTOCOL)
    
    return model