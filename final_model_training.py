#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import yaml
from sklearn.model_selection import RandomizedSearchCV
from python_scripts.model_performance import generate_df_summary, predict_max_f1, save_model
pd.set_option('display.max_columns', None)


# In[ ]:


X_train = pd.read_parquet('data/X_train.parquet')
y_train = pd.read_csv('data/y_train.csv')
X_val = pd.read_parquet('data/X_val.parquet')
y_val = pd.read_csv('data/y_val.csv')


# ### final random search

# In[ ]:


xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='auc'
)

# inital param search
param_dist_new = {
    'learning_rate': [0.015, 0.025, 0.05, 0.06, 0.075, 0.1],
    'max_depth': [2, 3, 4, 5, 6],
    'min_child_weight': [4,5,6,7],
    'reg_lambda': [0.1, 1, 5, 10],
    'reg_alpha': [0, 0.1, 0.5, 1, 3],
    'scale_pos_weight': [5, 15, 25, 40, 50, 65],
    'max_delta_step': [0, 1, 3, 5],
    'gamma': [0.3, 0.5, 1, 1.5, 2],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.75],  

    'n_estimators': [1000],
    'early_stopping_rounds': [20],
}


# RandomizedSearchCV setup for parameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_new,
    n_iter=1000, 
    scoring='average_precision',  
    cv=5,  
    verbose=0,  
    random_state=0
)

# Fit the model (train and tune hyperparameters)
random_search.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)])

# Get the best model and print the results
best_model = random_search.best_estimator_

# Get the best iteration of the best model
best_iteration = best_model.best_iteration + 1

# Predictions and evaluation
y_pred = best_model.predict(X_val)


# In[ ]:


y_pred, best_threshold = predict_max_f1(best_model, X_val, y_val)
generate_df_summary(best_model, y_val, y_pred, 'final_random_search', threshold=best_threshold)
save_model(random_search, "saved_models/final_random_search")


# ### create yaml for best parameters

# In[ ]:


# get best threshold for max f1 of best model
model_performance = pd.read_csv('output/model_performance.csv')
best_threshold = float(model_performance[model_performance['Name'] == 'final_random_search'].proba_threshold.values[0])

# get best params from final random search
final_params = pd.read_csv('saved_models/saved_models/final_random_search/results.csv').loc[0][['param_subsample', 'param_scale_pos_weight',
       'param_reg_lambda', 'param_reg_alpha', 'param_n_estimators',
       'param_min_child_weight', 'param_max_depth', 'param_max_delta_step',
       'param_learning_rate', 'param_gamma',
       'param_colsample_bytree']].to_dict()

# replace default 1000 from random search with best iteration number of best model
final_params['param_n_estimators'] = round(best_iteration * 1.05)

cleaned_params = {
    key.replace('param_', ''): value
    for key, value in final_params.items()
}

to_yaml = {'hyperparameters': cleaned_params,
           'best_threshold': best_threshold,}

# save to yaml
import yaml
with open('model_settings.yaml', 'w') as file:
    yaml.dump(to_yaml, file)

print('Hyperparameters and optimised threshold saved to model_settings.yaml')
to_yaml


# ### retrain model on train and valiation sets combined

# In[ ]:


# original training set but with the validation set included
X_train_val = pd.read_parquet('data/X_train_val.parquet')
y_train_val = pd.read_csv('data/y_train_val.csv')

# get settings from yaml
with open('model_settings.yaml', 'r') as file:
    model_settings = yaml.safe_load(file)

params = model_settings['hyperparameters']

# final model
final_model = xgb.XGBClassifier(
    **params,  
    objective='binary:logistic',  
    eval_metric='auc',       
)

# fit final model
final_model.fit(X_train_val, y_train_val)

# save final model
final_model.save_model('saved_models/final_model/final_model.json')

print("Final model saved as final_model.json in saved_models/final_model/")


# In[ ]:




