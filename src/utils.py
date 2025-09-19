import os
import sys

import numpy as np
import pandas as pd

from src.exception import Customexception
import dill

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import Customexception
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise Customexception(e,sys)
    
    
def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            
            model.fit(X_train,Y_train)
            
            Y_train_pred=model.predict(X_train)
            
            Y_test_pred=model.predict(X_test)
            
            train_model_score=r2_score(Y_train,Y_train_pred)
            
            test_model_score=r2_score(Y_test,Y_test_pred)
            
            report[list(models.keys())[i]]=test_model_score
            
        return report
    
    except Exception as e:
        raise Customexception(e,sys)