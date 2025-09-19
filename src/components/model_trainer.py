import os
import sys
from dataclasses import dataclass

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
from src.utils import save_object,evaluate_model

@dataclass
class model_trainer_config:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")
    
class modeltrainer:
    def __init__(self):
        self.modeltrainerconfig=model_trainer_config()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting the training and test input data")
            
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                "random forest":RandomForestRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "gradienr boosting":GradientBoostingRegressor(),
                "linear regresion":LinearRegression(),
                "xgb":XGBRegressor(),
                "catboost":CatBoostRegressor(),
                "adaboost":AdaBoostRegressor()
            }
            
            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)
            
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            save_object(file_path=self.modeltrainerconfig.trained_model_file_path,obj=best_model)
            
            predicted=best_model.predict(X_test)
            
            r2_square = r2_score(Y_test,predicted)
            
            return r2_square
        except Exception as e:
            raise Customexception(e,sys)