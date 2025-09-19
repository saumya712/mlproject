import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import Customexception
from src.logger import logging

from src.utils import save_object,evaluate_model


@dataclass
class datatransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
    
class datatransformation:
    def __init__(self):
        self.data_transformation_config=datatransformationconfig()
        
    def get_data_transformer_obj(self):
        try:
            numerical_feature=["writing score","reading score"]
            categorical_feature=["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
            
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )  
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("nummerical columns encoding complited")
            logging.info("categorical columns encoding complited")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipe",numerical_pipeline,numerical_feature),
                    ("cat_pipe",categorical_pipeline,categorical_feature)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise Customexception(e,sys)
        
    def initiate_data_transfomation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("the train and test path are read")
            
            logging.info("obtaining preprocessing obj")
            
            preprocessing_obj=self.get_data_transformer_obj()
            
            target_column="math score" 
            
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]
            
            logging.info(f"applying preproseccing object on training and testing dataset")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            
            logging.info("saved preprocessing object")
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise Customexception(e,sys)
            