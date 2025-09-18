import os
import sys
from src.exception import Customexception
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.datatransformation import datatransformation
from src.components.datatransformation import datatransformationconfig

@dataclass
class dataingestionconfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    
class dataingestion:
    def __init__(self):
        self.ingestion_config=dataingestionconfig()
        
    def initiate_data_ingestion(self):
        logging.info("entered the data ingetion method")
        try:
            df=pd.read_csv('nootbbok\\StudentsPerformance.csv')
            logging.info("read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("train test split initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise Customexception(e,sys)
        
if __name__=="__main__":
            obj=dataingestion()
            train_data,tesr_data=obj.initiate_data_ingestion()
            
            data_transformation=datatransformation()
            data_transformation.initiate_data_transfomation(train_data,tesr_data)