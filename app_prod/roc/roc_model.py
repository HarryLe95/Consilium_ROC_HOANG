import pandas as pd 
import numpy as np 
from typing import Sequence 

class Classification_Model:
    def fit(self, training_data):
        pass
    
    def predict(self, inference_data):
        pass 
    
class Dummy_Classification_Model(Classification_Model):
    def __init__(self):
        pass 
    
    def fit(self,training_data:pd.DataFrame):
        pass 
    
    def predict(self,inference_data:pd.DataFrame):
        n_samples = inference_data.shape[0]
        return np.random.randint(low=0,high=10,size=n_samples)
    
class Regression_Model:
    def fit(self, training_data):
        pass 
    
    def predict(self, inference_data:pd.DataFrame) -> np.ndarray:
        pass 
    
class Dummy_Regression_Model(Regression_Model):
    def __init__(self):
        pass 
    
    def fit(self,training_data:pd.DataFrame):
        pass 
        
    def predict(self, inference_data: pd.DataFrame):
        n_samples = inference_data.shape[0]
        return np.random.randint(low=0,high=30,size=n_samples)
    
class Feature_Extractor_Model:  
    @classmethod 
    def get_well_status(cls, inference_data:pd.DataFrame):
        def _get_flow_status(x:Sequence[float]):
            x = np.array(x)
            no_flow = np.all(x[int(len(x)/2):]==0)    
            return "SI" if no_flow else "Online"
        return inference_data.FLOW.apply(lambda x: _get_flow_status(x))
    
    @classmethod 
    def get_downtime(cls, inference_data:pd.DataFrame):
        def _get_percentage_downtime(x:Sequence[float]):
            x = np.array(x)
            return len(x[x==0])/len(x)
        return inference_data.FLOW.apply(lambda x: _get_percentage_downtime(x)) 
    
    @classmethod
    def get_max_volt(cls, inference_data:pd.DataFrame):
        def _get_max_volt(x:Sequence[float]):
            return max(x)
        return inference_data.ROC_VOLTAGE.apply(lambda x: _get_max_volt(x))
    
    @classmethod 
    def get_min_volt(cls, inference_data:pd.DataFrame):
        def _get_min_volt(x:Sequence[float]):
            x = np.array(x)
            x_ = x[x!=0]
            return 0 if len(x_) == 0 else min(x_)
        return inference_data.ROC_VOLTAGE.apply(lambda x: _get_min_volt(x))
    
    @classmethod 
    def get_charge_volt(cls,inference_data:pd.DataFrame):
        return cls.get_max_volt(inference_data) - cls.get_min_volt(inference_data)