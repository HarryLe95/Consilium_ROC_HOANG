import numpy as np 
import pandas as pd 

class ABC_Dataset:
    def extract_keyword(self, kwargs:dict) -> None:
        pass
    
    def get_dataset(self,start:str,end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> np.ndarray|pd.DataFrame|dict:
        pass 
        
class ABC_DataManager:
    def read_data(self) -> dict[str, pd.DataFrame]:
        pass 
    
    def write_data(self) -> None:
        pass 
    
    def update_metadata(self) -> None: 
        pass 

 