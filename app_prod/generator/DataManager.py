from datetime import datetime, timedelta

import pandas as pd 
import numpy as np 

from generator.generic import ABC_DataManager, ABC_Dataset
from utils.advancedanalytics_util import AAUConnection
from generator.Dataset import Dataset

class DataManager(ABC_DataManager):
    def __init__(self,
                 wells:str, 
                 connection: AAUConnection,
                 run_mode:str='live',
                 backfill_start:str='2022-01-01',
                 backfill_date_format:str="%Y-%m-%d",
                 perform_model_training:bool=True,
                 perform_model_inference:bool=False,
                 path:str="ROC/PROCESSED_DATA", 
                 file_prefix:str="ROC_PROCESSED_DATA",
                 file_suffix:str=".csv",
                 **kwargs,
                 ):
        self.wells = wells  
        self.perform_model_training = perform_model_training 
        self.perform_model_inference = perform_model_inference
        self.run_mode = run_mode
        self.dataset = Dataset(connection, path, file_prefix, file_suffix, **kwargs)
        self.backfill_start = backfill_start 
        self.backfill_date_format = backfill_date_format
        self.metadata = self.get_metadata()
             
    def get_metadata(self) -> dict:
        if self.run_mode == "live":
            return self.dataset.get_metadata(self.wells)
        elif self.run_mode == "backfill": 
            try: 
                start_date = datetime.strptime(self.backfill_start, self.backfill_date_format)
                metadata_dict = {well: pd.DataFrame.from_dict({"TS":start_date}) for well in self.wells}
                return metadata_dict
            except Exception as e: 
                raise e 
    
    #TODO Update generate metadata response 
    def generate_metadata_response(self, inference_output:dict)->dict:
        response = {} 
        for well in self.wells: 
            if well in inference_output:
                response[well] = {} 
                response[well]['TS'] = inference_output[well]['TS']
            else: #TODO Log here 
                print(f"Error getting inference output for well {well}")
                response[well] = self.metadata[well]
                
    def update_metadata(self, inference_output:dict) -> None: 
        response = self.generate_metadata_response(inference_output)
        self.metadata = response 
        
    def get_inference_dataset(self) -> dict[str, np.ndarray]:
        inference_dataset = {} 
        for well in self.wells:
            start = datetime.strptime(self.metadata[well]['TS'].values[0], self.backfill_date_format).date()
            end = start + timedelta(days=1)
            inference_dataset.update(self.dataset.get_inference_dataset([well],start,end))
        return inference_dataset

if __name__ == "__main__":
    import config.__config__ as base_config
    config = base_config.init()
    from utils.advancedanalytics_util import aauconnect_
    manager = DataManager(['ACRUS1'],connection = aauconnect_(config['cfg_file_info']))
    manager.get_inference_dataset()
    print("end")