from datetime import datetime, timedelta
from typing import Sequence
import logging 

import pandas as pd 
import numpy as np 

from Dataset.generic import ABC_DataManager
from Dataset.Dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DataManager(ABC_DataManager):    
    def __init__(self,
                 wells:Sequence[str], 
                 run_mode:str,
                 backfill_start:str,
                 dataset:Dataset,
                 backfill_date_format:str="%Y-%m-%d %H:%M",
                 inference_window:int=30,
                 perform_model_training:bool=True,
                 perform_model_inference:bool=False,
                 datetime_index_column:str="TS",
                 **kwargs,
                 ):
        self.wells = wells  
        self.perform_model_training = perform_model_training 
        self.perform_model_inference = perform_model_inference
        self.run_mode = run_mode
        self.dataset = dataset 
        self.backfill_start = backfill_start 
        self.backfill_date_format = backfill_date_format
        self.datetime_index_column = datetime_index_column
        self.inference_window = inference_window
        self.metadata = self.get_metadata()
             
    def get_metadata(self) -> dict[str,pd.DataFrame]:
        """Get medata data based on run mode.
        
        live mode: get latest metadata 
        backfill mode: create an artificial metadata with chosen backfill_start date 
    
        Raises:
            e: _description_

        Returns:
            dict[str,pd.DataFrame] - dictionary of well_code:metadata_dataframe
        """
        if self.run_mode == "live":
            return self.dataset.get_metadata(self.wells)
        elif self.run_mode == "backfill": 
            try: 
                metadata_dict = {well: pd.DataFrame.from_dict({self.datetime_index_column:[self.backfill_start]}) for well in self.wells}
                return metadata_dict
            except Exception as e: 
                raise e 
    
    #TODO update this later 
    def update_metadata(self, inference_output:dict) -> None: 
        """Update metadata based on inference output

        Args:
            inference_output (dict): _description_
        """
        for well in inference_output: 
            status = inference_output[well]['status']
            if status == 0: 
                current_date = datetime.strptime(self.metadata[well][self.datetime_index_column][0],self.backfill_date_format)
                next_date = current_date + timedelta(days=1)
                self.metadata[well][self.datetime_index_column] = next_date.strftime(self.backfill_date_format)
    
    #TODO update this later   
    def write_metadata(self):
        pass 
        
    def get_inference_dataset(self) -> dict[str, np.ndarray]:
        """Get inference dataset 
        
        Returns:
            dict[str, np.ndarray]: dictionary of inference data 
        """
        inference_dataset = {} 
        for well in self.wells:
            inf_start = datetime.strptime(self.metadata[well][self.datetime_index_column].values[0], self.backfill_date_format).date()
            window_start = inf_start - timedelta(days=self.inference_window)
            window_end = inf_start + timedelta(days=1)
            inference_dataset.update(self.dataset.get_wells_dataset([well],str(window_start),str(window_end)))
        return inference_dataset

    #TODO 
    def get_training_dataset(self) -> dict[str, np.ndarray]:
        pass 
