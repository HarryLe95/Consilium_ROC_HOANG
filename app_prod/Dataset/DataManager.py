from datetime import datetime, timedelta
from typing import Sequence
import logging 

import pandas as pd 
import numpy as np 

from Dataset.Dataset import DataOperator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DataManager: 
    def __init__(self,
                 wells:Sequence[str], 
                 run_mode:str,
                 backfill_start:str,
                 data_operator:DataOperator,
                 inference_window:int=30,
                 datetime_index_column:str="TS",
                 **kwargs,
                 ):
        """Metaclass that manages dataset functionalities. Primary methods include: 
    
        Methods:
            get_metadata: get metadata information for all inference wells 
            update_metadata: update metadata based on inference response 
            get_training_dataset: get training dataset 
            get_inference_dataset: get a dictionary of {well_name: well_inf_df}

        Args:
            wells (Sequence[str]): inference wells
            run_mode (str): inference run mode. One of ["live", "backfill"]
            backfill_start (str): backfill start date if run_mode is "backfill"
            data_operator (Dataset): dataset operator that handles read/write
            inference_window (int, optional): number of previous days of supporting data for inference. Defaults to 30.
            datetime_index_column (str, optional): column name to be interpreted as datetime index. Defaults to "TS".
        """
        self.wells = wells  
        self.run_mode = run_mode
        self.data_operator = data_operator
        self.backfill_start = backfill_start 
        self.datetime_index_column = datetime_index_column
        self.inference_window = inference_window
        self.metadata = self.get_metadata()
             
    def get_metadata(self) -> dict[str,pd.DataFrame]:
        """Get medata data based on run mode.
        
        live mode: get latest metadata 
        backfill mode: create an artificial metadata with chosen backfill_start date 
        
        Returns:
            dict[str,pd.DataFrame] - dictionary of {well_code:metadata_dataframe}
        """
        if self.run_mode == "live":
            logger.debug("Getting metadata in live mode.")
            return self.data_operator.get_metadata(self.wells)
        elif self.run_mode == "backfill": 
            try:
                logger.debug("Getting metadata in backfill mode.") 
                metadata_dict = {well: pd.DataFrame.from_dict({self.datetime_index_column:[self.backfill_start]}) for well in self.wells}
                return metadata_dict
            except Exception as e: 
                logger.error(f"Error getting backfill mode metadata. Error Message: {e}")
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
                current_date = datetime.strptime(self.metadata[well][self.datetime_index_column][0],"%Y-%m-%d %H:%M")
                next_date = current_date + timedelta(days=1)
                self.metadata[well][self.datetime_index_column] = next_date.strftime("%Y-%m-%d %H:%M")
    
    def get_inference_day_dataset_(self, well:str) -> tuple[datetime.date,pd.DataFrame]|None:
        """Get data for the inference day. Inference day is the last day (day in the last row) of the corresponding metadata

        Args:
            well (str): well code 

        Returns:
            tuple[datetime.date,pd.DataFrame]|None: if data is insufficient, return None. Otherwise return the data of the inference day.
        """
        inf_start = datetime.strptime(self.metadata[well][self.datetime_index_column].values[-1], "%Y-%m-%d %H:%M").date()
        inf_end = inf_start + timedelta(days=1)
        inf_data = self.data_operator.read_data(well, inf_start, inf_end)
        if len(inf_data)<1440:
            logger.error(f"Insufficient inference data for well: {well}")
            return None
        return (inf_start,inf_data)
            
    def get_inference_dataset(self) -> dict[str, np.ndarray]:
        """Get inference dataset 
        
        Returns:
            dict[str, np.ndarray]: dictionary of inference data 
        """
        inference_dataset = {} 
        for well in self.wells:
            inference_day_data = self.get_inference_day_dataset_(well)
            if inference_day_data is not None:
                inf_start, inf_day_df = inference_day_data
                window_start = inf_start - timedelta(days=self.inference_window-1)
                window_end = inf_start - timedelta(days=1)
                supp_day_df = self.data_operator.read_data(well, window_start, window_end)
                inf_df = pd.concat([supp_day_df, inf_day_df],axis=0)
                inference_dataset[well] = self.data_operator.process_data(inf_df)
            else:
                inference_dataset[well] = None
        return inference_dataset

    #TODO 
    def get_training_dataset(self) -> dict[str, np.ndarray]:
        pass 
