from generator.generic import ABC_DataManager, ABC_Dataset
from utils.advancedanalytics_util import AAUConnection

class DataManager(ABC_DataManager):
    def __init__(self,
                 wells:str, 
                 connection: AAUConnection,
                 run_mode:str='live',
                 perform_model_training:bool=True,
                 perform_model_inference:bool=False 
                 ):
        self.wells = wells 
        self.connection = connection 
        self.perform_model_training = perform_model_training 
        self.perform_model_inference = perform_model_inference
        self.run_mode = run_mode
        
    def get_metadata(self) -> dict:
        pass 
    
    def update_metadata(self) -> None: 
        pass 