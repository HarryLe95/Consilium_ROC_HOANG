from src.aau.S3Manager import S3Manager
import logging 
from datetime import datetime as datetime 
import pandas as pd 
import numpy as np 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class S3ROCManager(S3Manager):
    def __init__(self, info:dict):
        super().__init__(info)
        self.info = info 
        self.init_processed_data()
        self.init_solar_data()

    def init_processed_data(self):
        self.processed_data_dict = {"path": "ROC/PROCESSED_DATA", 
                                    "file_prefix": "ROC_PROCESSED_DATA",
                                    "args_ts": "DTSMIN",
                                    "bucket": self.bucket,
                                    "file_ext": "csv"}
        if 'procdata_kwargs' in self.info:
            if 'args_ts' in self.info['procdata_kwargs']:
                self.processed_data_dict["args_ts"] = self.info['procdata_kwargs']["args_ts"]
            if 'path' in self.info['procdata_kwargs']:
                self.processed_data_dict["path"] = self.info['procdata_kwargs']["path"]
            if 'file_ext' in self.info['procdata_kwargs']:
                self.processed_data_dict["file_ext"] = self.info['procdata_kwargs']["ext"]
            if 'file_prefix' in self.info['procdata_kwargs']:
                self.processed_data_dict["file_ext"] = self.info['procdata_kwargs']["ext"]
            if 'bucket' in self.info['procdata_kwargs']:
                self.processed_data_dict["bucket"] = self.info['procdata_kwargs']["bucket"]
        if "DTSMIN" in self.processed_data_dict['args_ts'] and isinstance(self.processed_data_dict['args_ts'], list):
            self.processed_data_dict['args_ts'] = "DTSMIN"

    def init_solar_data(self):
        self.solar_data_dict = {"path": "ROC/SOLAR_DATA", 
                                "file_prefix": "SOLAR_DATA",
                                "args_ts": "TS",
                                "bucket": self.bucket,
                                "file_ext": "csv"}
        if 'solardata_kwargs' in self.info:
            if 'args_ts' in self.info['solardata_kwargs']:
                self.solar_data_dict["args_ts"] = self.info['solardata_kwargs']["args_ts"]
            if 'path' in self.info['solardata_kwargs']:
                self.solar_data_dict["path"] = self.info['solardata_kwargs']["path"]
            if 'file_ext' in self.info['solardata_kwargs']:
                self.solar_data_dict["file_ext"] = self.info['solardata_kwargs']["ext"]
            if 'file_prefix' in self.info['solardata_kwargs']:
                self.solar_data_dict["file_ext"] = self.info['solardata_kwargs']["ext"]
            if 'bucket' in self.info['solardata_kwargs']:
                self.solar_data_dict["bucket"] = self.info['solardata_kwargs']["bucket"]

    def read_solar(self,
                   station_code:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d') -> pd.DataFrame:
        alldf =  self.read_from_storage(item_cd=station_code, start=start, end = end,
                                      strp_format=strp_format, strf_format=strf_format,
                                      **self.solar_data_dict)
            
        #Apply solar preprocessing - remove duplicates and make continuous time index
        TS = self.solar_data_dict['args_ts']
        alldf[TS]=pd.to_datetime(alldf[TS])
        alldf.set_index(TS,inplace=True)

        date_range = pd.date_range(alldf.index.min(), alldf.index.max(), freq='H')
        alldf = alldf.groupby(alldf.index).mean()
        alldf = alldf.reindex(date_range)
        alldf.index.name = "TS"
        alldf.reset_index(inplace=True)
        return alldf
        
    def read_processed_data(self,
                   well_code:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d',
                   nan_replace_method:str='interpolate') -> pd.DataFrame:
        alldf =  self.read_from_storage(item_cd=well_code, start=start, end = end,
                                      strp_format=strp_format, strf_format=strf_format,
                                      **self.processed_data_dict)
        
        #Processed data preprocessing - remove sub minute duplicates 
        #Pad data to form continuous time sequence
        #Create Nan Mask, and replace nan 
        TS = self.processed_data_dict['args_ts']
        alldf[TS]=pd.to_datetime(alldf[TS])
        alldf.set_index(TS,inplace=True)
        alldf = alldf.loc[:,['ROC_VOLTAGE','FLOW','PRESSURE_TH']]

        date_range = pd.date_range(alldf.index.min(), alldf.index.max(), freq='T')
        alldf = alldf.groupby(alldf.index).mean()
        alldf = alldf.reindex(date_range)
        alldf['Mask_ROC_VOLTAGE']=1-alldf.ROC_VOLTAGE.isna()
        alldf['Mask_FLOW']=1-alldf.FLOW.isna()
        alldf['Mask_PRESSURE_TH']=1-alldf.PRESSURE_TH.isna()

        if nan_replace_method == 'zero':
            alldf.fillna(0)
        elif nan_replace_method == 'interpolate':
            alldf.interpolate(method='linear', inplace=True, limit_direction='both')
        else:
            print("Invalid nan_replace_method. Accepts either zero or interpolate.")
        alldf.index.name = "TS"
        alldf.reset_index(inplace=True)
        return alldf
    
    def list_weather_stations(self):
        all_data = self.list_files(self.solar_data_dict['path'])
        unique_stations = {x.split(self.solar_data_dict['path'])[1].split('_'+self.solar_data_dict['file_prefix'])[0] for x in all_data}
        return unique_stations

    def list_all_wells(self):
        all_data = self.list_files(self.processed_data_dict['path'])
        unique_stations = {x.split(self.processed_data_dict['path'])[1].split('_'+self.processed_data_dict['file_prefix'])[0] for x in all_data}
        return unique_stations