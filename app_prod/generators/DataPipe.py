import logging 
from   typing   import List, Optional, Sequence 
from   datetime import timedelta 
from   datetime import datetime as datetime

import pandas as pd 
import numpy as np 

from geopy import distance 
from sklearn.preprocessing import StandardScaler

from S3Manager import S3Manager
from utils.PathManager import Paths as Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

well_features = ['ROC_VOLTAGE', 'FLOW', 'PRESSURE_TH']

class S3ROCManager(S3Manager):
    """S3 aau wrapper that provides a convenient interface for working with ROC data.

    - processed_data: raw ROC data from battery sensors containing ROC_VOLTAGE, FLOW, PRESSTURE_TH and their repective mask and corresponding timestamp.

    For convenience, file paths, prefixes, file_extension and related S3 keywords are stored in processed_data_dict

    Args:
        S3Manager (aau.S3Manager): inherits S3Manager from aau.S3Manager 
    """
    def __init__(self, info:dict):
        super().__init__(info)
        self.info = info 
        self.init_processed_data()

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

        logger.info(f"Initialising processed data S3 path: {self.processed_data_dict['path']}")
        logger.info(f"Initialising processed data S3 file prefix: {self.processed_data_dict['file_prefix']}")
        logger.info(f"Initialising processed data S3 file time stamp: {self.processed_data_dict['args_ts']}")
        logger.info(f"Initialising processed data S3 file extension: {self.processed_data_dict['file_ext']}")
    
    def read_processed_data(self,
                   well_code:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d',
                   nan_replace_method:str='interpolate',
                   to_csv:bool=False) -> pd.DataFrame:
        """Read combined sensor data from S3 database 

        Args:
            well_code (str): well code
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): interpretation format for start and end. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): S3 file storage date format. Defaults to '%Y%m%d'.
            nan_replace_method (str, optional): method to replace nan values. One of 'zero' or 'interpolate'. Defaults to 'interpolate'.
            to_csv (bool, optional): whether to save solar data to dedicated local directory. Defaults to False.

        Returns:
            pd.DataFrame: combined raw data
        """

        logger.info(f"Read well data from database for well: {well_code} from {start} to {end}")
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
            alldf.fillna(0, inplace=True)
        elif nan_replace_method == 'interpolate':
            alldf.interpolate(method='linear', inplace=True, limit_direction='both')
        else:
            logger.error("Invalid nan_replace_method. Accepts either zero or interpolate.")
        alldf.index.name = "TS"
        alldf.reset_index(inplace=True)
        alldf["WELL_CD"]=well_code
        if to_csv:
            alldf.to_csv(Path.data(f"{well_code}_{start}_{end}_raw.csv"),index=False)
            logger.info(f"Save well data to {well_code}_{start}_{end}_raw.csv")
        return alldf
    
    def read_data(self,
                   wells:str, 
                   start: datetime|str,
                   end: datetime|str,
                   strp_format:str='%Y-%m-%d',
                   strf_format:str='%Y%m%d',
                   nan_replace_method:str='interpolate'):
        all_df = []
        for well_code in wells:
            df = self.read_processed_data(well_code,start,end,strp_format,strf_format,nan_replace_method)
            df = df.groupby(df.TS.date).agg(list)
            all_df.append(df)
        all_df = pd.concat(all_df,axis=0)
        return all_df
            