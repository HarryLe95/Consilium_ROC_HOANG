from typing import Sequence 
from datetime import datetime 
from copy import deepcopy

import pandas as pd 
import numpy as np 

from generator.generic import ABC_Dataset
from utils.advancedanalytics_util import AAUConnection, S3Connection, FileConnection, AAPandaSQL, aauconnect_

class ROC_PROCESSOR_MIXIN:
    #TODO: fix process data once finalised 
    @classmethod
    def process_data(cls, data: pd.DataFrame) -> np.ndarray:
        return data
    
class ROC_FILENAMING_MIXIN:
    @staticmethod
    def parse_date(date:str, strp_format='%Y-%m-%d') -> datetime:
        """Parse str as datetime object

        Args:
            date (str): datestring
            strp_format (str, optional): format. Defaults to '%Y-%m-%d'.

        Returns:
            datetime: datetime object from date
        """
        try:
            return datetime.strptime(date, strp_format)
        except:
            raise ValueError(f"Incompatiable input date {date} and format: {strp_format}")

    @classmethod
    def get_filename(cls,
                well_cd:str, 
                file_prefix:str, 
                start:datetime|str, 
                end:datetime|str, 
                strp_format:str='%Y%m%d',
                strf_format:str='%Y%m%d',
                file_suffix:str='.csv') -> str:
        """Get filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("MOOMBA","SOLAR_DATA","2020-01-01","2020-02-01","%Y-%m-%d")
        >>> MOOMBA_SOLAR_DATA_2020-01-01_2020_02_01.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): format to read start and end if given as string. Defaults to '%Y%m%d'.
            strf_format (str, optional): format suffix date in file name. Defaults to '%Y%m%d'.
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        if isinstance(start,str):
            start = cls.parse_date(start, strp_format)
        if isinstance(end,str):
            end = cls.parse_date(end, strp_format)
        fn = '{}_{}_{}_{}{}'.format(well_cd, 
                                    file_prefix, 
                                    start.strftime(strf_format), 
                                    end.strftime(strf_format), 
                                    file_suffix)
        return fn

    @classmethod
    def get_date_range(self, start_date:str, end_date:str, freq:str='monthly_start', strp_format:str='%Y-%m-%d') -> pd.Series:
        """Get a date range from strings specifying the start and end date

        Args:
            start_date (str): start date
            end_date (str): end date
            freq (str): one of monthly_start, monthly_end, hourly, minutely. Defaults to monthly_start.
            strp_format (str): how the start and end date strings should be formatted. Defaults to Y-M-D

        Returns:
            pd.Series: date range 
        """
        start_date = self.parse_date(start_date, strp_format=strp_format)
        end_date = self.parse_date(end_date, strp_format=strp_format)
        freq_dict = {"monthly_start": "MS", "monthly_end": "M",
                     "daily": "D","hourly": "H", "minutely": "T"}
        return pd.date_range(start_date, end_date, freq=freq_dict[freq])
        
class Dataset_(ABC_Dataset, ROC_PROCESSOR_MIXIN, ROC_FILENAMING_MIXIN):
    def __init__(self, connection: AAUConnection, 
                 wells:Sequence[str], 
                 path:str, 
                 file_prefix:str, 
                 file_suffix:str, 
                 **kwargs) -> None:
        self.connection = connection 
        self.wells = wells
        self.partition_mode = None 
        self.path = path
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.kwargs = {"path":self.path, "partition_mode":self.partition_mode}
        self.extract_keyword(kwargs)
        
    def extract_keyword(self, kwargs:dict) -> None:
        """Extract relevant keywords from kwargs 

        Args:
            kwargs (dict): dictionary containing connection specific keywords 
        """
        if isinstance(self.connection, S3Connection):
            self.extract_s3_keyword(kwargs)
        if isinstance(self.connection, FileConnection):
            self.extract_file_keyword(kwargs) 
        if isinstance(self.connection, AAPandaSQL):
            self.extract_sql_keyword(kwargs)
    
    def extract_s3_keyword(self,kwargs:dict) -> None:
        if kwargs is None: 
            raise KeyError("Additional keywords must be provided for S3 connections. kwargs must include: bucket")
        try:
            self.bucket = kwargs['bucket']
        except: 
            raise KeyError("Bucket must be provided in config")
        
    def extract_file_keyword(self,kwargs:dict)-> None:
        pass 
    
    def extract_sql_keyword(self, kwargs:dict)-> None:
        pass 
    
    def read_data(self, well_cd:str, start:str, end:str, concat:bool=False, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> pd.DataFrame:
        response = {}
        date_range = self.get_date_range(start, end, strp_format=strp_format)
        kwargs=deepcopy(self.kwargs)
                
        for d in range(len(date_range)-1):
            file_start, file_end = date_range[d], date_range[d+1]
            file_name = self.get_filename(well_cd=well_cd, file_prefix=self.file_prefix, 
                                          start=file_start, end=file_end, file_suffix=self.file_suffix,
                                          strf_format=strf_format)
            kwargs['file']=file_name
            try:
                result = self.connection.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
                if result is not None:
                    response[file_name] = result
            except Exception as e:
                raise e 

        if concat:
            all_output = [data for data in response.values()]
            return pd.concat(all_output,axis=0,ignore_index=True)
        return response

class TrainingDataset(Dataset_):
    def __init__(self, connection: AAUConnection, 
                wells:Sequence[str], 
                path:str, 
                file_prefix:str, 
                file_suffix:str, 
                **kwargs) -> None:
        super().__init__(connection, wells, path, file_prefix,file_suffix,**kwargs)

    def get_dataset_(self, start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> pd.DataFrame:
        all_wells = []
        for well in self.wells:
            well_df = self.read_data(well, start, end, concat=True, strp_format=strp_format, strf_format=strf_format)
            well_df['WELL_CD'] = well
            if well_df is not None: 
                all_wells.append(well_df)
        if len(all_wells)==0:
            raise ValueError("Empty training dataframe")
        return pd.concat(all_wells, axis=0, ignore_index=True)
    
    def get_dataset(self, start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> np.ndarray:
        return self.process_data(self.get_dataset_(start, end, strp_format, strf_format))

class InferenceDataset(Dataset_):
    def __init__(self, connection: AAUConnection, 
            wells:Sequence[str], 
            path:str, 
            file_prefix:str, 
            file_suffix:str, 
            **kwargs) -> None:
        super().__init__(connection, wells, path, file_prefix,file_suffix,**kwargs)

    def get_dataset_(self, start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> pd.DataFrame:
        all_wells = {}
        for well in self.wells:
            import pdb; pdb.set_trace()
            well_df = self.read_data(well, start, end, concat=True, strp_format=strp_format, strf_format=strf_format)
            if well_df is not None: 
                well_df['WELL_CD'] = well
                all_wells[well]=well_df
        if len(all_wells)==0:
            raise ValueError("Empty inference dataframe")
        return all_wells
    
    def get_dataset(self, start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> dict[str,np.ndarray]:
        data_dict = self.get_dataset_(start, end, strp_format, strf_format)
        return {k: self.process_data(v) for k,v in data_dict.items()}
   
if __name__ == "__main__":
    import config.__config__ as base_config
    config = base_config.init()
    connection = aauconnect_(config['cfg_file_info'])
    
    dataset = Dataset_(connection, wells=['ACRUS1'],path="C:/Users/HoangLe/Desktop/Consilium_ROC_HOANG/app_prod/roc/PROCESSED_DATA",
                      file_prefix="ROC_PROCESSED_DATA",file_suffix=".csv",bucket=config['cfg_s3_info']['bucket'])
    data_dict = dataset.read_data("ACRUS1","2016-01-01","2017-01-01")
    
    training_dataset = TrainingDataset(connection, wells=['ACRUS1'],path="C:/Users/HoangLe/Desktop/Consilium_ROC_HOANG/app_prod/roc/PROCESSED_DATA",
                                       file_prefix="ROC_PROCESSED_DATA",file_suffix=".csv",bucket=config['cfg_s3_info']['bucket'])
    
    inference_dataset = InferenceDataset(connection, wells=['ACRUS1'],path="C:/Users/HoangLe/Desktop/Consilium_ROC_HOANG/app_prod/roc/PROCESSED_DATA",
                                         file_prefix="ROC_PROCESSED_DATA",file_suffix=".csv",bucket=config['cfg_s3_info']['bucket'])