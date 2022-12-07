from typing import Sequence 
from datetime import timedelta
import datetime 
from dateutil import relativedelta
from copy import deepcopy
import logging

import pandas as pd 
import numpy as np 

from Dataset.generic import ABC_Dataset
from utils.advancedanalytics_util import AAUConnection, S3Connection, FileConnection, AAPandaSQL, aauconnect_

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PROCESSOR_MIXIN:
    @classmethod 
    def extract_time_period(cls, data:pd.DataFrame)->tuple:
        """Extract the start and end timestamps from a given dataset. 
        start timestamp is rounded down to the nearest %Y-%m-%d 00:00:00
        end timestamp is rounded up to the nearest %Y-%m-%d 23:59

        Args:
            data (pd.DataFrame): _description_

        Returns:
            tuple: start and end timestamps 
        """
        start = data.index.min().replace(hour=0,minute=0)
        end = data.index.max().replace(hour=23,minute=59)
        return start, end 
    
    @classmethod
    def fill_missing_date(cls, data: pd.DataFrame, start:str, end:str) -> pd.DataFrame:
        """Fill minutely missing timestamps

        Args:
            data (pd.DataFrame): data 
            start (str): start date - minutely datetime
            end (str): end date - minutely datetime 

        Returns:
            pd.DataFrame: dataframe with nan values for missing time gaps 
        """
        index = pd.date_range(start,end,freq="T")
        return data.reindex(index)
        
    @classmethod 
    def select_features(cls, data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        """Select relevant features from dataframe 

        Args:
            data (pd.DataFrame): input dataframe
            features (Sequence[str]): feature list 

        Returns:
            pd.DataFrame: dataframe with selected features
        """
        assert "WELL_CD" in data, "Missing well code in dataframe."
        well_code = np.unique(data["WELL_CD"])
        for feature in features: 
            if feature not in data.columns: 
                logger.debug(f"Missing feature {feature} for well {well_code}")
        return data.loc[:,features]
    
    @classmethod 
    def create_mask_and_fill_nan(cls, data: pd.DataFrame, features: Sequence[str], fill_method:str="interpolate") -> pd.DataFrame:
        """Create a binary mask for each feature describing whether the data point at corresponding mask is raw or interpolated 

        Args:
            data (pd.DataFrame): input dataframe 
            features (Sequence[str]): relevant features for modelling purposes 
            fill_method (str, optional): how to fill nan values - either zero or interpolate

        Returns:
            pd.DataFrame: new data frame with mask and nan value filled 
        """
        for feature in features: 
            data[f'Mask_{feature}']=1-data[feature].isna()
            if fill_method == "zero":
                data[feature] = data[feature].fillna(value=0)
            else:
                data[feature] = data[feature].interpolate(method='linear', limit_direction='both')    
        return data         
    
    @classmethod 
    def normalise_data(cls, data:pd.DataFrame, normalise_params)->pd.DataFrame:
        return data
    
    @classmethod 
    def aggregate_data(cls, data: pd.DataFrame) -> pd.DataFrame: 
        """Aggregate data so that each row contains a list of 1440 data points

        Args:
            data (pd.DataFrame): input dataframe 

        Returns:
            pd.DataFrame: aggregated dataframe
        """
        return data.groupby(data.index.date).agg(list)
    
    @classmethod
    def process_data(cls, data: pd.DataFrame, features: Sequence[str], fill_method:str, normalise_params:dict) -> pd.DataFrame:
        """Apply transformation on unprocessed data 

        Args:
            data (pd.DataFrame): raw dataframe with index as date-time objects 
            features (Sequence[str]): relevant features to select from data.
            fill_method (str): method to fill nan values 
            normalise_params (dict): normalisation parameters 

        Returns:
            pd.DataFrame: processed data frame 
        """
        start, end = cls.extract_time_period(data)
        data = cls.fill_missing_date(data, start, end)
        data = cls.select_features(data, features)
        data = cls.create_mask_and_fill_nan(data, features, fill_method)
        data = cls.normalise_data(data, normalise_params)
        return cls.aggregate_data(data)
    
class FILENAMING_MIXIN:
    """A utility object:
    
    Methods:
        parse_date: receives an input string and outputs a datetime.datetime object
        get_filename: receives filename components (well_code,dates,etc) and returns the formatted string of filename 
        get_metadata_name: receives filename components and returns the formatted string of file metadata
        get_date_range: output pd.date_range from start to end 
    """
    @staticmethod
    def parse_date(date:str|datetime.datetime, strp_format='%Y-%m-%d') -> datetime.datetime:
        """Parse str as datetime object

        Args:
            date (str): datestring
            strp_format (str, optional): format. Defaults to '%Y-%m-%d'.

        Returns:
            datetime.datetime: datetime object from date
        """
        try:
            return datetime.datetime.strptime(date, strp_format)
        except:
            raise ValueError(f"Incompatiable input date {date} and format: {strp_format}")

    @classmethod
    def get_filename(cls,
                well_cd:str, 
                file_prefix:str, 
                start:datetime.datetime, 
                end:datetime.datetime,
                strf_format:str='%Y%m%d',
                file_suffix:str='.csv') -> str:
        """Get filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("MOOMBA","SOLAR_DATA","2020-01-01","2020-02-01","%Y-%m-%d")
        >>> MOOMBA_SOLAR_DATA_2020-01-01_2020_02_01.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            start (datetime.datetime): start date
            end (datetime.datetime): end date
            strf_format (str, optional): format suffix date in file name. Defaults to '%Y%m%d'.
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        return f"{well_cd}_{file_prefix}_{start.strftime(strf_format)}_{end.strftime(strf_format)}{file_suffix}"
        
    @classmethod 
    def get_metadata_name(cls,
                well_cd:str, 
                file_prefix:str, 
                file_suffix:str='.csv') -> str:
        """Get metadata filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("TIRRA80","ROC_PROCESSED_DATA",".csv")
        >>> TIRRA80_ROC_PROCESSED_DATA_LAST.csv
        Args:
            well_cd (str): well_cd 
            file_prefix (str): file_prefix
            file_suffix (str, optional): file_suffixension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        return f'{well_cd}_{file_prefix}_LAST{file_suffix}'
    
    @classmethod
    def get_date_range(self, start_date:str, end_date:str, strp_format:str='%Y-%m-%d') -> pd.Series:
        """Get a date range from strings specifying the start and end date. 
        
        start_date is rounded down to the first day of the month. end_date is rounded to the first day of next month (if current day is not 1).
        Santos files recorded data from the 1st of one month to the minute right before the 1st of next month. For instance: 
        ACRUS1_PROCESSED_DATA_20170101_20170201 records data in ACRUS1 from 1/1/2017 to 31/1/2017. Therefore, if start_date = 2017-01-05, end_date = 2017-02-06, the date range includes
        2017_01_01_2017_02_01 and 2017_02_01_2017_03_01
        
        Args:
            start_date (str): start date
            end_date (str): end date
            strp_format (str): how the start and end date strings should be formatted. Defaults to Y-M-D

        Returns:
            pd.Series: date range 
        """
        start_date = self.parse_date(start_date, strp_format=strp_format)
        end_date = self.parse_date(end_date, strp_format=strp_format)
        if start_date > end_date: 
            raise ValueError(f"End date {end_date} must come after start date {start_date}.")
        if end_date.day != 1:
            end_date = end_date.replace(day = 1)
            end_date += relativedelta.relativedelta(months=1) 
        start_date = start_date.replace(day = 1) 
        return pd.date_range(start_date, end_date, freq="MS")
    
    @classmethod
    def get_output_time_slice(cls, start:str|datetime.date, end:str|datetime.date, strp_format="%Y-%m-%d") -> tuple[str,str]:
        """Get start and end time for running inference. 
        
        Output the start and end indices for slicing data to get data for inference.  
                
        Example:
            If today is 3rd of Nov 2022, in live mode (start=2022-11-02, end=2022-11-03), the model runs inference for data 
            on the 2nd of Nov 2022. The output start and end time indices are - '2022-11-02 00:00' and '2022-11-02 23:59'. This means 
            the csv fetched will be ACRUS1_PROCESSED_DATA_20221101_20221201.csv, which will be further sliced from "2022-11-02 00:00" to 
            "2022-11-02 23:59:

        Args: 
            start (str | datetime.date): start date. Yesterday's date in live mode. Output start index = start 
            end (str | datetime.date): end date. Today's date in live mode. Output end index = end - 1 minute
            strp_format (str, optional): format of start and end string. Defaults to "%Y-%m-%d".

        Raises:
            TypeError: if input types are neither string nor datetime.date

        Returns:
            tuple[str,str]: start_, end_ time indices 
        """
        minute_string_format = "%Y-%m-%d %H:%M"
        def process_date_slices(d:str|datetime.date, offset:timedelta=timedelta(minutes=0)):
            if not isinstance(d,(str,datetime.date)):
                raise TypeError(f"Expected type: (str,datetime.date), actual type: {type(d)}")
            if isinstance(d,str):
                d = datetime.datetime.strptime(d, strp_format)
            if isinstance(d,datetime.date): #Convert from datetime.date to datetime.datetime object 
                d = datetime.datetime.strptime(d.strftime(minute_string_format),minute_string_format) 
            d = d + offset
            return d.strftime(minute_string_format)
        start_ = process_date_slices(start)
        end_ = process_date_slices(end, timedelta(minutes=-1))
        return start_, end_
        
class Dataset(ABC_Dataset, PROCESSOR_MIXIN, FILENAMING_MIXIN):
    def __init__(self, 
                 connection: AAUConnection, 
                 path:str, 
                 file_prefix:str, 
                 file_suffix:str,
                 features:Sequence[str], 
                 fill_method:str="zero",
                 normalise_params:dict=None,
                 datetime_index_column:str="TS",
                 **kwargs) -> None:
        self.connection = connection 
        self.partition_mode = None 
        self.path = path
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.features = features
        self.fill_method = fill_method 
        self.normalise_params = normalise_params
        self.datetime_index_column = datetime_index_column
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
    
    def read_data(self, well_cd:str, start:str, end:str, concat:bool=True, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> dict|pd.DataFrame:
        """Read well data from database 
        
        Args:
            well_cd (str): well code
            start (str): start date
            end (str): end date 
            concat (bool, optional): whether to concatenate to form a concatenated dataframe. Defaults to True.
            strp_format (str, optional): input dates (start, end) formats. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): output date string format (format on actual filename). Defaults to '%Y%m%d'.

        Raises:
            e: Credential Exception or File Not on Database Exception

        Returns:
            dict|pd.DataFrame: output object: dict whose keys are dates and values are the dataframes, or a concatenated dataframe
        """
        start_datetime = self.parse_date(start)
        end_datetime = self.parse_date(end)
        response = {}
        date_range = self.get_date_range(start, end, strp_format=strp_format)
        kwargs=deepcopy(self.kwargs)
        
        #Read data from database 
        for d in range(len(date_range)-1):
            file_start, file_end = date_range[d], date_range[d+1]
            file_name = self.get_filename(well_cd=well_cd, file_prefix=self.file_prefix, 
                                        start=file_start, end=file_end, file_suffix=self.file_suffix,
                                        strf_format=strf_format)
            kwargs['file']=file_name
            try:
                result = self.connection.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
                if result is not None:
                    result["TS"] = pd.to_datetime(result["TS"])
                    if start_datetime >= file_start: 
                        start_,_ = self.get_output_time_slice(start_datetime, file_end,strp_format)
                        result = result.loc[result.TS[result.TS >=start_].index,:]            
                    if end_datetime <= file_end:
                        start_, end_ = self.get_output_time_slice(end_datetime.replace(day=1), end_datetime, strp_format)
                        result = result.loc[result.TS[result.TS<=end_].index,:]
                    result.set_index("TS",inplace=True)
                    response[file_name] = result
            except Exception as e:
                raise e 

        #Concatenate data to form a single dataframe 
        if concat:
            try:
                all_output = [data for data in response.values() if data is not None]
                all_df =  pd.concat(all_output,axis=0,ignore_index=False)
                return all_df
            except Exception as e:
                logger.error(e)
                return None
        return response
    
    def read_metadata(self, well_cd:str) -> pd.DataFrame: 
        """Read meta data from database. Metadata file name is well_cd_ROC_PROCESSED_DATA_LAST.csv 

        Args:
            well_cd (str): well code

        Raises:
            e: invalid credential exception or file does not exist exception

        Returns:
            pd.DataFrame: metadata dataframe
        """
        kwargs = deepcopy(self.kwargs)
        file_name = self.get_metadata_name(well_cd, self.file_prefix, self.file_suffix)
        kwargs['file']=file_name
        try: 
            result = self.connection.read(sql=None, args={}, edit=[], orient="df", do_raise=False, **kwargs)
            if result is not None: 
                return {well_cd: result}
        except Exception as e:
            raise e 

    def get_metadata(self, wells: Sequence[str]) -> dict[str, pd.DataFrame]:
        """Get metadata for a group of wells 

        Args:
            wells (Sequence[str]): list of well codes whose metadata will be fetched

        Returns:
            dict[str, pd.DataFrame]: dictionary whose keys are the well codes and values are the metadata dataframe 
        """
        metadata_dict = {}
        for well in wells:
            metadata_dict.update(self.read_metadata(well))
        return metadata_dict 

    def get_wells_dataset(self, wells:Sequence[str], start:str, end:str, strp_format='%Y-%m-%d',strf_format:str='%Y%m%d') -> dict[str,pd.DataFrame]:
        """Fetch all data for either training or inference 

        Args:
            wells (Sequence[str]): group of wells 
            start (str): start date 
            end (str): end date 
            strp_format (str, optional): input string format. Defaults to '%Y-%m-%d'.
            strf_format (str, optional): output string format. Defaults to '%Y%m%d'.

        Raises:
            ValueError: if data frame is empty

        Returns:
            dict: dictionary whose key is the well code and value is the corresponding well inference data  
        """
        all_wells = {}
        for well in wells:
            well_df = self.read_data(well, start, end, concat=True, strp_format=strp_format, strf_format=strf_format)
            if well_df is not None: 
                well_df['WELL_CD'] = well
                all_wells[well]=self.process_data(well_df, self.features, self.fill_method, self.normalise_params)
            else:
                all_wells[well]=None
        return all_wells
     
if __name__ == "__main__":
    import config.__config__ as base_config
    config = base_config.init()
    connection = aauconnect_(config['cfg_file_info'])
    
    dataset = Dataset(connection, path="C:/Users/HoangLe/Desktop/Consilium_ROC_HOANG/app_prod/roc/PROCESSED_DATA",
                      file_prefix="ROC_PROCESSED_DATA",file_suffix=".csv",bucket=config['cfg_s3_info']['bucket'], features=['ROC_VOLTAGE','FLOW','PRESSURE_TH'])
    data_dict = dataset.read_data("ACRUS1","2016-11-04","2016-11-05",True)
    print("End")
    
    