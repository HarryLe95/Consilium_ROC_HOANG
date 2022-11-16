from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional
import dateutils
import pandas as pd
import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataParser(ABC):
    def __init__(self, state: dict, config: dict, kwdict:dict|None=None):
        self._state = state
        self._config = config

    @property
    @abstractmethod
    def get_well_data_df(self) -> pd.DataFrame:
        pass
    
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
                item_cd:str, 
                file_prefix:str, 
                start:datetime|str, 
                end:datetime|str, 
                strp_format:str='%Y%m%d',
                strf_format:str='%Y%m%d',
                file_ext:str='csv') -> str:
        """Get filename that adheres to Santos' naming convention

        Example: 
        >>> S3Manager.get_filename("MOOMBA","SOLAR_DATA","2020-01-01","2020-02-01","%Y-%m-%d")
        >>> MOOMBA_SOLAR_DATA_2020-01-01_2020_02_01.csv
        Args:
            item_cd (str): well_cd or weather station code
            file_prefix (str): file_prefix
            start (datetime | str): start date
            end (datetime | str): end date
            strp_format (str, optional): format to read start and end if given as string. Defaults to '%Y%m%d'.
            strf_format (str, optional): format suffix date in file name. Defaults to '%Y%m%d'.
            file_ext (str, optional): file_extension. Defaults to 'csv'.
        
        Returns:
            str: formatted filename 
        """
        if isinstance(start,str):
            start = cls.parse_date(start, strp_format)
        if isinstance(end,str):
            end = cls.parse_date(end, strp_format)
        fn = '{}_{}_{}_{}.{}'.format(item_cd, 
                                    file_prefix, 
                                    start.strftime(strf_format), 
                                    end.strftime(strf_format), 
                                    file_ext)
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
    
    def _get_tag_file_list(self) -> list:
        logger.info("Running get_tag_file_list")
        file_start_date = self._config['start_time']
        file_end_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) + dateutils.relativedelta(months=1)
        file_end_list = [file_start_date + dateutils.relativedelta(months=months) for months in
                         range(1, DataParser._get_month_data_duration(file_start_date, file_end_date) + 1)]
        
        tag_path = f"ROC_{self._config['tag_data_kwargs']['path']}"
        tag_data_file_list = []
        for well in self._config['inference_wells']:
            fs = file_start_date.strftime("%Y%m%d")
            for end_date in file_end_list:
                fe = end_date.strftime("%Y%m%d")
                tag_data_file_list.append(self._get_file_name(well,fs,fe))
                fs = fe
        logger.info(f"Number of TAG_DATA files: {len(tag_data_file_list)}")
        return tag_data_file_list

    def _get_label_file_df(self, well: str) -> Optional[pd.DataFrame]:
        logger.info(f"Running _get_label_file_list for well: {well}")
        label_df = pd.DataFrame()
        qc_name = self._config['file_prefixes']['qc_name']
        label_data_prefix = self._config['file_prefixes']['label_data_file_prefix']
        file_format = self._config['file_prefixes']['label_data_format']
        args = {}
        sql = self._config['label_data_sql']
        kwargs = self._config['label_data_kwargs']
        if self._state['label_data_con'].info['connection_type'] in ["s3", "file"]:
            if qc_name is None:
                kwargs['file'] = f"{well}_{label_data_prefix}{file_format}"
            else:
                kwargs['file'] = f"{well}_{label_data_prefix}_{qc_name}{file_format}"
        else:
            args = {
                'W': well
            }
        try:
            monthly_df = self._state['label_data_con'].read(sql=sql, args=args, edit=[], orient='df', do_raise=False, **kwargs)
            label_df = pd.concat([label_df, monthly_df])
        except Exception as e:
            logger.exception(f"Trying to parse well: {well} label data raised exception: {e}")
            return None
        return label_df

    def _create_label_df(self, well: str) -> Optional[pd.DataFrame]:
        logger.info(f"_create_label_df function call")
        label_df = self._get_label_file_df(well)
        if label_df is None:
            return None
        label_df['MODEL_RUN_TIME'] = pd.to_datetime(label_df['MODEL_RUN_TIME'],
                                                    format="%d/%m/%Y %H:%M",
                                                    errors="coerce").fillna(
            pd.to_datetime(label_df['MODEL_RUN_TIME'], format="%Y-%m-%d %H:%M:%S", errors="coerce"))
        label_df['BDTA_START_TS'] = pd.to_datetime(label_df['BDTA_START_TS'],
                                                   format="%d/%m/%Y %H:%M",
                                                   errors="coerce").fillna(
            pd.to_datetime(label_df['BDTA_START_TS'], format="%Y-%m-%d %H:%M:%S", errors="coerce"))
        label_df['BDTA_END_TS'] = pd.to_datetime(label_df['BDTA_END_TS'],
                                                 format="%d/%m/%Y %H:%M",
                                                 errors="coerce").fillna(
            pd.to_datetime(label_df['BDTA_END_TS'], format="%Y-%m-%d %H:%M:%S", errors="coerce"))
        label_df['LAST_QC_TS'] = pd.to_datetime(label_df['LAST_QC_TS'], format="%d/%m/%Y %H:%M",
                                                errors="coerce").fillna(
            pd.to_datetime(label_df['LAST_QC_TS'], format="%Y-%m-%d %H:%M:%S", errors="coerce"))

        logger.info(f"Head of created label_df: {label_df.head()}")
        return label_df

    @staticmethod
    def _get_month_data_duration(file_start_date: datetime, file_end_date: datetime) -> int:
        time_difference = dateutils.relativedelta(file_end_date, file_start_date)
        number_of_months = time_difference.years * 12 + time_difference.months
        return number_of_months
