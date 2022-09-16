from aau.advancedanalytics_util import S3
from pathlib import Path
from datetime import datetime  
import pandas as pd

class S3Manager(S3):
    def _list_dir(self, bucket:str, prefix:str='') -> tuple[dict, list]:
        """Internal method for listing directory in a bucket given a prefix

        Example: if an S3 folder structure of MyBucket is as follows:
        ROC/
            PROCESSED_DATA/
            SOLAR_DATA/
            LABEL_DATA/
                HUMAN_LABEL/
                MACHINE_LABEL/

        Then:
        >>> S3Manager._list_dir(MyBucket,'')
        >>> {'':['ROC']}, ['ROC']
        >>> S3Manager._list_dir(MyBucket,'ROC')
        >>> {'ROC': ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA']}, ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA']
        >>> S3Manager._list_dir(MyBucket, 'ROC/LABEL_DATA')
        >>> {'ROC/LABEL_DATA': ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']}, ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']

        Args:
            bucket (str): s3 bucket
            prefix (str, optional): subdirectory prefix. Defaults to ''.

        Returns:
            tuple[dict, list]: dictionary and list of all sub_directories under the current prefix directory at 1 level
        """
        prefix_path = prefix +'/' if prefix != '' else prefix
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix_path, Delimiter='/')
        try:
            sub_folder_prefix = [Path(item['Prefix']).as_posix() for item in response['CommonPrefixes']]
            sub_folder_rel_path = {prefix: sub_folder_prefix}
            return sub_folder_rel_path, sub_folder_prefix
        except KeyError as e:
            return {}, []

    def list_dir(self, bucket:str, prefix:str='',recursive:bool=False) -> dict:
        """List directory in a bucket given a prefix

        Example: if an S3 folder structure of MyBucket is as follows:
        ROC/
            PROCESSED_DATA/
            SOLAR_DATA/
            LABEL_DATA/
                HUMAN_LABEL/
                MACHINE_LABEL/

        Then:
        >>> S3Manager._list_dir(MyBucket,'')
        >>> {'':['ROC']}
        >>> S3Manager._list_dir(MyBucket,'ROC')
        >>> {'ROC': ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA']}
        >>> S3Manager._list_dir(MyBucket, 'ROC/LABEL_DATA')
        >>> {'ROC/LABEL_DATA': ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']}
        >>> S3Manager._list_dir(MyBucket,'',True)
        >>> {'':['ROC'], 'ROC': ['ROC/PROCESSED_DATA', 'ROC/SOLAR_DATA', 'ROC/LABEL_DATA'], 'ROC/LABEL_DATA': ['ROC/LABEL_DATA/HUMAN_LABEL','ROC/LABEL_DATA/MACHINE_LABEL']}

        Args:
            bucket (str): s3 bucket
            prefix (str, optional): subdirectory prefix. Defaults to ''.
            recursive (bool): if False, only get directories that are direct children of the directory specified in prefix. Otherwise, get all descendent directories. Defaults to False.

        Returns:
            dict: dictionary whose values are immediate children directory of the corresponding key directory.
        """
        ls, next = self._list_dir(bucket, prefix)
        status=True
        if recursive: 
            while status:
                if len(next)==0:
                    break
                prefix = next.pop()
                current_ls, next_prefix = self._list_dir(bucket, prefix)
                next.extend(next_prefix)
                ls.update(current_ls)
        return ls 

    def list_files(self, bucket:str, prefix:str='ROC', file_prefix: str=None) -> list:
        """List all files in directory specified by prefix containing file_prefix

        Example: folder structure in MyBucket
        LABEL_DATA/
            V1_LABEL_20180101_20180201.csv
            V1_LABEL_20180201_20180301.csv
            V2_LABEL_20180101_20180201.csv

        >>> S3Manager.list_files(MyBucket, 'LABEL_DATA')
        >>> [V1_LABEL_20180101_20180201.csv, V1_LABEL_20180201_20180301.csv, V2_LABEL_20180101_20180201.csv]
        >>> S3Manager.list_files(MyBucket, 'LABEL_DATA', 'V1')
        >>> [V1_LABEL_20180101_20180201.csv, V1_LABEL_20180201_20180301.csv]
        >>> S3Manager.list_files(MyBucket, 'LABEL_DATA', 'V2')
        >>> [V2_LABEL_20180101_20180201.csv]

        Args:
            bucket (str): bucket
            prefix (str, optional): directory to file. Defaults to 'ROC'.
            file_prefix (str, optional): file prefix. Defaults to None.

        Returns:
            list: list of all files in prefix directory containing file_prefix as prefix
        """
        prefix_path = prefix +'/' if file_prefix is None else prefix+'/'+file_prefix
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix_path, Delimiter='/')
        keys = [item['Key'] for item in response['Contents']]
        while response['IsTruncated']:
            continuation_token = response['NextContinuationToken']
            response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix_path, Delimiter='/', ContinuationToken=continuation_token)
            next_keys = [item['Key'] for item in response['Contents']]
            keys.extend(next_keys)
        return keys
    
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
    def get_date_range(self, start_date:str, end_date:str) -> pd.Series:
        """Get a data

        Args:
            start_date (str): _description_
            end_date (str): _description_

        Returns:
            pd.Series: _description_
        """
        start_date = self.parse_date(start_date)
        end_date = self.parse_date(end_date)
        return pd.date_range(start_date, end_date, freq='MS')
    
    

    