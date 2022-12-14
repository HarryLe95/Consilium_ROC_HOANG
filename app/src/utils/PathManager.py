from pathlib import Path
import os 
from typing import Sequence
import src.aau.advancedanalytics_util as aau
import pandas as pd 

_UTIL_PATH = Path(__file__)

class PathManager:
    UTILS_PATH = _UTIL_PATH
    BASE_PATH = UTILS_PATH.parents[3]
    DATA_PATH = BASE_PATH / "data"
    CONFIG_PATH = BASE_PATH / "config"
    MODEL_PATH = BASE_PATH /"model"
    VIZ_PATH = BASE_PATH /"viz"
    LOG_PATH = BASE_PATH/"log"
    
    @staticmethod  
    def _file_from_path(path: Path, file_name:str, sub_folder:str|Sequence[str]='') -> Path:
        """Get location of a file stored in a base path whose sub_paths are specified under sub_folder.
        
        Example:
        If a csv "ACRUS1.csv" has the absolute path "Data/PROCESSED_DATA/TEMP/ACRUS1.csv", and path is a pathlib.Path to "Data": 
        >>> Paths._file_from_path(path, "ACRUS1.csv", ['PROCESSED_DATA', 'TEMP'])
        >>> Path("Data/PROCESSED_DATA/TEMP/ACRUS1.csv")

        Args:
            path (Path): pathlib.Path object to the base folder. 
            file_name (str): file
            sub_folder (str | Sequence[str], optional): sub_paths from path to file. Defaults to ''.

        Returns:
            Path: absolute path to the file location
        """
        file_path = path
        if isinstance(sub_folder, str):
            file_path /= sub_folder
        elif hasattr(sub_folder, '__len__'):
            for sub_path in sub_folder:
                file_path /= sub_path
        file_path /= file_name
        return file_path 
    
    @classmethod
    def config(cls, file_name:str, sub_folder:str|Sequence[str]='') -> Path:
        """Get the path to a file stored in config folder.

        Args:
            file_name (str): file
            sub_folder (str | Sequence[str], optional): sub_paths from config to file. Defaults to ''.

        Returns:
            Path: absolute path to the file locaton
        """
        return cls._file_from_path(cls.CONFIG_PATH, file_name, sub_folder)
    
    @classmethod
    def data(cls, file_name:str, sub_folder: str|Sequence[str]='')-> Path:
        """Get the path to a file stored in data folder.

        Args:
            file_name (str): file
            sub_folder (str | Sequence[str], optional): sub_paths from data to file. Defaults to ''.

        Returns:
            Path: absolute path to the file locaton
        """
        return cls._file_from_path(cls.DATA_PATH, file_name, sub_folder)
    
    @classmethod
    def viz(cls, file_name:str, sub_folder: str|Sequence[str]='')-> Path:
        """Get the path to a file stored in visualisation folder.

        Args:
            file_name (str): file
            sub_folder (str | Sequence[str], optional): sub_paths from viz to file. Defaults to ''.

        Returns:
            Path: absolute path to the file locaton
        """
        return cls._file_from_path(cls.VIZ_PATH, file_name, sub_folder)
    
    @classmethod
    def model(cls, file_name:str, sub_folder: str|Sequence[str]='')-> Path:
        """Get the path to a file stored in model folder.

        Args:
            file_name (str): file
            sub_folder (str | Sequence[str], optional): sub_paths from model to file. Defaults to ''.

        Returns:
            Path: absolute path to the file locaton
        """
        return cls._file_from_path(cls.MODEL_PATH, file_name, sub_folder)
    
    @classmethod
    def log(cls, file_name:str, sub_folder: str|Sequence[str]='')-> Path:
        """Get the path to a file stored in log folder.

        Args:
            file_name (str): file
            sub_folder (str | Sequence[str], optional): sub_paths from log to file. Defaults to ''.

        Returns:
            Path: absolute path to the file locaton
        """
        return cls._file_from_path(cls.LOG_PATH, file_name, sub_folder)
    
    @classmethod 
    def read_config(cls, file_name:str, sub_folder: str|Sequence[str]='')->dict:
        base = {
            '_type' : 'file',
            '_info': {
                'connection_type' : 'file',
                'path' : cls.config(file_name,sub_folder).parents[0],
                'file' : file_name
            },
            '_sql': None,
            '_kwargs': {
                }
        }
        sql = base['_sql']
        kwargs = base['_kwargs']
        config = aau.aauconnect_(base['_info']).read(sql=sql, args={}, edit=[], orient='config', do_raise=True, **kwargs)
        return config 

    @classmethod 
    def write_config(cls, config:dict|pd.DataFrame, file_name:str,sub_folder:str|Sequence[str]='') -> None:
        file_path = cls._file_from_path(cls.CONFIG_PATH, file_name, sub_folder)
        config_df = pd.DataFrame({"KEY":config.keys(),
                                  "VALUE":config.values(), 
                                  "TYPE":[type(config[i]).__name__ if config[i] is not None else 'str' for i in config.keys()]})
        config_df.to_csv(file_path,index=False)
