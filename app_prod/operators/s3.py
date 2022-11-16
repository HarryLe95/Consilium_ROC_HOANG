from typing import Optional

import pandas as pd

import utils.advancedanalytics_util as aau


class S3Operator:

    @staticmethod
    def connect_(info):
        """
        Standard connect function - may end up moving into util
        """
        connection_type = None
        if info['connection_type'] == 's3':
            connection_type = aau.S3(info)
        elif info['connection_type'] == 'p2':
            connection_type = aau.P2(info)
        elif info['connection_type'] == 'file':
            connection_type = aau.File(info)
        elif info['connection_type'] == 'rts':
            connection_type = aau.RTS(info)
        elif info['connection_type'] == 'ora':
            connection_type = aau.Oracle(info)
        elif info['connection_type'] == 'sql':
            connection_type = aau.SQL(info)
        return connection_type

    @staticmethod
    def get_bucket_contents(state, config, **kwargs) -> Optional[list]:
        """
        Get bucket contents implementation allows for pathfilter to
        select specific sub-folders and files from a bucket
        """
        bucket = config['s3_info']['bucket']
        bucket_list = state['s3_con'].get_buckets(bucket, **kwargs)
        if len(bucket_list) == 0:
            print('S3 Bucket empty')
            return None
        bucket_list.sort()
        return bucket_list

    @staticmethod
    def get_dataframe_from_s3_bucket(state: dict, path: str, file: str) -> pd.DataFrame:
        """
        Load dataframe from S3 bucket given path and file
        """

        kwargs = {
            'path': path,
            'file': file
        }

        return state['s3_con'].read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
