"""
Store config information.
@created 2022-01-20
"""
from datetime import datetime

def init():
    config = {
        'nn_info' : {
                'connection_type' : 's3',
                'user'   : None,
                'region' : 'ap-southeast-2',
                'bucket' : 'aa-roc-model-1-task0355600-poc-syd-174396275689',
            },
        'nn_sql' : """SELECT * FROM ROC_MODEL_NN WHERE WELL_CD = :W""",
        'nn_kwargs' : {
                'path' : 'ROC',
                'src_table' : 'ROC_MODEL_NN',
                'file' : 'ROC_MODEL_NEAREST_NEIGHBOURS.csv'
            },
        'procdata_info' : {
                'connection_type' : 's3',
                'user'   : None,
                'region' : 'ap-southeast-2',
                'bucket' : 'aa-roc-model-1-task0355600-poc-syd-174396275689',
            },
        'procdata_sql' : None,
        'procdata_kwargs' : {
                'args_ts' : ['TS','DTSMIN'],
                'path' : 'ROC/PROCESSED_DATA',
                'file' : None
            },
        'procdatarts_info' : [
                {
                    'type' : 's3',
                    'connection_info' : {
                        'user'   : None,
                        'region' : 'ap-southeast-2',
                        'bucket' : 'aa-roc-model-1-task0355600-poc-syd-174396275689',
                        'append' : False,
                        'path' : 'ROC/PROCESSED_DATA'
                        },
                    'file' : ['BROKA1_ROC_PROCESSED_DATA_20180501_20180601.csv', 'BROKA1_ROC_PROCESSED_DATA_20180601_20180701.csv'],
                    'pathfilter' : [{'ftype' : 'equals', 'filter' : 'ROC'},
                                    {'ftype' : 'equals', 'filter' : 'PROCESSED_DATA'},
                                    {'ftype' : 'prefix', 'filter' : 'BROKA1_ROC_PROCESSED_DATA_2'}],
                    'colnames' : ['DTSMIN', 'TS', 'ROC_VOLTAGE', 'VGAPS', 'ROC_VOLTAGE_I',
                                  'ROC_VOLTAGE_I_CLASS', 'ROC_VOLTAGE_IFILL', 'ROC_VOLTAGE_IFILL_CLASS',
                                  'ROC_VOLTAGE_IFILL_ERR', 'FLOW', 'FGAPS', 'FLOW_I', 'FLOW_I_CLASS',
                                  'PRESSURE_TH', 'PGAPS', 'PRESSURE_TH_I', 'PRESSURE_TH_I_CLASS'],
                    'tzname' : 'UTC',
                    'kwargs' : {
                            'read_string' : True,
                            'file' : None,
                            'append' : False
                        },
                    'read_size' : 1440, #{'min_read' : 0, 'max_read' : 1440},
                    'src_table' : 'T1'
                }
            ],
        'procdatarts_sql' : None,
        'procdatarts_kwargs' : {
                'args_ts' : ['TS','DTSMIN'],
                'path' : 'ROC/PROCESSED_DATA',
                'file' : None,
                'append' : False
            },
        'procdata_file_prefix' : 'ROC_PROCESSED_DATA',
        'procdata_file_ext'   : 'csv'
    }
    return config
