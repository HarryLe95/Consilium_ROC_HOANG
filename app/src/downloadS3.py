"""
Created by Hoang Son Le - 15/08/2022
This is a CLI tool to process data and download to the right folder. Work with bash
"""
import os
import pandas as pd 
from datetime import datetime
from roc_manual_labels import state, get_filestartend, config
from aau.advancedanalytics_util import S3 as S3
import argparse
#ROC/SOLAR_DATA

def get_filename(item_cd:str, 
                prefix:str, 
                start:datetime, 
                end:datetime, 
                file_ext:str='.csv'):
    fn = '{}_{}_{}_{}.{}'.format(item_cd, 
                                 prefix, 
                                 start.strftime('%Y%m%d'), 
                                 end.strftime('%Y%m%d'), 
                                 file_ext)
    return fn

def parse_date(date:str):
    try:
        ret = pd.to_datetime(date, format='%Y-%m-%d')
        return ret
    except:
        raise ValueError(f"Input str {date} must be in the %Y-%m-%d format")

def get_date_range(start_date:str, end_date:str):
    start_date = parse_date(start_date)
    end_date = parse_date(end_date)
    return pd.date_range(start_date, end_date, freq='MS')

def get_weather_data(S3,
                     prefix:str,
                     item_cd:str,
                     file_ext:str,
                     start_date:str,
                     end_date:str,
                     **kwargs):
    alldf = pd.DataFrame()
    date_range = get_date_range(start_date, end_date)
    for d in range(len(date_range)-1):
        fs, fe = date_range[d], date_range[d+1]
        fn = get_filename(item_cd=item_cd, prefix=prefix, start=fs, end=fe, file_ext=file_ext)
        kwargs['file'] = fn
        try:
            result = S3.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
            alldf = pd.concat([alldf, result], ignore_index=True)   
        except:
            print ('{} preset target {} out of data range'.format(item_cd, d))
            continue
    import pdb; pdb.set_trace()
    alldf['TS']=pd.to_datetime(alldf['TS'])
    alldf.set_index("TS",inplace=True)

    date_range = pd.date_range(alldf.index.min(), alldf.index.max(), freq='H')
    alldf = alldf.groupby(alldf.index).mean()
    alldf = alldf.reindex(date_range)
    alldf.index.name = "TS"
    alldf.reset_index(inplace=True)
    return alldf
    

def get_raw_data(S3, 
                prefix: str, 
                item_cd:str,  
                file_ext:str,
                start_date:str,
                end_date:str, 
                **kwargs):
    alldf = pd.DataFrame()
    date_range = get_date_range(start_date, end_date)
    for d in range(len(date_range)-1):
        fs, fe = date_range[d], date_range[d+1]
        fn = get_filename(item_cd=item_cd, prefix=prefix, start=fs, end=fe, file_ext=file_ext)
        kwargs['file'] = fn
        try:
            result = S3.read(sql=None, args={}, edit=[], orient='df', do_raise=False, **kwargs)
            alldf = pd.concat([alldf, result], ignore_index=True)   
        except:
            print ('{} preset target {} out of data range'.format(item_cd, d))
            continue
    
    alldf['TS']=pd.to_datetime(alldf['TS'])
    alldf.set_index("TS",inplace=True)
    alldf = alldf.loc[:,['ROC_VOLTAGE','FLOW','PRESSURE_TH']]

    date_range = pd.date_range(alldf.index.min(), alldf.index.max(), freq='T')
    alldf = alldf.groupby(alldf.index).mean()
    alldf = alldf.reindex(date_range)
    alldf['Mask_ROC_VOLTAGE']=1-alldf.ROC_VOLTAGE.isna()
    alldf['Mask_FLOW']=1-alldf.FLOW.isna()
    alldf['Mask_PRESSURE_TH']=1-alldf.PRESSURE_TH.isna()
    alldf.interpolate(method='linear', inplace=True, limit_direction='both')
    alldf.index.name = "TS"
    alldf.reset_index(inplace=True)

    return alldf

def get_labelled_data(S3, 
                prefix: str, 
                item_cd:str,  
                file_ext:str,
                label:dict,
                **kwargs):
    label_df = pd.DataFrame({"TS":label.keys(), "labels":label.values()})
    label_df.TS = pd.to_datetime(label_df.TS)
    label_df.set_index("TS",inplace=True)
    return label_df

def main(state, config):
    parser=argparse.ArgumentParser(description="Download data from S3 bucket. Used to download Raw data, Labelled Data, and Weather Data")
    parser.add_argument("-item_cd", type=str,  default='all',help="Either well_cd or area code (for weather data). Default will download everything")
    parser.add_argument("-start_date", type=str, default='2016-01-01', help="Start period for unlabelled data" )
    parser.add_argument("-end_date", type=str, default='2022-09-05',help="End period for unlabelled data")
    parser.add_argument("-save_folder",type=str,default='csv_files', help='Where to save the downloaded files')
    parser.add_argument("-mode", choices=['raw','label','weather'], default='raw')
    parser.add_argument("-prefix", type=str, default=None, help="S3 csv file prefix")
    parser.add_argument("-path", type=str, default=None, help="S3 subfolder")
    args = parser.parse_args()

    data_config={}
    data_config['args_ts'] = config['procdatarts_kwargs']['args_ts']
    data_config['prefix'] = args.prefix if args.prefix else config['procdata_file_prefix']
    data_config['path'] = args.path if args.path else config['procdatarts_kwargs']['path']
    data_config['file_ext'] = config['procdata_file_ext']
    data_config['S3'] = state['procdata_con']

    def _download(item_cd):
        os.makedirs(args.save_folder,exist_ok=True)
        if args.mode == 'label':
            label = state['preset_targets'][item_cd]
            output_df = get_labelled_data(label=label,item_cd=item_cd,**data_config)
            save_file = os.path.join(args.save_folder, f'{item_cd}_labelled.pkl')
        elif args.mode == 'raw':
            output_df = get_raw_data(start_date=args.start_date,end_date=args.end_date,item_cd=item_cd,**data_config)
            save_file = os.path.join(args.save_folder, f'{item_cd}_raw.csv')
        elif args.mode == 'weather':
            output_df = get_weather_data(start_date=args.start_date,end_date=args.end_date,item_cd=item_cd,**data_config)
            save_file = os.path.join(args.save_folder, f'{item_cd}_weather.csv')
        else:
            raise ValueError(f"Invalid mode entered: {args.mode}")
        
        print(f"Writing to {save_file}")
        if '.pkl' in save_file:
            output_df.to_pickle(save_file)
        else:
            output_df.to_csv(save_file,index=False)

    if args.item_cd == "all":
        if args.mode == 'weather':
            raise ValueError("Weather mode and item_cd=all are incompatible")
        for item_cd in state['preset_targets']:
            _download(item_cd)
    else:
        _download(args.item_cd)

    
if __name__ == "__main__":
    main(state,config)
    

