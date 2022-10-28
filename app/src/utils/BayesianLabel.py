

import numpy as np
import pandas as pd
import yaml
from src.utils.PathManager import Paths as Path 


radiation_window = 15 
drop_radiation_LB = -50
integral_cloudcover_UB = 200
method='median'
window_length=30

def get_weather_normalised_dV(agg_df, weather_df, radiation_window=10, drop_radiation_LB=-50, integral_cloudcover_UB=200, cloudcover_label = 'int_cloudcover_day', window_length=15, method='weighted_mean'):
    
    def get_normal_weather_label(weather_df, radiation_window=10, drop_radiation_LB=-50, integral_cloudcover_UB=200, cloudcover_label='int_cloudcover_day'):
        weather_groupby = weather_df.groupby(weather_df.index.date)
        weather_extract = pd.DataFrame()
        weather_extract['max_radiation'] = weather_groupby.direct_radiation.max()
        weather_extract['int_cloudcover'] = weather_groupby.cloudcover.sum()
        weather_extract['int_cloudcover_day'] = weather_groupby.agg(list).cloudcover.apply(lambda x: np.sum(x[:int(len(x)/2)]))
        weather_extract.index = pd.to_datetime(weather_extract.index)
        weather_extract['dRad'] = weather_extract['max_radiation'] - weather_extract['max_radiation'].rolling(window=radiation_window, 
                                                                                                              min_periods=int(radiation_window/2)).mean()
        weather_extract['dCloud'] = weather_extract['int_cloudcover_day'] - weather_extract['int_cloudcover_day'].rolling(window=radiation_window,
                                                                                                                          min_periods=int(radiation_window/2)).mean()
        weather_extract['normal_day_label'] = (weather_extract['dRad'] >= drop_radiation_LB).astype(np.int8) * (weather_extract[cloudcover_label] <= integral_cloudcover_UB).astype(np.int8)
        return weather_extract
    
    def get_robust_V(x):
        x = np.array(x[720:])
        x = x[x!=0]
        if len(x) <= 10:
            return np.nan
        return x.min()

    def get_exp_weighted_V(x):
        index = x.index
        sub_df = agg_df.loc[index,:]
        weight = np.arange(len(sub_df))
        alpha = 2/(window_length + 1)
        weight = np.power(alpha, weight)
        sub_df['weight'] = weight
        mask = sub_df[sub_df.normal_day_label == 1].index
        if len(mask) == 0:
            return np.nan
        return np.sum(sub_df.loc[mask].weight * sub_df.loc[mask].minV)/np.sum(sub_df.loc[mask].weight)
    
    def get_mean_V(x):
        index = x.index
        sub_df = agg_df.loc[index,:]
        mask = sub_df[sub_df.normal_day_label == 1].index
        return sub_df.loc[mask].minV.mean()
    
    def get_median_V(x):
        index = x.index
        sub_df = agg_df.loc[index,:]
        mask = sub_df[sub_df.normal_day_label == 1].index
        if len(mask)==0:
            return np.nanmedian(sub_df.minV)
        else:
            return np.nanmedian(sub_df.loc[mask,'minV']) 
        
    weather_extract = get_normal_weather_label(weather_df, radiation_window, drop_radiation_LB, integral_cloudcover_UB)
    weather_extract = weather_extract.loc[agg_df.index.min().date():agg_df.index.max().date()]

    for col in weather_extract.columns:
        agg_df[col] = weather_extract[col]
    if method == 'weighted_mean':
        mean_fn = get_exp_weighted_V
    elif method == 'mean':
        mean_fn = get_mean_V
    elif method == 'median':
        mean_fn =  get_median_V
    mean_fn = get_median_V
    agg_df['minV'] = agg_df.ROC_VOLTAGE.apply(lambda x: get_robust_V(x))
    agg_df['meanV']=agg_df['minV'].rolling(window=window_length, min_periods=int(window_length/2)).apply(mean_fn, raw=False)
    agg_df['dV'] = agg_df['minV'] - agg_df['meanV']
    agg_df['dV_normed'] = agg_df['minV']/agg_df['meanV']
    return agg_df, weather_extract

def get_conditional_probability(agg_df,base_rate=0.0198, bins = [0.96, 0.98], num_days=3):
    lowerbounds = [0] + bins
    upperbounds = bins + [10]
    
    def get_conditional_probability_(data, days=0):
        pos = 1
        neg = 1
        for i in range(days+1):
            pos *= data[f"P_dV_{upperbounds[data[f'dV_normed_group_{i}']]}_S1_{i}"]
            neg *= data[f"P_dV_{upperbounds[data[f'dV_normed_group_{i}']]}_S0_{i}"]
        return (pos*0.02)/(pos*0.02 + neg*0.98 + 1e-5)
    
    for day in range(num_days):
        agg_df[f'P_S1_{day}'] = 0
        agg_df[f"dV_normed_{day}"] = agg_df['dV_normed'].shift(day,fill_value=0)
        
        agg_df[f"dV_normed_group_{day}"] = np.sum([agg_df[f"dV_normed_{day}"] > bound for bound in upperbounds],axis=0) 
        
        pos_df = agg_df[agg_df.labels.isin([2,5])].loc[:,[f'dV_normed_{day}','labels']]
        pos_df[f'n_occ_{day}'] = np.arange(1,len(pos_df)+1)
        
        neg_df = agg_df[~agg_df.labels.isin([2,5])].loc[:,[f'dV_normed_{day}','labels']]
        neg_df.dropna(subset=f'dV_normed_{day}',inplace=True)
        neg_df[f'n_occ_{day}'] = np.arange(1,len(neg_df)+1)
        
        for lowerbound, upperbound in zip(lowerbounds, upperbounds):
            pos_df[f'thres_{upperbound}'] = np.cumsum(((pos_df[f'dV_normed_{day}'] > lowerbound)&(pos_df[f'dV_normed_{day}'] <= upperbound) ).astype(np.int8))
            agg_df[f'P_dV_{upperbound}_S1_{day}'] = pos_df[f'thres_{upperbound}']/pos_df[f'n_occ_{day}']
            agg_df.loc[:,[f'P_dV_{upperbound}_S1_{day}']] = agg_df.loc[:,[f'P_dV_{upperbound}_S1_{day}']].bfill().ffill()
            
            neg_df[f'thres_{upperbound}'] = np.cumsum(((neg_df[f'dV_normed_{day}'] > lowerbound)&(neg_df[f'dV_normed_{day}'] <= upperbound) ).astype(np.int8))
            agg_df[f'P_dV_{upperbound}_S0_{day}'] = neg_df[f'thres_{upperbound}']/neg_df[f'n_occ_{day}']
            agg_df.loc[:,[f'P_dV_{upperbound}_S0_{day}']] = agg_df.loc[:,[f'P_dV_{upperbound}_S0_{day}']].bfill().ffill()
        
    for day in range(num_days):
        agg_df[f"P_S1_{day}"] = agg_df.apply(lambda x: get_conditional_probability_(x,day),axis=1)
            
    return agg_df

def get_data(well_cd,radiation_window = radiation_window, drop_radiation_LB = drop_radiation_LB, integral_cloudcover_UB = integral_cloudcover_UB, method = method, window_length = window_length):
    with open(Path.config("nearest_station.yaml"),'r') as file:
        station_dict = yaml.safe_load(file)

    station_name = station_dict[well_cd]
    try:
        label_df = pd.read_pickle(Path.data(f"{well_cd}_2016-01-01_2023-01-01_labelled.pkl"))
    except Exception as e:
        print(e)
        label_df = None 
    raw_df = pd.read_csv(Path.data(f"{well_cd}_2016-01-01_2023-01-01_raw.csv"),index_col="TS",parse_dates=['TS'])
    weather_df = pd.read_csv(Path.data(f"{station_name}_2016-01-01_2023-01-01_weather.csv"), index_col="TS",parse_dates=['TS'])

    agg_df = raw_df.groupby(raw_df.index.date).agg(list)
    agg_df["length"]=agg_df.ROC_VOLTAGE.apply(lambda x:len(x))
    agg_df = agg_df[agg_df.length==1440]
    if label_df is not None:
        agg_df["labels"] = label_df.labels
    agg_df.index = pd.to_datetime(agg_df.index)
    agg_df, weather_extract = get_weather_normalised_dV(agg_df, weather_df=weather_df, radiation_window = radiation_window, drop_radiation_LB = drop_radiation_LB, integral_cloudcover_UB = integral_cloudcover_UB, method=method, window_length=window_length)
    return agg_df, weather_extract, label_df, raw_df, weather_df 