

import numpy as np
import pandas as pd
import yaml
from src.utils.PathManager import Paths as Path 

class BayesianLabeler:
    def __init__(self,well_cd:str, 
                 data_features:list=['ROC_VOLTAGE'],
                 mode:str='all',
                 window_length:int=30,
                 bound_dict:dict={(0.0,0.96): {'S1':{0:7,1:2,2:1}, 'S0':{0:0,1:0,2:0}},
                                  (0.96,0.98):{'S1':{0:3,1:6,2:5}, 'S0':{0:5,1:5,2:5}}, 
                                  (0.98,10):  {'S1':{0:0,1:2,2:4}, 'S0':{0:5,1:5,2:5}}},
                 method:str='exp_mean',
                 num_days:int=3,
                 n_std:float=2.3
                 ):
        self.well_cd=well_cd 
        self.data_features = data_features 
        self.mode = mode 
        self.window_length = window_length
        self.bound_dict = bound_dict 
        self.method = method 
        self.num_days = num_days 
        self.n_std = n_std
        self.agg_df, self.label_df, self.raw_df, self.weather_df = self.get_data(self.well_cd)
        self.agg_df, self.rolling_mean, self.rolling_std, self.rolling_max = self.output()
        
    @staticmethod 
    def get_data(well_cd):
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
        label_df = label_df.loc[:,['labels']]
        weather_df = weather_df.loc[:,['cloudcover','direct_radiation']]
        raw_df = raw_df.loc[:,['ROC_VOLTAGE','FLOW','PRESSURE_TH']]
        return agg_df, label_df, raw_df, weather_df 
        
    @staticmethod 
    def get_integral_features(agg_df:pd.DataFrame, 
                            data_features:list=['ROC_VOLTAGE'],
                            mode:str='all'):
        agg_df = agg_df.loc[:,data_features]
        feature_list = []
        for feature in data_features:
            if mode == 'day':
                agg_df[f"{feature}_day_integral"] = agg_df[feature].apply(lambda x: np.array(x[:int(len(x)/2)]).sum()/(len(x[:int(len(x)/2)])+0.001) if hasattr(x,"__len__") else 0)
                feature_list = [f"{feature}_day_integral"]
            elif mode == 'night':
                agg_df[f"{feature}_night_integral"] = agg_df[feature].apply(lambda x: np.array(x[int(len(x)/2):]).sum()/(len(x[int(len(x)/2):])+0.001) if hasattr(x,"__len__") else 0)
                feature_list = [f"{feature}_night_integral"]
            elif mode == 'fullday':
                agg_df[f"{feature}_integral"] = agg_df[feature].apply(lambda x: np.array(x).sum()/(len(x)+0.001) if hasattr(x,"__len__") else 0)
                feature_list = [f"{feature}_integral"]
            else:
                agg_df[f"{feature}_day_integral"] = agg_df[feature].apply(lambda x: np.array(x[:int(len(x)/2)]).sum()/(len(x[:int(len(x)/2)])+0.001) if hasattr(x,"__len__") else 0)
                agg_df[f"{feature}_night_integral"] = agg_df[feature].apply(lambda x: np.array(x[int(len(x)/2):]).sum()/(len(x[int(len(x)/2):])+0.001) if hasattr(x,"__len__") else 0)
                feature_list = [f"{feature}_night_integral",f"{feature}_day_integral"]
        return agg_df, feature_list 
    
    @staticmethod 
    def get_nonzero_min(agg_df:pd.DataFrame, feature_list:list=[]):
        def get_robust_V(x):
            x = np.array(x[720:])
            x = x[x!=0]
            if len(x) <= 10:
                return np.nan
            return x.min()
        agg_df['minV'] = agg_df.ROC_VOLTAGE.apply(lambda x: get_robust_V(x))
        feature_list.append("minV")
        return agg_df, feature_list 
        
    
    
    @staticmethod 
    def get_running_features(agg_df:pd.DataFrame,
                            feature_list:list=['ROC_VOLTAGE_day_integral'],
                            window_length:int=30):
        sub_df = agg_df.loc[:,feature_list]
        rolling_window = sub_df.rolling(window=window_length)
        rolling_mean = rolling_window.mean()
        rolling_std = rolling_window.std()
        rolling_max = rolling_window.max()
        return sub_df,rolling_mean, rolling_std, rolling_max 
    
    @staticmethod 
    def get_weather_label(sub_df:pd.DataFrame,
                        rolling_mean:pd.DataFrame, 
                        rolling_std:pd.DataFrame, 
                        rolling_max:pd.DataFrame,
                        n_std:float=2.0,
                        combine_rule:str='or'):
        label=pd.DataFrame({},columns=sub_df.columns)
        label= (sub_df < rolling_mean - n_std*rolling_std).astype(np.int8)
        if combine_rule == 'or':
            label['weather_label'] = label.agg(lambda x: (x.sum()>0).astype(np.int8),axis=1) 
        else:
            label['weather_label'] = label.agg(lambda x: x.prod(),axis=1) 
        label['percentage_cloud'] = sub_df['ROC_VOLTAGE_day_integral']/rolling_max['ROC_VOLTAGE_day_integral']
        return label

    @staticmethod 
    def get_weather_normalised_dV(agg_df:pd.DataFrame, 
                                  window_length:int=15, 
                                  method:str='mean'):
        def get_mean_V(x):
            index = x.index
            sub_df = agg_df.loc[index,:]
            mask = sub_df[sub_df.weather_label == 0].index
            return sub_df.loc[mask].minV.mean()
        
        def get_median_V(x):
            index = x.index
            sub_df = agg_df.loc[index,:]
            mask = sub_df[sub_df.weather_label == 0].index
            if len(mask)==0:
                return np.nanmedian(sub_df.minV)
            else:
                return np.nanmedian(sub_df.loc[mask,'minV']) 
        
        def get_90_percentile_V(x):
            index = x.index
            sub_df = agg_df.loc[index,:]
            mask = sub_df[sub_df.weather_label == 0].index
            if len(mask)==0:
                return np.nanpercentile(sub_df.minV,90)
            else:
                return np.nanpercentile(sub_df.loc[mask,'minV'],90) 
            
        def get_10_percentile_V(x):
            index = x.index
            sub_df = agg_df.loc[index,:]
            mask = sub_df[sub_df.weather_label == 0].index
            if len(mask)==0:
                return np.nanpercentile(sub_df.minV,10)
            else:
                return np.nanpercentile(sub_df.loc[mask,'minV'],10) 
            
        def get_exp_mean_V(x):
            index = x.index
            sub_df = agg_df.loc[index,:]
            weight = np.arange(len(sub_df))
            alpha = -2/(window_length + 1)
            weight = np.power(alpha, weight)
            sub_df['weight'] = weight
            mask = sub_df[sub_df.weather_label == 0].index
            if len(mask) == 0:
                return np.nan
            return np.sum(sub_df.loc[mask].weight * sub_df.loc[mask].minV)/np.sum(sub_df.loc[mask].weight)
        
            
        if method == 'mean':
            mean_fn = get_mean_V
        elif method == 'median':
            mean_fn =  get_median_V
        elif method == "90_percentile":
            mean_fn = get_90_percentile_V
        elif method == "exp_mean":
            mean_fn = get_exp_mean_V
        elif method == '10_percentile':
            mean_fn = get_10_percentile_V
            
        agg_df['meanV']=agg_df['minV'].rolling(window=window_length).apply(mean_fn, raw=False)
        agg_df['dV'] = agg_df['minV'] - agg_df['meanV']
        agg_df['dV_normed'] = agg_df['minV']/agg_df['meanV']
        return agg_df

    @staticmethod 
    def get_conditional_probability(agg_df,num_days:int=3, bound_dict:dict = {(0.0,0.96): {'S1':{0:7,1:2,2:1}, 'S0':{0:0,1:0,2:0}},
                                                                              (0.96,0.98):{'S1':{0:3,1:6,2:5}, 'S0':{0:5,1:5,2:5}}, 
                                                                              (0.98,10):  {'S1':{0:0,1:2,2:4}, 'S0':{0:5,1:5,2:5}}}):
        lowerbounds = [i[0] for i in bound_dict.keys()]
        upperbounds = [i[1] for i in bound_dict.keys()]
        
        for day in range(num_days):
            assert(sum([j['S1'][day] for j in bound_dict.values()])==10)
            assert(sum([j['S0'][day] for j in bound_dict.values()])==10)
        
        def get_conditional_probability_(data, days=0):
            pos = 1
            neg = 1
            for i in range(days+1):
                pos *= data[f"P_dV_{upperbounds[data[f'dV_normed_group_{i}']]}_S1_{i}"]
                neg *= data[f"P_dV_{upperbounds[data[f'dV_normed_group_{i}']]}_S0_{i}"]
            return (pos*0.02)/(pos*0.02 + neg*0.98 + 1e-16)
        
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
                pos_df[f'thres_{upperbound}'] = np.cumsum(((pos_df[f'dV_normed_{day}'] > lowerbound)&(pos_df[f'dV_normed_{day}'] <= upperbound) ).astype(np.int8)) + bound_dict[(lowerbound,upperbound)]['S1'][day]
                agg_df[f'P_dV_{upperbound}_S1_{day}'] = pos_df[f'thres_{upperbound}']/(pos_df[f'n_occ_{day}']+10)
                agg_df.loc[:,[f'P_dV_{upperbound}_S1_{day}']] = agg_df.loc[:,[f'P_dV_{upperbound}_S1_{day}']].bfill().ffill()
                
                neg_df[f'thres_{upperbound}'] = np.cumsum(((neg_df[f'dV_normed_{day}'] > lowerbound)&(neg_df[f'dV_normed_{day}'] <= upperbound) ).astype(np.int8))+ bound_dict[(lowerbound,upperbound)]['S0'][day]
                agg_df[f'P_dV_{upperbound}_S0_{day}'] = neg_df[f'thres_{upperbound}']/(neg_df[f'n_occ_{day}'])
                agg_df.loc[:,[f'P_dV_{upperbound}_S0_{day}']] = agg_df.loc[:,[f'P_dV_{upperbound}_S0_{day}']].bfill().ffill()
            
        for day in range(num_days):
            agg_df[f"P_S1_{day}"] = agg_df.apply(lambda x: get_conditional_probability_(x,day),axis=1)
                
        return agg_df
    
    def output(self):
        agg_df, label_df = self.agg_df, self.label_df
        agg_df, feature_list = self.get_integral_features(agg_df, self.data_features, self.mode)
        agg_df, feature_list = self.get_nonzero_min(agg_df, feature_list)
        
        sub_df,rolling_mean,rolling_std,rolling_max = self.get_running_features(agg_df, feature_list, self.window_length)
        weather_label = self.get_weather_label(sub_df,rolling_mean,rolling_std,rolling_max,self.n_std)
        agg_df['weather_label'] = weather_label.weather_label
        agg_df = self.get_weather_normalised_dV(agg_df, self.window_length,self.method)
        agg_df['labels'] = label_df.labels
        agg_df = self.get_conditional_probability(agg_df,self.num_days,self.bound_dict)
        return agg_df, rolling_mean, rolling_std, rolling_max 