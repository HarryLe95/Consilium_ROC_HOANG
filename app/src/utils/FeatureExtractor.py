import numpy as np
import pandas as pd
from src.utils.PathManager import Paths as Path 
from typing import Sequence, Callable, List
from scipy.interpolate import CubicSpline, interp1d 

def get_absolute_minimum(y:list)->float:
    y = np.array(y)
    y = y[y!=0]
    if len(y) <= 10:
        return np.nan
    return y.min()

def get_minimum_feature(agg_df:pd.DataFrame, feature:str, min_function_handler:Callable = get_absolute_minimum)->pd.DataFrame:
        return agg_df[feature].apply(lambda x: min_function_handler(x))

def get_rolling_window(agg_df:pd.DataFrame, feature:str, window:int=30):
    return agg_df[feature].rolling(window=window)


class FeatureExtractor:
    def __init__(self, well_code:str):
        self.well_code = well_code
        
    @staticmethod 
    def get_label_df(well_cd:str)->pd.DataFrame:
        #Get label dataframe preprocessed
        try:
            label_df = pd.read_pickle(Path.data(f"{well_cd}_2016-01-01_2023-01-01_labelled.pkl"))
        except Exception as e:
            print(e)
            label_df = None 
        if label_df is not None: 
            label_df = label_df.loc[:,['labels']]
        return label_df 
    
    @staticmethod 
    def get_raw_df(well_cd:str, features:list[str]=["ROC_VOLTAGE","FLOW","PRESSURE_TH"])->pd.DataFrame:
        #Get raw dataframe unprocessed
        try: 
            raw_df = pd.read_csv(Path.data(f"{well_cd}_2016-01-01_2023-01-01_raw.csv"),index_col="TS",parse_dates=['TS'])
        except Exception as e: 
            print(e)
            raw_df = None 
        if raw_df is not None: 
            available_features = [f for f in features if f in raw_df.columns]
            available_features += [f"Mask_{f}" for f in available_features]
            raw_df = raw_df.loc[:,available_features]
        return raw_df 
    
    @staticmethod 
    def get_agg_df(raw_df:pd.DataFrame, label_df:pd.DataFrame, main_feature:str="ROC_VOLTAGE")->pd.DataFrame:
        #Get aggregated dataframe and remove missing rows
        if raw_df is None: 
            return None 
        agg_df = raw_df.groupby(raw_df.index.date).agg(list)
        agg_df["length"] = agg_df[main_feature].apply(lambda x: len(x))
        agg_df = agg_df[agg_df["length"]==1440]
        if label_df is not None: 
            agg_df["labels"] = label_df["labels"]
        agg_df.index = pd.to_datetime(agg_df.index)
        return agg_df 
         
    @classmethod  
    def get_data(cls, well_cd:str)->dict[str,pd.DataFrame|None]:
        #Get all data 
        raw_df = cls.get_raw_df(well_cd)
        label_df = cls.get_label_df(well_cd)
        agg_df = cls.get_agg_df(raw_df, label_df)
        return {"raw_df": raw_df, "label_df": label_df, "agg_df":agg_df}

    @classmethod
    def get_minimum_feature(cls, agg_df:pd.DataFrame, feature:str, method:str="absolute_minimum")->pd.Series:
        min_method_dict = {"absolute_minimum": get_absolute_minimum}
        min_function_handler = min_method_dict[method]
        return agg_df[feature].apply(lambda x: min_function_handler(x))
    
    @classmethod 
    def get_integral_feature(cls, agg_df:pd.DataFrame, feature:str) ->pd.Series:
        return agg_df[feature].apply(lambda x: np.trapz(x))
    
    @classmethod
    def get_relative_performance(cls, agg_df:pd.DataFrame, feature:str, window:int=30, method:str="max")->pd.Series:
        method_dict = {"max":np.nanmax, "min":np.nanmin, "median":np.nanmedian, "mean":np.nanmean}
        rolling_window = get_rolling_window(agg_df, feature, window)
        reference = rolling_window.apply(method_dict[method])
        return agg_df[feature]/reference
    
    @classmethod
    def get_interpolated_feature(cls, agg_df:pd.DataFrame, feature:str, method:str='linear')->pd.Series:
        mask_feature = f"Mask_{feature}"
        data = agg_df.loc[:,[feature,mask_feature]]
        def interpolate(x:pd.Series)->pd.Series:
            data = np.array(x[feature])
            mask = np.array(x[mask_feature])
            data[mask==0]=np.nan
            df = pd.DataFrame.from_dict({feature:data})
            df = df.interpolate(method,limit_direction='both')
            return df[feature].values
        data[feature] = data.apply(lambda x: interpolate(x),raw=False,axis=1)
        return data[feature]
    
    @classmethod
    def get_gradient_feature(cls, agg_df:pd.DataFrame, feature:str)->pd.Series:
        return agg_df[feature].apply(lambda x: np.gradient(x))
    
    @classmethod 
    def get_operation_corrected_feature(cls, agg_df:pd.DataFrame, feature:str, method:str='linear', gradient_threshold:float=0.025)->pd.Series:
        interpolated_feature = cls.get_interpolated_feature(agg_df, feature, method).to_frame()
        gradient = cls.get_gradient_feature(interpolated_feature, feature)
        
        def get_correction_region(x:pd.Series)->List[List]:
            neg_idx = np.array(np.where(x<=-gradient_threshold)[0])
            neg_idx = neg_idx[(neg_idx >= 600) & (neg_idx<=1200)]
            pos_idx = np.array(np.where(x>=gradient_threshold)[0])
            pos_idx = pos_idx[(pos_idx >=600) & (pos_idx<=1200)]

            if len(pos_idx)==0 or len(neg_idx)==0:
                return []
            pos_idx = np.array(pos_idx).reshape(-1,1)
            neg_idx = np.array(neg_idx).reshape(1,-1)
            mask = pos_idx > neg_idx
            neg_idx = neg_idx.reshape(-1,)[np.argmax(np.cumsum(mask,axis=1),axis=-1)]
            pos_idx = pos_idx.reshape(-1,)
            intervals = [[neg_idx[i],pos_idx[i]] for i in range(len(pos_idx))]

            joined_intervals = []
            for i,interval in enumerate(intervals):
                if i == 0:
                    export_interval = interval
                    continue
                if interval[0]!=export_interval[0]:
                    joined_intervals.append(export_interval)
                    export_interval = interval
                else:
                    export_interval[1] = intervals[i][1]
            joined_intervals.append(export_interval)

            return joined_intervals
        
        interpolated_feature['correction_region'] = gradient.apply(lambda x: get_correction_region(x))
        
        def correct(x:pd.Series)->pd.Series:
            intervals = x['correction_region']
            data = np.array(x[feature])
            if len(intervals)==0:
                return data
            for interval in intervals:
                dV = data[interval[0]] - data[interval[0]-1]
                data[interval[0]:interval[1]] = data[interval[0]:interval[1]]+dV
            return data
        return interpolated_feature.apply(lambda x: correct(x),axis=1,raw=False)
    
    @classmethod 
    def get_dawn_gradient_feature(cls, agg_df:pd.DataFrame, feature:str, method:str="linear")