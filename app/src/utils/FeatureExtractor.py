import numpy as np
import pandas as pd
from src.utils.PathManager import Paths as Path 
from typing import Sequence, Callable, List
from itertools import groupby
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

class DataLoader:
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

class FeatureExtractor_:
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
    

class IntervalUtility:
    def __init__(self, threshold:float, start:float=600, end:float=1200, feature:str="ROC_VOLTAGE", min_length:int=60):
        self.start=start
        self.end=end 
        self.feature=feature
        self.threshold = threshold 
        self.min_length = min_length
        
    def get_correction_region(self, x:pd.Series)->List[List]:
        neg_idx = np.array(np.where(x<=-self.threshold)[0])
        neg_idx = neg_idx[(neg_idx >= self.start) & (neg_idx<=self.end)]
        pos_idx = np.array(np.where(x>=self.threshold)[0])
        pos_idx = pos_idx[(pos_idx >=self.start) & (pos_idx<=self.end)]

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

    def correct(self,x:pd.Series)->pd.Series:
        intervals = x['correction_region']
        data = np.array(x[self.feature])
        if len(intervals)==0:
            return data
        for interval in intervals:
            dV = data[interval[0]] - data[interval[0]-1]
            data[interval[0]:interval[1]] = data[interval[0]:interval[1]]+dV
        return data
    
    def get_largest_outage_interval(self, mask:pd.Series, start:float, end:float)->tuple:
        """Get (start,end) tuple specifying the longest consecutive zeros in a sequence

        Args:
            mask (pd.Series): mask sequence - binary 
            start (float): start of period
            end (float): end of period 

        Returns:
            tuple: (start, end) if valid. None if either there is no 0 in mask or if mask is None
        """
                
        if mask is None:
            return None
        mask = np.array(mask[start:end])
        index = 0
        zero_index = []
        zero_length = [] 
        for mask_val,mask_group in groupby(mask):
            new_length = len(list(mask_group))
            if mask_val == 0: 
                zero_index.append(index)
                zero_length.append(new_length)
            index+=new_length
        zero_index = np.array(zero_index)
        zero_length = np.array(zero_length)

        zero_index = zero_index[zero_length >= self.min_length]
        zero_length = zero_length[zero_length >= self.min_length]

        if len(zero_length) == 0:
            return None 
        if len(zero_length) == len(mask):
            return None 
        index = np.argmax(zero_length)
        start_index = zero_index[index]
        end_index = start_index + zero_length[index]
        return start_index, end_index

    @classmethod 
    def verify_intersection(cls, intervals:list[tuple])->int:
        """Verify that the intervals overlaps - i.e the intersection of all intervals is nonzero

        Args:
            intervals (list[tuple]): list of intervals. Tuple contains the lower_bound and upper_bound of the interval

        Returns:
            int: 0 if no overlap, 1 if there is overlap, 2 if all data is None 
        """
        lb = np.array([i[0] for i in intervals if i is not None])
        ub = np.array([i[1] for i in intervals if i is not None])
        if len(lb)==0:
            return 2
        sup_lb = lb.max()
        inf_ub = ub.min()
        return int(sup_lb < inf_ub)
    
class FeatureExtractor(FeatureExtractor_):
    def __init__(self, well_code:str, feature:str="ROC_VOLTAGE")->None:
        self.well_code = well_code
        self.loader = DataLoader()
        self.data = self.get_data()
        self.feature = feature
        self.interpolated_data = self.get_interpolated_feature(self.data['agg_df'],self.feature).to_frame()
        
    def get_data(self)->dict[str,pd.DataFrame]:
        return self.loader.get_data(self.well_code)
        
    def get_operation_corrected_feature(self, gradient_threshold:float=0.025,start:float=600,end:float=1200)->pd.Series:
        agg_df = self.data['agg_df']
        interpolated_feature = self.get_interpolated_feature(agg_df, self.feature, "linear").to_frame()
        gradient = self.get_gradient_feature(interpolated_feature, self.feature)
        region_identifier = IntervalUtility(gradient_threshold,start,end)
        interpolated_feature['correction_region'] = gradient.apply(lambda x: region_identifier.get_correction_region(x))
        return interpolated_feature.apply(lambda x: region_identifier.correct(x),axis=1,raw=False)
    
    def get_data_outage_failure_label(self, start:float=600, end:float=1400, min_length:int=60)->pd.Series:
        """Check for presence of battery failure that cause data outage for all tags.
        
        In many wells, failures right before dawn correspond to a gap in all tag records (ROC_VOLTAGE, FLOW, PRESSURE_TH) in the same period.
        This method works on raw, uninterpolated data. 
        Failure labels assigned by:
        - Getting periods of no flow (when FLOW data was non-zero before data outage). If FLOW data was zero through-out, skip this step.
        - Getting periods of 0 pressure (when PRESSURE_TH drops to zero). If PRESSURE_TH tag is unavalable, skip this step.
        - Getting period of missing/zero ROC_VOTALGE. If the entire period was missing - get uncertain label - i.e. label 2
        If the three periods overlap, assign data-outage failure label (1). If both PRESSURE_TH is unavailable and FLOW data is zero throughout OR 
        If data for the entire start-end period is missing, return uncertain lable.

        Args:
            start (float, optional): index of the start of evening. Defaults to 600.
            end (float, optional): index of the end of evening and early morning. Defaults to 1400.
            min_count (int, optional): minumum number of minutes in an outage event for it to be classified as data outage. Defaults to 60 mins = 1 hour.

        Returns:
            pd.Series: label - 0: no failure, 1: has failure, 2: uncertain
        """
        VOLTAGE_data = self.data['agg_df']['Mask_ROC_VOLTAGE']
        FLOW_data = self.data['agg_df']['Mask_FLOW']
        PRESSURE_data = self.data['agg_df']['Mask_PRESSURE_TH'] if 'Mask_PRESSURE_TH' in self.data['agg_df'] else None 
        
        labeler = IntervalUtility(threshold=0.05,start=start,end=end,min_length=min_length)
        temp_df = pd.DataFrame()
        temp_df['VOLTAGE_interval'] = VOLTAGE_data.apply(lambda x: labeler.get_largest_outage_interval(x, start, end))
        temp_df['FLOW_interval'] = FLOW_data.apply(lambda x: labeler.get_largest_outage_interval(x, start, end))
        temp_df['PRESSURE_interval'] = PRESSURE_data.apply(lambda x: labeler.get_largest_outage_interval(x, start, end)) if PRESSURE_data is not None else None 
        
        def label_outage(x:pd.Series)->pd.Series:
            if x['VOLTAGE_interval'] is None:
                return 2 
            if x['PRESSURE_interval'] is None and x['FLOW_interval'] is None:
                return 2 
            return IntervalUtility.verify_intersection([x['VOLTAGE_interval'],x['FLOW_interval'],x['PRESSURE_interval']])
            
        return temp_df, temp_df.apply(lambda x: label_outage(x), raw=False,axis=1)
        
    
    def get_dawn_VOLTAGE_drop_failure_label(self, 
                                            first_derivative_threshold:float=0.4, 
                                            second_derivative_threshold:float=0.4, 
                                            use_second_derivate:bool=True)->pd.Series: 
        """Check for presence of battery failure that cause a sharp drop in ROC_VOLTAGE before dawn. 
        
        This method computes the first and second derivatives and assigns labels by thresholding. This method works on interpolated data. 
        
        Args:
            first_derivative_threshold (float, optional): threshold of the first derivative. Defaults to 0.4.
            second_derivative_threshold (float, optional): threshold of the second derivative. Defaults to 0.4.
            use_second_derivate (bool, optional): whether to use 2nd derivative to derive label. Defaults to True.

        Returns:
            pd.Series: failure label. 0 - no failure, 1 - failure present 
        """
        first_derivative = self.get_gradient_feature(self.interpolated_data.to_frame(), self.feature)
        first_derivative_label = first_derivative.apply(lambda x: np.any(np.abs(x[600:1400])>=first_derivative_threshold))
        if use_second_derivate:
            second_derivative = self.get_gradient_feature(first_derivative.to_frame(), self.feature)
            second_derivative_label = second_derivative.apply(lambda x: np.any(np.abs(x[600:1400])>=second_derivative_threshold))
            return first_derivative_label*second_derivative_label
        return first_derivative_label
    
    
    def get_well_shutin_failure_label(self)->pd.Series:
        """Check for presence of battery failure causing shutins. 
        
        Low battery VOlTAGE can cause unexpected shut-ins. Shut-ins can be characterised by periods in which FLOW data drops to zero (from non-zero) and PRESSURE_TH sharply increases.
        This period should also correspond to a drop in ROC_VOLTAGE.

        Returns:
            pd.Series: _description_
        """
        pass 