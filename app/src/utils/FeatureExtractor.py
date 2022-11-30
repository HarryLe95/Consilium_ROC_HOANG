import numpy as np
import pandas as pd
from src.utils.PathManager import Paths as Path 
from typing import Sequence, Callable, List
from itertools import groupby
from scipy.interpolate import CubicSpline, interp1d 
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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
        if len(zero_length) >= 0.8*len(mask):
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
    def __init__(self, well_code:str, feature:str="ROC_VOLTAGE", 
                 operation_correction_dict:dict={"gradient_threshold":0.025,"start":600,"end":1200},
                 anomaly_detection_dict:dict={"missing_rate":0.8},
                 data_outage_detection_dict:dict={"start":1000, "end":1400, "min_length":60, "missing_rate":0.8},
                 dawn_VOLTAGE_drop_detection_dict:dict={"first_derivative_threshold":0.4, "second_derivative_threshold":0.4,"use_second_derivative":True},
                 charging_fault_detection_dict:dict={"max_error_threshold":0.08, "mean_absolute_error_threshold":0.02,"mean_squared_error_threshold":0.008,"r2_threshold":0.7},
                 )->None:
        self._verify_operation_correction_dict(operation_correction_dict)
        self._verify_anomaly_detection_dict(anomaly_detection_dict)
        self._verify_data_outage_detection_dict(data_outage_detection_dict)
        self._verify_dawn_VOLTAGE_drop_detection_dict(dawn_VOLTAGE_drop_detection_dict)
        self._verify_charging_fault_detection_dict(charging_fault_detection_dict)
        self.well_code = well_code
        self.loader = DataLoader()
        self.data = self.get_data()
        self.feature = feature
        self.interpolated_data = self.get_interpolated_feature(self.data['agg_df'],self.feature).to_frame()
    
    def _verify_operation_correction_dict(self, data_dict:dict)->None:
        gradient_threshold = 0.025 if 'gradient_threshold' not in data_dict else data_dict['gradient_threshold']
        start = 600 if 'start' not in data_dict else data_dict['start']
        end = 1200 if 'end' not in data_dict else data_dict['end']
        self.operation_correction_dict={"gradient_threshold":gradient_threshold, "start":start, "end":end}
        
    def _verify_data_outage_detection_dict(self, data_dict:dict)->None:
        min_length = 60 if 'min_length' not in data_dict else data_dict['min_length']
        start = 1000 if 'start' not in data_dict else data_dict['start']
        end = 1200 if 'end' not in data_dict else data_dict['end']
        missing_rate = 0.8 if "missing_rate" not in data_dict else data_dict['missing_rate']
        self.data_outage_detection_dict={"min_length":min_length, "start":start, "end":end, "missing_rate":missing_rate}
        
    def _verify_anomaly_detection_dict(self, data_dict:dict)->None:
        missing_rate = 0.8 if "missing_rate" not in data_dict else data_dict['missing_rate']
        self.anomaly_detection_dict={"missing_rate":missing_rate}
    
    def _verify_dawn_VOLTAGE_drop_detection_dict(self, data_dict:dict)->None:
        first_derivative_threshold=0.4 if "first_derivative_threshold" not in data_dict else data_dict['first_derivative_threshold']
        second_derivative_threshold=0.4 if "second_derivative_threshold" not in data_dict else data_dict["second_derivative_threshold"] 
        use_second_derivate=True if "use_second_derivate" not in data_dict else data_dict["use_second_derivate"]
        self.dawn_VOLTAGE_drop_detection_dict={"first_derivative_threshold":first_derivative_threshold, "second_derivative_threshold":second_derivative_threshold, "use_second_derivative":use_second_derivate}
             
    def _verify_charging_fault_detection_dict(self, data_dict:dict)->None:
        max_error_threshold = 0.08 if "max_error_threshold" not in data_dict else data_dict["max_error_threshold"]
        mean_absolute_error_threshold= 0.02 if "mean_absolute_error_threshold" not in data_dict else data_dict["mean_absolute_error_threshold"]
        mean_squared_error_threshold=0.008 if "mean_squared_error_threshold" not in data_dict else data_dict["mean_squared_error_threshold"]
        r2_threshold=0.7 if "r2_threshold" not in data_dict else data_dict["r2_threshold"]
        self.charging_fault_detection_dict = {"max_error_threshold":max_error_threshold,"mean_absolute_error_threshold":mean_absolute_error_threshold,"mean_squared_error_threshold":mean_squared_error_threshold,"r2_threshold":r2_threshold}
    
    def get_data(self)->dict[str,pd.DataFrame]:
        return self.loader.get_data(self.well_code)
        
    def get_operation_corrected_feature(self)->pd.Series:
        agg_df = self.data['agg_df']
        interpolated_feature = self.get_interpolated_feature(agg_df, self.feature, "linear").to_frame()
        gradient = self.get_gradient_feature(interpolated_feature, self.feature)
        region_identifier = IntervalUtility(self.operation_correction_dict['gradient_threshold'], self.operation_correction_dict['start'], self.operation_correction_dict['end'])
        interpolated_feature['correction_region'] = gradient.apply(lambda x: region_identifier.get_correction_region(x))
        return interpolated_feature.apply(lambda x: region_identifier.correct(x),axis=1,raw=False)
    
    def get_data_anomaly_label(self)->pd.Series:
        """Assign data anomaly label based on rate of missing ROC_VOLTAGE data 
        
        Args - embedded in anomaly_detection_dict
            missing_rate (float): fraction of missing data in a day for it to be considered data anomaly. Defaults to 0.8

        Returns:
            pd.Series: 0 - not abnormal, 1 - data anomaly 
        """
        VOLTAGE_data = self.data['agg_df']['Mask_ROC_VOLTAGE']
        return VOLTAGE_data.apply(lambda x: int((1-sum(x)/len(x))>=self.anomaly_detection_dict['missing_rate']))
    
    def get_data_outage_failure_label(self)->pd.Series:
        """Check for presence of battery failure that cause data outage for all tags.
        
        In many wells, failures right before dawn correspond to a gap in all tag records (ROC_VOLTAGE, FLOW, PRESSURE_TH) in the same period.
        This method works on raw, uninterpolated data. 
        Failure labels assigned by:
        - Getting periods of no flow (when FLOW data was non-zero before data outage). If FLOW data was zero through-out, skip this step.
        - Getting periods of 0 pressure (when PRESSURE_TH drops to zero). If PRESSURE_TH tag is unavalable, skip this step.
        - Getting period of missing/zero ROC_VOTALGE. If the entire period was missing - get uncertain label - i.e. label 2
        If the three periods overlap, assign data-outage failure label (1). If both PRESSURE_TH is unavailable and FLOW data is zero throughout OR 
        If data for the entire start-end period is missing, return uncertain lable.

        Args: embedded in data_outage_detection_dict
            start (float, optional): index of the start of evening. Defaults to 600.
            end (float, optional): index of the end of evening and early morning. Defaults to 1400.
            min_count (int, optional): minumum number of minutes in an outage event for it to be classified as data outage. Defaults to 60 mins = 1 hour.
            missing_rate (float, optional): fraction of missing ROC_VOLTAGE data for a day to be considered abnormal. Defaults to 0.8 - If 80% of data is missing -> Data Anomaly

        Returns:
            pd.Series: label - 0: no failure, 1: has failure, 2: uncertain/data anomaly
        """
        VOLTAGE_data = self.data['agg_df']['Mask_ROC_VOLTAGE']
        FLOW_data = self.data['agg_df']['Mask_FLOW']
        PRESSURE_data = self.data['agg_df']['Mask_PRESSURE_TH'] if 'Mask_PRESSURE_TH' in self.data['agg_df'] else None 
        
        labeler = IntervalUtility(threshold=0.05,start=self.data_outage_detection_dict['start'],
                                  end=self.data_outage_detection_dict['end'],
                                  min_length=self.data_outage_detection_dict['min_length'])
        temp_df = pd.DataFrame()
        temp_df['VOLTAGE_interval'] = VOLTAGE_data.apply(lambda x: labeler.get_largest_outage_interval(x, self.data_outage_detection_dict['start'], self.data_outage_detection_dict['end']))
        temp_df['FLOW_interval'] = FLOW_data.apply(lambda x: labeler.get_largest_outage_interval(x, self.data_outage_detection_dict['start'], self.data_outage_detection_dict['end']))
        temp_df['PRESSURE_interval'] = PRESSURE_data.apply(lambda x: labeler.get_largest_outage_interval(x, self.data_outage_detection_dict['start'], self.data_outage_detection_dict['end'])) if PRESSURE_data is not None else None 
        
        def label_outage(x:pd.Series)->pd.Series:
            if x['VOLTAGE_interval'] is None:
                return 2 
            if x['PRESSURE_interval'] is None and x['FLOW_interval'] is None:
                return 2 
            return IntervalUtility.verify_intersection([x['VOLTAGE_interval'],x['FLOW_interval'],x['PRESSURE_interval']])
        
        #Post Processing - remove labels caused by data anomaly - i.e. long periods of missing ROC_VOLTAGE data
        outage_label_df = temp_df.apply(lambda x: label_outage(x), raw=False,axis=1)
        anomaly_label = self.get_data_anomaly_label()

        anomaly_index = anomaly_label[anomaly_label==1].index
        outage_label_df[outage_label_df.index.isin(anomaly_index)]=2
        
        return outage_label_df
        
    def get_dawn_VOLTAGE_drop_failure_label(self)->pd.Series: 
        """Check for presence of battery failure that cause a sharp drop in ROC_VOLTAGE before dawn. 
        
        This method computes the first and second derivatives and assigns labels by thresholding. This method works on interpolated data. 
        
        Args: embedded in dawn_VOLTAGE_drop_detection_dict
            first_derivative_threshold (float, optional): threshold of the first derivative. Defaults to 0.4.
            second_derivative_threshold (float, optional): threshold of the second derivative. Defaults to 0.4.
            use_second_derivate (bool, optional): whether to use 2nd derivative to derive label. Defaults to True.

        Returns:
            pd.Series: failure label. 0 - no failure, 1 - failure present 
        """
        first_derivative = self.get_gradient_feature(self.interpolated_data, self.feature)
        first_derivative_label = first_derivative.apply(lambda x: int(np.any(np.abs(x[600:1400])>=self.dawn_VOLTAGE_drop_detection_dict['first_derivative_threshold'])))
        if self.dawn_VOLTAGE_drop_detection_dict['use_second_derivative']:
            second_derivative = self.get_gradient_feature(first_derivative.to_frame(), self.feature)
            second_derivative_label = second_derivative.apply(lambda x: int(np.any(np.abs(x[600:1400])>=self.dawn_VOLTAGE_drop_detection_dict['second_derivative_threshold'])))
            return first_derivative_label*second_derivative_label
        return first_derivative_label
    
    def get_charging_fault_label(self)->pd.Series:
        """Get charging fault label.
        
        Charging Fault results in a continuous decline of ROC_VOLTAGE that is almost monotonic. A linear regressor is fitted on the data, 
        with different fitness metrics compared against thresholds to provide label.

        Args: embedded in charging_fault_config_dict
            max_error_threshold (float): max error 
            mean_squared_error_threshold (float): mse threshold 
            mean_absolute_error_threshold (float): mae threshold
            r2_threshold (float): r2 score threshold

        Returns:
            pd.Series: 0 - no charging fault, 1 - charging fault, 2 - data anomaly 
        """
        
        df = self.data['agg_df']
        
        def get_charging_fault_label_(data):
            Y = np.array(data['ROC_VOLTAGE'])
            X = np.arange(len(Y))
            mask = np.array(data['Mask_ROC_VOLTAGE'])
            Y_ = Y[mask==1].reshape(-1,1)
            X_ = X[mask==1].reshape(-1,1)
            if len(Y_)<=0.10*len(Y):
                return 2 
            reg = LinearRegression().fit(X_,Y_)
            Y_pred = reg.predict(X_)
            max_e = max_error(Y_,Y_pred)
            mae_e = mean_absolute_error(Y_,Y_pred)
            mse_e = mean_squared_error(Y_,Y_pred)
            r2_e = r2_score(Y_,Y_pred)
            return int((max_e <= self.charging_fault_detection_dict['max_error_threshold']) and
                       (mae_e<= self.charging_fault_detection_dict['mean_absolute_error_threshold']) and
                       (mse_e<= self.charging_fault_detection_dict['mean_squared_error_threshold']) and
                       (r2_e>=self.charging_fault_detection_dict['r2_threshold']))
            
        #Post Processing - remove labels caused by data anomaly - i.e. long periods of missing ROC_VOLTAGE data
        return df.apply(lambda x: get_charging_fault_label_(x),axis=1,raw=False)
    
    def evaluate_labeller(self, prediction_df:pd.Series, verbose:bool=False, positive_labels:Sequence[int]=[2,5])->dict:
        """Get confusion matrix evaluation for a given prediction dataframe

        Args:
            prediction_df (pd.Series): prediction dataframe 
            verbose (bool, optional): whether to get the date indices rather than the metric results. Defaults to False.

        Raises:
            ValueError: if self.data['label_df'] is None. Either labelled data is not available or data fetching encountered an error

        Returns:
            dict: confusion matrix metric evaluation 
        """
        if self.data['label_df'] is None:
            raise ValueError("No ground truth provided for comparison")
        gt = self.data['label_df']
        gt_pos_index = set(gt[gt['labels'].isin(positive_labels)].index)
        gt_neg_index = set(prediction_df.index) - gt_pos_index
        pred_pos_index = set(prediction_df[prediction_df==1].index)
        pred_neg_index = set(prediction_df[prediction_df!=1].index)
        TP_index = gt_pos_index & pred_pos_index
        FP_index = pred_pos_index - gt_pos_index
        FN_index = gt_pos_index - pred_pos_index
        TN_index = gt_neg_index & pred_neg_index
        TP, FP, TN, FN = len(TP_index), len(FP_index), len(TN_index), len(FN_index)
        if verbose: 
            return {"true_pos": TP_index, "true_neg": TN_index, "false_pos": FP_index, "false_neg": FN_index}
        return {"precision": TP/(TP+FP),
                "recall": TP/(TP+FN),
                "accuracy": (TP+TN)/(TP+TN+FP+FN),
                "true_pos": TP,
                "false_pos": FP,
                "true_neg": TN,
                "false_neg": FN}
        
    def get_weather_label(self)->pd.Series:
        pass 
    
    