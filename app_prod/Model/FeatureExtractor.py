import numpy as np
import pandas as pd
from typing import Sequence
from itertools import groupby
from utils.ModelUtils import FolderUtils, PlotUtils
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from functools import cached_property
from scipy.signal import savgol_filter 

def upper_IQR(x):
    return np.nanquantile(x, 0.75)  

def lower_IQR(x):
    return np.nanquantile(x, 0.25)
     
def get_largest_outage_interval(mask:pd.Series, start:float, end:float,min_length:int)->tuple:
    """Get (start,end) tuple specifying the longest consecutive zeros in a sequence

    Args:
        mask (pd.Series): mask sequence - binary 
        start (float): start of period
        end (float): end of period 

    Returns:
        tuple: (start, end) if valid. None if either there is no 0 in mask or if mask is None
    """
    mask = np.array(mask[start:end])
    index = 0
    zero_index = []
    zero_length = [] 
    
    #Get index and length of consecutive zero sequences
    for mask_val,mask_group in groupby(mask):
        new_length = len(list(mask_group))
        if mask_val == 0: 
            zero_index.append(index)
            zero_length.append(new_length)
        index+=new_length
    zero_index = np.array(zero_index)
    zero_length = np.array(zero_length)

    #Filter out short outages
    zero_index = zero_index[zero_length >= min_length]
    zero_length = zero_length[zero_length >= min_length]

    #Handle edege cases
    if len(zero_length) == 0: #No outage
        return None 

    #Get longest zero sequence
    index = np.argmax(zero_length)
    start_index = zero_index[index] + start
    end_index = start_index + zero_length[index]
    return start_index, end_index

def verify_intersection(intervals:list[tuple])->int:
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

def nan_minimum(y:list)->float:
    y = np.array(y)
    y = y[y!=0]
    if len(y) <= 10:
        return np.nan
    return y.min()

def get_minimum_feature(agg_df:pd.DataFrame, feature:str)->pd.DataFrame:
    return agg_df[feature].apply(lambda x: nan_minimum(x))
    
def get_maximum_feature(agg_df:pd.DataFrame, feature:str)->pd.DataFrame:
    return agg_df[feature].apply(lambda x: max(x))
    
def get_integral_feature(agg_df:pd.DataFrame, feature:str) ->pd.Series:
    return agg_df[feature].apply(lambda x: np.trapz(x))
    
def get_relative_feature(agg_df:pd.DataFrame, feature:str, window:int=30, method:str="max")->pd.Series:
    method_dict = {"max":np.nanmax, "min":np.nanmin, "median":np.nanmedian, "mean":np.nanmean, "upper_IQR":upper_IQR, "lower_IQR":lower_IQR}
    rolling_window = get_rolling_window(agg_df, feature, window)
    reference = rolling_window.apply(method_dict[method])
    return agg_df[feature]/reference

def get_rolling_window(agg_df:pd.DataFrame, feature:str, window:int=30):
    return agg_df[feature].rolling(window=window)

def get_interpolated_feature(agg_df:pd.DataFrame, feature:str, method:str='linear')->pd.Series:
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

def get_outage_length(x:pd.Series)->pd.Series:
    zero_length = [len(list(g)) for k,g in groupby(x) if k==0] 
    if len(zero_length)==0:
        return 0
    return max(zero_length)

def get_gradient_feature(agg_df:pd.DataFrame, feature:str)->pd.Series:
    return agg_df[feature].apply(lambda x: np.gradient(x))

def get_linear_model_metrics(data:pd.Series)->tuple:
    Y = np.array(data['ROC_VOLTAGE'])
    X = np.arange(len(Y))
    mask = np.array(data['Mask_ROC_VOLTAGE'])
    Y_ = Y[mask==1].reshape(-1,1)
    X_ = X[mask==1].reshape(-1,1)
    if len(Y_)<=0.10*len(Y):
        return {"max_error": None, "mae": None, "mse": None}
    reg = LinearRegression().fit(X_,Y_)
    Y_pred = reg.predict(X_)
    max_e = max_error(Y_,Y_pred)
    mae_e = mean_absolute_error(Y_,Y_pred)
    mse_e = mean_squared_error(Y_,Y_pred)
    return {"max_error": max_e, "mae":mae_e, "mse":mse_e}
    
def get_linear_model_night_metrics(data:pd.Series)->tuple:  
    Y = np.array(data['ROC_VOLTAGE'][1000:])
    X = np.arange(len(Y))
    mask = np.array(data['Mask_ROC_VOLTAGE'][1000:])
    Y_ = Y[mask==1].reshape(-1,1)
    X_ = X[mask==1].reshape(-1,1)
    if len(Y_)<=0.10*len(Y):
        return {"max_error": None, "mae": None, "mse": None}
    reg = LinearRegression().fit(X_,Y_)
    Y_pred = reg.predict(X_)
    max_e = max_error(Y_,Y_pred)
    mae_e = mean_absolute_error(Y_,Y_pred)
    mse_e = mean_squared_error(Y_,Y_pred)
    return {"max_error": max_e, "mae":mae_e, "mse":mse_e}

def get_max_feature_dawn(data:pd.Series)->tuple:
    data = np.array(data)[1000:]
    return data.max()

class RWLS:
    def __init__(self, threshold:float, k:float, w:float):
        """Running Weighted Least Square regressor 
        Fit weighted linear model regression: 
            y_pred = np.exp(k*x)*a + b
        Loss function:
            l(a,b) = w||y_actual - y_pred||**2
        
        where 
            y_pred is the predicted minimum Voltage 
            y_actual is the actual minimum Voltage 
            x is the sequence index 
            k is scalling hyperparameter 
            (a,b) are the linear model parameter
            w - sample weight. Higher weights are placed on low Voltage samples
            
        Args:
            threshold (float): threshold for failure - i.e. 11/22V
            k (float): scaling hyper parameter
            w (float): weight of low value samples 
        
        Output:
            d - number of days until the predicted voltage drops below hreshold 
        """
        self.threshold = threshold 
        self.k = k
        self.w = w
        
    def __call__(self,y) -> int:
        y= np.array(y)
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        x = x[mask].reshape(-1,1)
        y = y[mask].reshape(-1,1)
        weights = np.ones(len(y))
        mask_low = np.where(y<=np.mean(y))[0]
        weights[mask_low]=self.w

        reg = LinearRegression().fit(np.exp(self.k*x),y, weights)
        x_pred = np.arange(30+len(y)).reshape(-1,1)
        y_pred = reg.predict(np.exp(self.k*x_pred))
        index = np.where(y_pred <= self.threshold)[0]
        if len(index)==0:
            return 30 
        return np.max(x_pred[index[0]] - len(y),0)
        


class FeatureExtractor:
    def __init__(self, 
                 agg_df:pd.DataFrame,
                 operation_correction_dict:dict={"gradient_threshold":0.025,"start":600,"end":1200},
                 anomaly_detection_dict:dict={"missing_length":360},
                 data_outage_detection_dict:dict={"start":1100, "end":1440, "min_length":60, "missing_length":20},
                 dawn_VOLTAGE_drop_detection_dict:dict={"first_derivative_threshold":0.4, "second_derivative_threshold":0.4,"use_second_derivative":True},
                 charging_fault_detection_dict:dict={"max_error_threshold":0.08, "mean_absolute_error_threshold":0.02,"mean_squared_error_threshold":0.008,\
                                                     "start":1000, "gradient_threshold":0.003},
                 weather_detection_dict:dict={"end":400,"quantile":0.85,"weather_threshold":0.96,"window":30},
                 )->None:
        self._verify_operation_correction_dict(operation_correction_dict)
        self._verify_anomaly_detection_dict(anomaly_detection_dict)
        self._verify_data_outage_detection_dict(data_outage_detection_dict)
        self._verify_dawn_VOLTAGE_drop_detection_dict(dawn_VOLTAGE_drop_detection_dict)
        self._verify_charging_fault_detection_dict(charging_fault_detection_dict)
        self._verify_weather_detection_dict(weather_detection_dict)
        self.agg_df = agg_df
    
    @cached_property
    def max_outage_length(self)->pd.Series:
        mask_VOLTAGE = self.agg_df['Mask_ROC_VOLTAGE']
        return mask_VOLTAGE.apply(lambda x: get_outage_length(x))
           
    @cached_property
    def interpolated_VOLTAGE(self)->pd.DataFrame:
        return get_interpolated_feature(self.agg_df,"ROC_VOLTAGE").to_frame()
    
    @cached_property
    def integral_VOLTAGE_morning(self)->pd.Series:
        result = self.interpolated_VOLTAGE.apply(lambda x: np.trapz(x['ROC_VOLTAGE'][:400]),axis=1)
        result.name = "ROC_VOLTAGE"
        return result
    
    @cached_property
    def integral_VOLTAGE_ratio_morning(self)->pd.Series:
        result = get_relative_feature(self.integral_VOLTAGE_morning.to_frame(), "ROC_VOLTAGE",30,"upper_IQR")
        result.name = "ROC_VOLTAGE"
        return result
    
    @cached_property
    def first_derivative(self)->pd.Series:
        return get_gradient_feature(self.interpolated_VOLTAGE, "ROC_VOLTAGE")
    
    @cached_property
    def second_derivative(self)->pd.Series: 
        return get_gradient_feature(self.first_derivative.to_frame(), "ROC_VOLTAGE")
    
    @cached_property
    def lm_VOLTAGE_metrics(self)->pd.DataFrame:
        df=self.agg_df.apply(lambda x: get_linear_model_metrics(x), axis=1,raw=False)
        return df.apply(pd.Series)
    
    @cached_property
    def lm_VOLTAGE_night_metrics(self)->pd.DataFrame:
        df=self.agg_df.apply(lambda x: get_linear_model_night_metrics(x), axis=1,raw=False)
        return df.apply(pd.Series)
    
    @cached_property
    def max_gradient_dawn(self)->pd.Series:
        """Max value of the gradient array sliced before dawn period - define around index 1000 in 1440 array 

        Returns:
            pd.Series: max gradient series 
        """
        return self.first_derivative.apply(lambda x: get_max_feature_dawn(x))
        
    @cached_property
    def anomaly_label(self)->pd.Series:
        """Anomaly label for days where the longest outage is more than 6 hours (or other defined threshold)

        Returns:
            pd.Series: anomaly label series 
        """
        return self.get_data_anomaly_label()
    
    @cached_property
    def min_VOLTAGE(self)->pd.Series:
        """Series of non-zero minimum VOLTAGE

        Returns:
            pd.Series: min VOLTAGE series 
        """
        return get_minimum_feature(self.interpolated_VOLTAGE,"ROC_VOLTAGE")
    
    @cached_property
    def max_VOLTAGE(self)->pd.Series:
        """Maximum daily VOLTAGE

        Returns:
            pd.Series: series of maximum daily VOLTAGE
        """
        return get_maximum_feature(self.interpolated_VOLTAGE, "ROC_VOLTAGE")
    
    @cached_property
    def charge_VOLTAGE(self)->pd.Series:
        return self.max_VOLTAGE-self.min_VOLTAGE
    
    def get_downtime(self)->pd.Series:
        """Downtime defined as the % of period when the well is non-flowing

        Returns:
            pd.Series: downtime in %
        """
        flow_data = self.agg_df["FLOW"]
        return flow_data.apply(lambda x: 1-len([i for i in x if i != 0])/len(x))
    
    def get_well_status(self)->pd.Series:
        """Well flow status. 0 - no flow, 1 - flow

        Returns:
            pd.Series: flow status 
        """
        downtime_label = self.get_downtime()
        return downtime_label.apply(lambda x: int(x>0))
    
    def get_days_to_load_off(self)->pd.Series:
        threshold = 11 if self.min_VOLTAGE.mean() <= 19 else 22 
        regression = RWLS(threshold, 0.25, 2)
        return regression(self.min_VOLTAGE)
    
    def _verify_operation_correction_dict(self, data_dict:dict)->None:
        gradient_threshold = 0.025 if 'gradient_threshold' not in data_dict else data_dict['gradient_threshold']
        start = 600 if 'start' not in data_dict else data_dict['start']
        end = 1200 if 'end' not in data_dict else data_dict['end']
        self.operation_correction_dict={"gradient_threshold":gradient_threshold, "start":start, "end":end}
        
    def _verify_data_outage_detection_dict(self, data_dict:dict)->None:
        min_length = 60 if 'min_length' not in data_dict else data_dict['min_length']
        start = 1000 if 'start' not in data_dict else data_dict['start']
        end = 1200 if 'end' not in data_dict else data_dict['end']
        missing_length = 180 if "missing_length" not in data_dict else data_dict['missing_length']
        self.data_outage_detection_dict={"min_length":min_length, "start":start, "end":end, "missing_length":missing_length}
        
    def _verify_anomaly_detection_dict(self, data_dict:dict)->None:
        missing_length = 180 if "missing_length" not in data_dict else data_dict['missing_length']
        self.anomaly_detection_dict={"missing_length":missing_length}
    
    def _verify_dawn_VOLTAGE_drop_detection_dict(self, data_dict:dict)->None:
        first_derivative_threshold=0.4 if "first_derivative_threshold" not in data_dict else data_dict['first_derivative_threshold']
        second_derivative_threshold=0.4 if "second_derivative_threshold" not in data_dict else data_dict["second_derivative_threshold"] 
        use_second_derivate=True if "use_second_derivate" not in data_dict else data_dict["use_second_derivate"]
        self.dawn_VOLTAGE_drop_detection_dict={"first_derivative_threshold":first_derivative_threshold, "second_derivative_threshold":second_derivative_threshold, "use_second_derivative":use_second_derivate}
             
    def _verify_charging_fault_detection_dict(self, data_dict:dict)->None:
        max_error_threshold = 0.08 if "max_error_threshold" not in data_dict else data_dict["max_error_threshold"]
        mean_absolute_error_threshold= 0.02 if "mean_absolute_error_threshold" not in data_dict else data_dict["mean_absolute_error_threshold"]
        mean_squared_error_threshold=0.008 if "mean_squared_error_threshold" not in data_dict else data_dict["mean_squared_error_threshold"]
        gradient_threshold = 0.025 if 'gradient_threshold' not in data_dict else data_dict['gradient_threshold']
        start = 1000 if 'start' not in data_dict else data_dict['start']
        self.charging_fault_detection_dict = {"max_error_threshold":max_error_threshold,"mean_absolute_error_threshold":mean_absolute_error_threshold,
                                              "mean_squared_error_threshold":mean_squared_error_threshold,
                                              "start":start, "gradient_threshold":gradient_threshold}
    
    def _verify_weather_detection_dict(self, data_dict:dict)->None:
        end=400 if "end" not in data_dict else data_dict["end"]
        window=30 if "window" not in data_dict else data_dict["window"]
        quantile=0.85 if "quantile" not in data_dict else data_dict["quantile"]
        weather_threshold=0.98 if "weather_threshold" not in data_dict else data_dict["weather_threshold"]
        self.weather_detection_dict = {"end":end, "window":window, "quantile":quantile, "weather_threshold": weather_threshold}
    
    #TESTED   
    def get_data_anomaly_label(self)->pd.Series:
        """Assign data anomaly label based on rate of missing ROC_VOLTAGE data 
        
        Args - embedded in anomaly_detection_dict
            missing_length (int): threshold length of continuous missing data for it to be considered anomaly. Defaults to 180 or 3 hours.

        Returns:
            pd.Series: 0 - not abnormal, 1 - data anomaly 
        """
        anomaly_label =  self.max_outage_length >= self.anomaly_detection_dict["missing_length"]
        anomaly_label.name = "labels"
        return anomaly_label
    
    #TESTED
    def get_dawn_data_outage_feature(self)->pd.Series:
        outage_length = self.max_outage_length
        normal_days = outage_length[outage_length <= 0.8*1440].index
        FLOW_data = self.agg_df['Mask_FLOW'].loc[normal_days]
        PRESSURE_data = self.agg_df['Mask_PRESSURE_TH'].loc[normal_days] if 'Mask_PRESSURE_TH' in self.agg_df else None 

        temp_df = pd.DataFrame()

        temp_df['FLOW_interval'] = FLOW_data.apply(lambda x: get_largest_outage_interval(x,
                                                                self.data_outage_detection_dict['start'], 
                                                                self.data_outage_detection_dict['end'],
                                                                self.data_outage_detection_dict['missing_length']))
        if PRESSURE_data is not None:
            temp_df['PRESSURE_interval'] = PRESSURE_data.apply(lambda x: get_largest_outage_interval(x,
                                                                    self.data_outage_detection_dict['start'], 
                                                                    self.data_outage_detection_dict['end'],
                                                                    self.data_outage_detection_dict['missing_length']))
        else:
            temp_df['PRESSURE_interval'] = 0
        return temp_df

    #TESTED
    def get_dawn_data_outage_label(self):
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
            missing_length (int, optional): length of continuous missing ROC_VOLTAGE data for a day to be considered abnormal. Defaults to 180 mins or 3 hours

        Returns:
            pd.Series: label - 0: no failure, 1: has failure, 2: uncertain/data anomaly
        """
        #Remove days when data is completely missing: 
        temp_df = self.get_dawn_data_outage_feature()
        
        def label_outage(x:pd.Series)->pd.Series:
            if (x['FLOW_interval'] is None) or (x['PRESSURE_interval'] is None): #No missing -> normal data
                return 0 
            if isinstance(x["PRESSURE_interval"],int):
                return 2
            return verify_intersection([x['FLOW_interval'],x['PRESSURE_interval']])

        label = temp_df.apply(lambda x: label_outage(x), raw=False,axis=1)
        label.name = "labels"
        
        #Post processing for high voltage: 
        min_VOLTAGE = self.min_VOLTAGE.mean()
        if min_VOLTAGE < 19:
            cut_off = 12
        else: 
            cut_off = 22.5
        low_VOLTAGE_dates = self.min_VOLTAGE[self.min_VOLTAGE<=cut_off].index
        non_anommaly_dates = self.anomaly_label[self.anomaly_label==0].index
        label = label.loc[list(set(non_anommaly_dates)&set(low_VOLTAGE_dates))]
        return label
        
    #TESTED
    def get_dawn_VOLTAGE_drop_features(self)->pd.DataFrame:
        data = self.agg_df.ROC_VOLTAGE
        data = data[self.anomaly_label==0]
        def process_fn(x):
            x = np.array(x)[600:]
            x = x[x!=0]
            filter_data = savgol_filter(savgol_filter(x, window_length=10, polyorder=2),window_length=10,polyorder=1)
            first_derivative = np.gradient(filter_data)
            second_derivative = np.gradient(first_derivative)
            onset_location = np.where((second_derivative>=0.002) & (first_derivative>=0.005))[0]
            if len(onset_location)==0:
                onset_location=len(first_derivative)
            else:
                onset_location=onset_location[0]
            
            return {"filtered_data": filter_data,
                    "first":first_derivative,
                    "filtered_data_onset":filter_data[:onset_location], 
                    "first_onset":first_derivative[:onset_location]}
        all_data = data.apply(lambda x: process_fn(x)).apply(pd.Series)
        return all_data
    
    #TESTED
    def get_dawn_VOLTAGE_drop_label(self)->pd.Series: 
        """Check for presence of battery failure that cause a sharp drop in ROC_VOLTAGE before dawn. 
        
        This method computes the first and second derivatives and assigns labels by thresholding. This method works on interpolated data. 
        
        Args: embedded in dawn_VOLTAGE_drop_detection_dict
            first_derivative_threshold (float, optional): threshold of the first derivative. Defaults to 0.4.
            second_derivative_threshold (float, optional): threshold of the second derivative. Defaults to 0.4.
            use_second_derivate (bool, optional): whether to use 2nd derivative to derive label. Defaults to True.

        Returns:
            pd.Series: failure label. 0 - no failure, 1 - failure present 
        """
        min_VOLTAGE = self.min_VOLTAGE.mean()
        if min_VOLTAGE < 19:
            cut_off = 11.8
        else: 
            cut_off = 22.5
        low_VOLTAGE_dates = self.min_VOLTAGE[self.min_VOLTAGE<=cut_off].index

        all_data = self.get_dawn_VOLTAGE_drop_features()
        label = all_data["first_onset"].apply(lambda x: x[-20:].min()<-0.05)
        label = label.loc[list(set(label.index)&set(low_VOLTAGE_dates))]
        label.name = "labels"
        return label

    #TODO
    def get_min_VOLTAGE_label(self)->pd.Series:
        mean_min_V = self.min_VOLTAGE.mean()
        if mean_min_V < 19:
            cut_off = 11.0
        else:
            cut_off = 22
        label = (self.min_VOLTAGE <= cut_off).astype(int)
        label.name = "labels"
        return label
    
    #TESTED
    def get_charging_fault_label(self)->pd.Series:
        """Get charging fault label.
        
        Charging Fault results in a continuous decline of ROC_VOLTAGE that is almost monotonic. A linear regressor is fitted on the data, 
        with different fitness metrics compared against thresholds to provide label.

        Args: embedded in charging_fault_config_dict
            max_error_threshold (float): max error 
            mean_squared_error_threshold (float): mse threshold 
            mean_absolute_error_threshold (float): mae threshold

        Returns:
            pd.Series: 0 - no charging fault, 1 - charging fault, 2 - data anomaly 
        """
        lm_metric = self.lm_VOLTAGE_metrics[self.anomaly_label==0]
        lm_metric_night = self.lm_VOLTAGE_night_metrics[self.anomaly_label==0]
        gradient_dawn = self.max_gradient_dawn[self.anomaly_label==0]

        def compare_metrics(x:pd.Series)->pd.Series:
            if x['max_error'] is not None and x['mae'] is not None and x['mse'] is not None:
                return (x['max_error'] <= self.charging_fault_detection_dict['max_error_threshold']) and \
                       (x["mae"] <= self.charging_fault_detection_dict["mean_absolute_error_threshold"]) and \
                       (x["mse"] <= self.charging_fault_detection_dict['mean_squared_error_threshold'])
                       
        lm_label = lm_metric.apply(lambda x: compare_metrics(x), axis=1,raw=False)
        lm_label_night = lm_metric_night.apply(lambda x: compare_metrics(x), axis=1, raw=False)
        gradient_label = gradient_dawn.apply(lambda x: x<=self.charging_fault_detection_dict["gradient_threshold"])
        lm_label[(gradient_label==1)|(lm_label==1)|(lm_label_night==1)]=1
        lm_label.name = "labels"
        return lm_label
    
    #TESTED
    def get_weather_label(self)->pd.Series:
        """Get weather label.
        
        Achieved by comparing the ratio of the integral of the day period (specified by config kwd "end") over the benchmark 
        in a specified window length (controlled by config kwd "window") against a specific weather threshold.

        Args: embedded in weather_detection_dict
            end (int|Optional) - index of end of day. Defaults to 400
            window (int|Optional) - window length of review. Defaults to 30 days.
            quantile (float|Optional) - quantile at which the day in window is considered a benchmark. Defaults to 0.85
            weather_threshold (float|Optional) - weather threshold ratio
        Returns:
            pd.Series: weather label 
        """
        ratio = self.integral_VOLTAGE_ratio_morning[self.anomaly_label==0]
        label = (ratio<=self.weather_detection_dict["weather_threshold"]).astype(int)
        label.name = "labels"
        return label
    
    #TESTED
    def get_cloud_cover_ratio(self)->pd.Series:
        return self.integral_VOLTAGE_ratio_morning
    
    #TESTED
    def get_failure_label(self)->pd.Series:
        dawn_VOLTAGE_drop_label = self.get_dawn_VOLTAGE_drop_label()
        dawn_VOLTAGE_drop_label = dawn_VOLTAGE_drop_label[dawn_VOLTAGE_drop_label==1].index
        
        dawn_data_outage_label = self.get_dawn_data_outage_label()
        dawn_data_outage_label = dawn_data_outage_label[dawn_data_outage_label==1].index
        
        min_VOLTAGE_label = self.get_min_VOLTAGE_label()
        min_VOLTAGE_label = min_VOLTAGE_label[min_VOLTAGE_label==1].index
        
        charge_fault_label = self.get_charging_fault_label()
        charge_fault_label = charge_fault_label[charge_fault_label==1].index
        
        positive_label = sorted((set(dawn_data_outage_label)|set(dawn_VOLTAGE_drop_label)|set(min_VOLTAGE_label)) - set(charge_fault_label))
        negative_label = sorted(set(self.anomaly_label.index) - set(positive_label))
        all_labels = sorted(set(positive_label)|set(negative_label))
        label = pd.Series(index = all_labels, dtype=int)
        label.name = "labels"
        label.loc[positive_label]=1
        label.loc[negative_label]=0
        return label
           
class LabelEvaluator:
    def __init__(self, well_code:str, gt:pd.DataFrame, labeller:FeatureExtractor, raw_df: pd.DataFrame):
        self.well_code = well_code
        self.gt = gt
        self.raw_df = raw_df
        self.labeller = labeller
        
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
        if self.gt is None:
            raise ValueError("No ground truth provided for comparison")
        gt = self.gt
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
        try:
            return {"precision": TP/(TP+FP),
                    "recall": TP/(TP+FN),
                    "accuracy": (TP+TN)/(TP+TN+FP+FN),
                    "true_pos": TP,
                    "false_pos": FP,
                    "true_neg": TN,
                    "false_neg": FN}
        except:
            return{"true_pos":TP, "false_pos":FP, "true_neg":TN, "false_neg":FN}
        
    def get_diagnostic_plots(self, label_type)->None:
        label_map = {"weather":(self.labeller.get_weather_label,9),
                      "charging_fault":(self.labeller.get_charging_fault_label,6),
                      "dawn_VOLTAGE_drop":(self.labeller.get_dawn_VOLTAGE_drop_label,[2,5]),
                      "dawn_data_outage":(self.labeller.get_dawn_data_outage_label,[2,5]),
                      "min_VOLTAGE": (self.labeller.get_min_VOLTAGE_label,[2,5]),
                      "failure_label": (self.labeller.get_failure_label, [2,5]),
                      "data_anomaly":(self.labeller.get_data_anomaly_label,8)}
        if label_type not in label_map:
            raise KeyError(f"label_type must be one of {list(label_map.keys())}")
        folder_manager = FolderUtils(self.well_code, label_type)
        folder_manager.clear_path()
        preds = label_map[label_type][0]()
        target_label = label_map[label_type][1]
        label_dict = PlotUtils.get_label_dict(self.gt,preds.to_frame(),target_label)
        plot_windows = PlotUtils.get_plot_window(label_dict)
        for i in range(len(plot_windows)):
            plot_name, plot = PlotUtils.get_diagnostic_plots(self.raw_df, label_dict, plot_windows,i,self.well_code)
            plot_name = plot_name + f"_{label_type}.png"
            plot.savefig(folder_manager.parent_dir/plot_name)
            
        eval=self.evaluate_labeller(preds,verbose=True,positive_labels=target_label)
        FN= sorted(list(set(eval['false_neg']) - set(self.labeller.anomaly_label[self.labeller.anomaly_label==1].index)))
        anomaly = sorted(list(set(eval['false_neg']) & set(self.labeller.anomaly_label[self.labeller.anomaly_label==1].index)))
        FP= sorted(list(eval['false_pos']))
        metrics = {d.strftime("%Y-%m-%d"):"false negative" for d in FN}
        metrics.update({d.strftime("%Y-%m-%d"):"false positive" for d in FP})
        metrics.update({d.strftime("%Y-%m-%d"):"anomaly" for d in anomaly})
        df = pd.DataFrame({"Date":metrics.keys(), "Type": metrics.values()})
        df.to_csv(folder_manager.parent_dir/f"metrics_{self.well_code}.csv",index=False)
    