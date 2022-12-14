import datetime 
from typing import Sequence
import logging 

import numpy as np
import pandas as pd
from Model.FeatureExtractor import FeatureExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelManager:
    STATUS_CODE = {"0": "Inference completed successfully",
                             "1": "RuntimeExceptions encountered",
                             "2": "Insufficient data"}   
    
    LOCAL_TIMEZONE = str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)
    
    FLOW_STATUS = {0: "Shut-In", 1: "Online"} 
    def __init__(self, **kwargs):
        self.model_kwargs = kwargs 
    
    def run_inference(self, inference_data:tuple[dict[str,pd.DataFrame], dict[str,tuple]])->dict:
        logger.debug("Running inference for all wells.")
        all_responses = {}
        inference_df = inference_data[0]
        inference_metadata = inference_data[1]
        for well in inference_df:
            well_inference_df = inference_df[well]
            well_inference_metadata = inference_metadata[well]
            all_responses[well] = self.run_inference_(well_inference_df, well_inference_metadata)
        return all_responses
    
    def run_inference_(self, inference_df:pd.DataFrame, inference_metadata:pd.DataFrame)->dict:
        response = {"TREND_DATE":None,
                    "WELL_STATUS":None,
                    "FAILURE_CATEGORY":"Normal", 
                    "FAILURE_DESCRIPTION":"",
                    "SEVERITY_LEVEL":0,
                    "SEVERITY_CATEGORY":"Normal",
                    "VOLTAGE_MAX":-1,
                    "VOLTAGE_MIN":-1,
                    "CHARGE_VOLTS":-1,
                    "NO_CHARGE":"F",
                    "INSUFFICIENT_CHARGE":"F",
                    "HIGH_VOLTAGE": "F", #TODO
                    "VOLTAGE_CAUSED_OUTAGE": "F", #TODO
                    "CURENT_OUTAGE": "F", #TODO 
                    "DAYS_TO_LOAD_OFF": 30,
                    "DOWNTIME_PERCENT": "", #TODO 
                    "PRODUCTION_LOSS":"", #TODO 
                    "NOTIFICATION_FLAG":"F",                    
                    "SENSOR_FAULT":"F",
                    "DEAD_CELL":"F"}
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        if inference_df is None:
            return {"STATUS":2, 
                    "BODY": None, 
                    "START_TS":start_time, 
                    "END_TS":start_time, 
                    "FIRST_TS": inference_metadata[0].strftime("%Y-%m-%d %H:%M"),
                    "LAST_TS": inference_metadata[1].strftime("%Y-%m-%d %H:%M")} 
        status = 0
        exception_message = "Inference run successfully"
        feature_extractor = FeatureExtractor(inference_df, **self.model_kwargs)
        try:
            trend_date = feature_extractor.trend_date
            response["TREND_DATE"] =  trend_date.strftime("%Y-%m-%d %H:%M")
            response["WELL_STATUS"] = self.FLOW_STATUS[feature_extractor.get_well_status().loc[trend_date]]
            response["VOLTAGE_MAX"] = feature_extractor.max_VOLTAGE.loc[trend_date]
            response["VOLTAGE_MIN"] = feature_extractor.min_VOLTAGE.loc[trend_date]
            response["CHARGE_VOLTS"] = feature_extractor.charge_VOLTAGE.loc[trend_date]
            response["DAYS_TO_LOAD_OFF"] = feature_extractor.get_days_to_load_off()[trend_date]
            
            #Handle Failure Category and Failure Description
            anomaly_label = feature_extractor.anomaly_label.loc[trend_date]
            failure_label = feature_extractor.get_failure_label().loc[trend_date]
            charging_fault_label = feature_extractor.get_charging_fault_label().loc[trend_date]
            
            if anomaly_label: 
                response["FAILURE_DESCRIPTION"] = "Significant Data Outage. Rerun model inference on this data at another time."
                response["SEVERITY_LEVEL"] = 1
                response["SEVERITY_CATEGORY"] = "Notice"
                status = 2
            elif charging_fault_label: 
                response["FAILURE_CATEGORY"] = "Charging Fault"
                response["FAILURE_DESCRIPTION"] = "Battery no longer charging. Repair charging circuit, check fuse and fuse holder."
                response["SENSOR_FAULT"] = "T"
                response["NO_CHARGE"] = "T"
                response["INSUFFICIENT_CHARGE"] = "T"
            elif failure_label:
                response["FAILURE_CATEGORY"] = "Battery Fault"
                response["FAILURE_DESCRIPTION"] = "A Cell has died causing battery to deteriorate."
                response["DEAD_CELL"] = "T"  
            
            #Handle severity level, severity category, notification flag 
            if response["DAYS_TO_LOAD_OFF"] >= 14:
                response["SEVERITY_LEVEL"] = 1
                response["SEVERITY_CATEGORY"] = "Notice"
            elif response["DAYS_TO_LOAD_OFF"] >= 7:
                response["SEVERITY_LEVEL"] = 3
                response["SEVERITY_CATEGORY"] = "Medium"
            elif response["DAYS_TO_LOAD_OFF"] >=3:
                response["SEVERITY_LEVEL"] = 4
                response["SEVERITY_CATEGORY"] = "High"
            else:
                response["SEVERITY_LEVEL"] = 5
                response["SEVERITY_CATEGORY"]="Immediate actions required"
        except Exception as exception_message:
            logger.log(f"Errors encountered when running inference. Exception encountered: {exception_message}")            
            status = 1
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        return {"STATUS":status, "BODY": response, "START_TS": start_time, "END_TS": end_time, 
                "FIRST_TS": inference_metadata[0].strftime("%Y-%m-%d %H:%M"),
                "LAST_TS": inference_metadata[1].strftime("%Y-%m-%d %H:%M")} 
        