import datetime 
from typing import Sequence
import logging 

import numpy as np
import pandas as pd
from Model.FeatureExtractor import FeatureExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelManager:
    INFERENCE_STATUS_CODE = {"0": "Inference completed successfully",
                             "1": "RuntimeExceptions encountered",
                             "2": "Insufficient data"}   
    
    FLOW_STATUS = {0: "Shut-In", 1: "Online"} 
    
    def __init__(self, 
                 feature_extractor:FeatureExtractor):
        self.feature_extractor = feature_extractor
        
    def get_trend_date(self):
        return self.feature_extractor.agg_df.index.max()
        
    def get_failure_info(self)->dict:
        response = {"TREND_DATE":None,
                    "WELL_CD":None,
                    "WELL_STATUS":None,
                    "FAILURE_CATEGORY":"Normal", 
                    "FAILURE_DESCRIPTION":"",
                    "SEVERITY_LEVEL":0,
                    "SEVERITY_CATEGORY":"Normal",
                    "VOLT_MAX":-1,
                    "VOLT_MIN":-1,
                    "CHARGE_VOLTS":-1,
                    "NO_CHARGE":"F",
                    "INSUFFICIENT_CHARGE":"F",
                    "HIGH_VOLAGE": "F", #TODO
                    "VOLAGE_CAUSED_OUTAGE": "F", #TODO
                    "CURENT_OUTAGE": "F", #TODO 
                    "DAYS_TO_LOAD_OFF": 30,
                    "DOWNTIME_PERCENT": "", #TODO 
                    "PRODUCTION_LOSS":"", #TODO 
                    "NOTIFICATION_FLAG":"F",                    
                    "SENSOR_FAULT":"F",
                    "DEAD_CELL":"F"}
        status = 0
        trend_date = self.get_trend_date()
        try: 
            response["TREND_DATE"] = trend_date 
            response["WELL_STATUS"] = self.FLOW_STATUS[self.feature_extractor.get_well_status()]
            response["VOLTAGE_MAX"] = self.feature_extractor.max_VOLTAGE.loc[trend_date]
            response["VOLAGE_MIN"] = self.feature_extractor.min_VOLTAGE.loc[trend_date]
            response["CHARGE_VOLTS"] = self.feature_extractor.charge_VOLTAGE.loc[trend_date]
            response["DAYS_TO_LOAD_OFF"] = self.feature_extractor.get_days_to_load_off[trend_date]
            
            #Handle Failure Category and Failure Description
            anomaly_label = self.feature_extractor.anomaly_label.loc[trend_date]
            failure_label = self.feature_extractor.get_failure_label().loc[trend_date]
            charging_fault_label = self.feature_extractor.get_charging_fault_label().loc[trend_date]
            
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
        except:
            status = 1
        return status, response 
        

#Quick Test with dummy data
if __name__ == "__main__":
    wells = ["ACRUS1","POND1","POND14"]
    x = np.linspace(0, 2*np.pi, 1440)
    sin_wave = np.sin(x)
    v_data = sin_wave + np.random.normal(0,5,size=(len(wells),1440))
    v_data = [list(v_data[i,:]) for i in range(len(wells)) ]
    f_data = 10 + np.random.normal(0,5,size=(len(wells),1440))
    f_data = [list(f_data[i,:]) for i in range(len(wells))]
    inference_data = pd.DataFrame.from_dict({"WELL_CD":wells, "ROC_VOLTAGE":v_data, "FLOW": f_data})
    inference_data = {wells[i]: inference_data.loc[[i],:] for i in range(3)}
    
    from Model.testing import Dummy_Classification_Model, Dummy_Regression_Model, Dummy_Feature_Extractor_Model
    classification_model = Dummy_Classification_Model()
    regression_model = Dummy_Regression_Model()
    feature_extractor = Dummy_Feature_Extractor_Model()
    notification_system = ModelManager(feature_extractor, classification_model, regression_model)
    df = notification_system.get_all_notifications(inference_data['ACRUS1'])
    output_dict = notification_system.get_inference_output(inference_data_dict=inference_data)
    print("End")