from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd
from roc_model import (Classification_Model, Feature_Extractor_Model,
                       Regression_Model)


class roc:
    LABEL_MAPPING = {0: "Normal", 1:"Battery Capacity", 2:"SIDLV - Battery Capacity",
                     3: "Battery Degrading", 4: "Battery Fault", 5:"SIDLV Battery Fault",
                     6: "Charging Fault", 7: "Battery Recovering", 8:"Data Anomaly", 9:"Cloud Cover"}
    
    FAILURE_DESCRIPTION = {0:"",
                           1:"A Cell has died causing battery to deteriorate. Battery replacement required within the next month.",
                           2:"SIDLV",
                           3:"A Cell has died causing battery to deteriorate. Battery replacement required within the next month.",
                           4:"A Cell has died causing battery to deteriorate. Battery replacement required within the next month.",
                           5:"SIDLV",
                           6:"Battery no longer charging. Repair charging circuit, check fuse and fuse holder.",
                           7:"",
                           8:"Data Outage",
                           9:""}
    # TODO: Modify logics to account for days to load-off and label category 
    SEVERITY_LEVEL = {0:0, 1:2, 2:5, 3:2, 4:4, 5:5, 6:5, 7:0, 8:1, 9:0}
    
    SEVERITY_DESRIPTION = {0: "Normal", 1:"Notice", 2:"Low", 3:"Medium", 4:"High", 5:"Immediate Actions Required"}
    
    def __init__(self, feature_extractor:Feature_Extractor_Model, classification_model:Classification_Model, regression_model:Regression_Model):
        self.classification_model = classification_model
        self.regression_model = regression_model
        self.feature_extractor = feature_extractor
    
    @classmethod 
    def get_failure_label(cls, classification_model: Classification_Model, inference_data:pd.DataFrame) -> int:
        failure_label = classification_model.predict(inference_data)
        return failure_label 

    @classmethod 
    def get_well_features(cls, feature_extractor:Feature_Extractor_Model, inference_data:pd.DataFrame) -> tuple:
        well_status = feature_extractor.get_well_status(inference_data)
        downtime = feature_extractor.get_downtime(inference_data)
        volt_max = feature_extractor.get_max_volt(inference_data)
        volt_min = feature_extractor.get_min_volt(inference_data)
        charge_volt = feature_extractor.get_charge_volt(inference_data)
        return {"well_status": well_status, "downtime":downtime ,
                "volt_max": volt_max, "volt_min": volt_min, "charge_volt":charge_volt}
        
    @classmethod 
    def get_days_to_load_off(cls, regression_model:Regression_Model, inference_data:pd.DataFrame) -> int:
        return regression_model.predict(inference_data)

    @classmethod 
    def get_failure_category(cls, failure_label:int|Sequence[int]) -> str:
        if hasattr(failure_label,"__len__"):
            return [cls.LABEL_MAPPING[f] for f in failure_label]
        return cls.LABEL_MAPPING[failure_label]
    
    @classmethod 
    def get_failure_description(cls, failure_label:int|Sequence[int]) -> str:
        if hasattr(failure_label,"__len__"):
            return [cls.FAILURE_DESCRIPTION[f] for f in failure_label]
        return cls.FAILURE_DESCRIPTION[failure_label]
    
    @classmethod
    def get_severity_level(cls, failure_label:int|Sequence[int]) -> str:
        if hasattr(failure_label,"__len__"):
            return [cls.SEVERITY_LEVEL[f] for f in failure_label]
        return cls.SEVERITY_LEVEL[failure_label]
    
    @classmethod
    def get_severity_category(cls, severity_level:int|Sequence[int]) -> str:
        if hasattr(severity_level,"__len__"):
            return [cls.SEVERITY_DESRIPTION[f] for f in severity_level]
        return cls.SEVERITY_DESRIPTION[severity_level]
    
    #TODO: verify logic for cell status 
    @classmethod 
    def get_dead_cell_status(cls, failure_label:int|Sequence[int])->bool:
        if hasattr(failure_label,"__len__"):
            return [f in [1,2,4,5] for f in failure_label]
        return failure_label in [1,2,4,5]
    
    @classmethod 
    def get_sensor_fault_status(cls, failure_label:int|Sequence[int])->bool:
        if hasattr(failure_label, '__len__'):
            return [f == 8 for f in failure_label]
        return failure_label == 8
    
    #TODO: add logic for notification flag
    @classmethod 
    def get_notification_flag(cls, failure_label:int|Sequence[int])->bool:
        if hasattr(failure_label,"__len__"):
            return [True for _ in failure_label]
        return True 
    
    #TODO: add logic for production loss in dollars value
    @classmethod 
    def get_production_loss(cls, downtime:float|Sequence[float])->float:
        if hasattr(downtime,"__len__"):
            return [0.0 for _ in downtime]
        return 0.0
    
    @classmethod 
    def get_current_outage_status(cls, failure_label:int|Sequence[float])->bool:
        if hasattr(failure_label,"__len__"):
            return [f in [2,5] for f in failure_label]
        return failure_label in [2,5]
    
    @classmethod 
    def get_no_charge_status(cls, failure_label:int|Sequence[int])->bool:
        if hasattr(failure_label, "__len__"):
            return [f == 6 for f in failure_label]
        return failure_label == 6
    
    #TODO Fix get_insufficient_charge_status
    @classmethod 
    def get_insufficient_charge_status(cls, failure_label:int|Sequence[int])->bool:
        if hasattr(failure_label, "__len__"):
            return [False for _ in failure_label]
        return False 
    
    #TODO Fix get_high_voltage_status
    @classmethod 
    def get_high_voltage_status(cls, max_voltage:float|Sequence[float])->bool:
        if hasattr(max_voltage,"__len__"):
            return [False for _ in max_voltage]
        return False 
    
    #TODO Fix get_voltage_caused_outage_status
    @classmethod 
    def get_voltage_caused_outage_status(cls, failure_label:int|Sequence[int])->bool:
        if hasattr(failure_label,"__len__"):
            return [False for f in failure_label]
        return False 
    
    def get_all_notifications(self, inference_data: pd.DataFrame)->pd.DataFrame:
        """Generate a csv with all the notifications for the current day assessment 

        Args:
            inference_data (Sequence[int]): processed data to feed into inference models 

        Returns:
            pd.DataFrame: ROC notification csv
        """
        failure_labels = self.get_failure_label(self.classification_model, inference_data)
        days_to_load_off = self.get_days_to_load_off(self.regression_model, inference_data)
        feature_dict = self.get_well_features(self.feature_extractor, inference_data)
        trend_date = date.today()
        well_cd = inference_data.WELL_CD
        well_status = feature_dict['well_status']
        failure_category = self.get_failure_category(failure_label=failure_labels)
        failure_description = self.get_failure_description(failure_label=failure_labels)
        severity_level = self.get_severity_level(failure_label=failure_labels)
        severity_category = self.get_severity_category(severity_level=severity_level)
        volt_max = feature_dict['volt_max']
        volt_min = feature_dict['volt_min']
        charge_volt = feature_dict['charge_volt']
        no_charge = self.get_no_charge_status(failure_label=failure_labels)
        insufficient_charge = self.get_insufficient_charge_status(failure_label=failure_labels)
        high_voltage = self.get_high_voltage_status(max_voltage=volt_max)
        voltage_caused_outage = self.get_voltage_caused_outage_status(failure_label=failure_labels)
        current_outage = self.get_current_outage_status(failure_label=failure_labels)
        downtime = feature_dict['downtime']
        production_loss = self.get_production_loss(downtime)
        notification_flag = self.get_notification_flag(failure_label=failure_labels)
        sensor_fault = self.get_sensor_fault_status(failure_label=failure_labels)
        dead_cell = self.get_dead_cell_status(failure_label=failure_labels)
        return pd.DataFrame.from_dict({"TREND_DATA":trend_date,
                             "WELL_CD":well_cd,
                             "WELL_STATUS":well_status,
                             "FAILURE_CATEGORY":failure_category,
                             "FAILURE_DESCRIPTION":failure_description,
                             "SEVERITY_LEVEL":severity_level,
                             "SEVERITY_CATEGORY":severity_category,
                             "VOLT_MAX":volt_max,
                             "VOLT_MIN":volt_min,
                             "CHARGE_VOLTS":charge_volt,
                             "NO_CHARGE":no_charge,
                             "INSUFFICIENT_CHARGE":insufficient_charge,
                             "HIGH_VOLTAGE":high_voltage,
                             "VOLTAGE_CAUSED_OUTAGE":voltage_caused_outage,
                             "CURRENT_OUTAGE":current_outage,
                             "DAYS_TO_LOAD_OFF":days_to_load_off,
                             "DOWNTIME_PERCENT":downtime,
                             "PRODUCTION_LOSS":production_loss,
                             "NOTIFICATION_FLAG":notification_flag,
                             "SENSOR_FAULT":sensor_fault,
                             "DEAD_CELL":dead_cell
                             })
        
#Test with dummy data
if __name__ == "__main__":
    wells = ["ACRUS1","POND1","POND14"]
    x = np.linspace(0, 2*np.pi, 1440)
    sin_wave = np.sin(x)
    v_data = sin_wave + np.random.normal(0,5,size=(len(wells),1440))
    v_data = [list(v_data[i,:]) for i in range(len(wells)) ]
    f_data = 10 + np.random.normal(0,5,size=(len(wells),1440))
    f_data = [list(f_data[i,:]) for i in range(len(wells))]
    inference_data = pd.DataFrame.from_dict({"WELL_CD":wells, "ROC_VOLTAGE":v_data, "FLOW": f_data})
    from roc_model import Dummy_Classification_Model, Dummy_Regression_Model
    classification_model = Dummy_Classification_Model()
    regression_model = Dummy_Regression_Model()
    feature_extractor_model = Feature_Extractor_Model()
    notification_system = roc(feature_extractor_model, classification_model, regression_model)
    df = notification_system.get_all_notifications(inference_data)
    print("End")