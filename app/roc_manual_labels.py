# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:31:29 2022

@author: Steve Lechowicz
"""
from datetime import datetime, timedelta
import src.aau.advancedanalytics_util as aau
import pandas as pd
from matplotlib import pyplot as plt
import yaml 
import src.roc._manual_labels_config as roc_manual_labels_config
import src.roc.state as roc_state
from src.utils.PathManager import Paths as Path 

def connect_(info, connection_type=None):
    if connection_type is None:
        connection_type = info['connection_type']
    # creates a connection to a data source / dest via the aau utility functions
    # connection type can be modified in configuration without changing code
    if connection_type == 's3':
        con = aau.S3(info)
    elif connection_type == 'p2':
        con = aau.P2(info)
    elif connection_type == 'file':
        con = aau.File(info)
    elif connection_type == 'rts':
        con = aau.RTS(info)
    elif connection_type == 'ora':
        con = aau.Oracle(info)
    elif connection_type == 'sql':
        con = aau.SQL(info)
    return con

def get_filestartend(state, ts):
    y = ts.year
    m = ts.month
    start = datetime(y, m, 1)
    m += 1
    if m > 12:
        m = 1
        y += 1
    end = datetime(y, m, 1)
    return start, end

def get_filename(state, start, end):
    well_cd = state['well_cd']
    fn = '{}_{}_{}_{}.{}'.format(well_cd, 
                                 config['procdata_file_prefix'], 
                                 start.strftime('%Y%m%d'), 
                                 end.strftime('%Y%m%d'), 
                                 config['procdata_file_ext'])
    return fn

# Initialise the config and state
config = roc_manual_labels_config.init()
state = roc_state.init()

###########################################################################
# CHECKED
# WELL CLASSES (NOTE wells may change between classes)
# 1=Normal (smooth voltage)
# 2=SPAC (regular frequent voltage dips due to high current device)
# 3=Cycling (irregular voltage steps correlated with well cycling state - flow and thp)
# 4=Variable (irregular voltage steps not correlated with flow and/or thp)
#
# VOLTAGE CLASSES
# 0=Normal
# 1=Battery Capacity
# 2=SIDLV - Battery Capacity
# 3=Battery Degrading
# 4=Battery Fault
# 5=SIDLV - Battery Fault
# 6=Charging Fault
# 7=Battery Recovering
# 8=Data Anomaly
# 9=Cloud Cover
state['well_classes'] = {}
state['preset_targets'] = {}
state['preset_targets']['ACRUS1'] = {}
dy =  datetime(2018, 1, 1)
while dy < datetime(2018, 2, 2):
    state['preset_targets']['ACRUS1'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['ACRUS1'][datetime(2018, 2, 3)] = 5 # really not sure about this one - looks like battery was replaced so shut in was caused by maintenance
state['preset_targets']['ACRUS1'][datetime(2018, 2, 14)] = 1
state['preset_targets']['ACRUS1'][datetime(2018, 2, 26)] = 3
state['preset_targets']['ACRUS1'][datetime(2018, 2, 27)] = 3
state['preset_targets']['ACRUS1'][datetime(2018, 3, 8)] = 1
state['preset_targets']['ACRUS1'][datetime(2018, 3, 9)] = 1
# CHECKED
state['well_classes']['BIGL22'] = [1]
state['preset_targets']['BIGL22'] = {}
dy =  datetime(2022, 1, 1)
while dy < datetime(2022, 7, 1):
    state['preset_targets']['BIGL22'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['BIGL22'][datetime(2022, 1, 17)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 1, 18)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 1, 19)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 1, 22)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 1, 23)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 1, 24)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 1, 25)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 1, 26)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 2, 1)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 2, 2)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 2, 3)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 2, 10)] = 3
state['preset_targets']['BIGL22'][datetime(2022, 2, 11)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 2, 12)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 2, 13)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 2, 14)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 3, 23)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 3, 24)] = 1
state['preset_targets']['BIGL22'][datetime(2022, 3, 25)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 4, 18)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 4, 19)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 4, 24)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 4, 25)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 4, 26)] = 2
state['preset_targets']['BIGL22'][datetime(2022, 4, 27)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 4, 28)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 4, 29)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 4, 30)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 5, 1)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 2)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 3)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 4)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 10)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 5, 11)] = 8
state['preset_targets']['BIGL22'][datetime(2022, 5, 12)] = 8
state['preset_targets']['BIGL22'][datetime(2022, 5, 13)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 5, 14)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 18)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 5, 19)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 5, 20)] = 1
state['preset_targets']['BIGL22'][datetime(2022, 5, 21)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 22)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 23)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 5, 28)] = 1
state['preset_targets']['BIGL22'][datetime(2022, 5, 29)] = 2
state['preset_targets']['BIGL22'][datetime(2022, 5, 30)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 5, 31)] = 8
state['preset_targets']['BIGL22'][datetime(2022, 6, 1)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 2)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 3)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 4)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 6, 5)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 6, 12)] = 2
state['preset_targets']['BIGL22'][datetime(2022, 6, 13)] = 8
state['preset_targets']['BIGL22'][datetime(2022, 6, 14)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 15)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 16)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 17)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 18)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 19)] = 8
state['preset_targets']['BIGL22'][datetime(2022, 6, 20)] = 5
state['preset_targets']['BIGL22'][datetime(2022, 6, 21)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 6, 25)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 6, 26)] = 7
state['preset_targets']['BIGL22'][datetime(2022, 6, 29)] = 9
state['preset_targets']['BIGL22'][datetime(2022, 6, 30)] = 7
# CHECKED
state['well_classes']['BIGL20'] = [1,3]
state['preset_targets']['BIGL20'] = {}
dy =  datetime(2022, 1, 1)
while dy < datetime(2022, 7, 1):
    state['preset_targets']['BIGL20'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['BIGL20'][datetime(2022, 1, 17)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 1, 18)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 1, 22)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 1, 23)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 1, 24)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 1, 25)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 1, 26)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 2, 1)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 2, 2)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 2, 3)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 3, 24)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 3, 25)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 4, 18)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 4, 19)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 4, 20)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 4, 24)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 4, 25)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 4, 26)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 4, 27)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 4, 30)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 5, 10)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 5, 11)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 5, 12)] = 7
dy =  datetime(2022, 5, 15)
while dy < datetime(2022, 6, 9):
    state['preset_targets']['BIGL20'][dy] = 8
    dy = dy + timedelta(days=1)
state['preset_targets']['BIGL20'][datetime(2022, 6, 19)] = 9
state['preset_targets']['BIGL20'][datetime(2022, 6, 20)] = 7
state['preset_targets']['BIGL20'][datetime(2022, 6, 29)] = 1
state['preset_targets']['BIGL20'][datetime(2022, 6, 30)] = 7
# CHECKED
state['preset_targets']['MOOM115'] = {}
dy =  datetime(2022, 1, 1)
while dy < datetime(2022, 7, 10):
    state['preset_targets']['MOOM115'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['MOOM115'][datetime(2022, 1, 17)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 1, 18)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 1, 22)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 1, 23)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 1, 24)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 1, 25)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 1, 30)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 1, 31)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 2, 1)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 2, 2)] = 2
state['preset_targets']['MOOM115'][datetime(2022, 2, 3)] = 4
state['preset_targets']['MOOM115'][datetime(2022, 2, 4)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 2, 5)] = 7
dy =  datetime(2022, 2, 8)
while dy < datetime(2022, 3, 6):
    state['preset_targets']['MOOM115'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['MOOM115'][datetime(2022, 2, 18)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 3, 6)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 3, 7)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 3, 24)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 3, 25)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 4, 18)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 4, 24)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 4, 25)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 4, 26)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 4, 27)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 4, 28)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 4, 30)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 5, 1)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 5, 10)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 5, 11)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 5, 12)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 5, 18)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 5, 19)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 5, 20)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 5, 29)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 5, 30)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 6, 1)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 6, 2)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 6, 3)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 6, 4)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 6, 5)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 6, 6)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 6, 12)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 6, 13)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 6, 19)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 6, 20)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 6, 29)] = 9
state['preset_targets']['MOOM115'][datetime(2022, 6, 30)] = 7
state['preset_targets']['MOOM115'][datetime(2022, 7, 1)] = 9
# CHECKED - NOTE THAT THIS WELL HAS AN INVALID CALIBRATION (VOLTAGE RANGE IS 17-18.5)
state['well_classes']['MOOM157'] = [3,4]
state['preset_targets']['MOOM157'] = {}
dy =  datetime(2022, 1, 1)
while dy < datetime(2022, 7, 11):
    state['preset_targets']['MOOM157'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['MOOM157'][datetime(2022, 1, 17)] = 1
state['preset_targets']['MOOM157'][datetime(2022, 1, 18)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 1, 19)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 1, 21)] = 3
state['preset_targets']['MOOM157'][datetime(2022, 1, 22)] = 1
state['preset_targets']['MOOM157'][datetime(2022, 1, 23)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 1, 24)] = 1
state['preset_targets']['MOOM157'][datetime(2022, 1, 25)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 1, 26)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 1, 30)] = 1
state['preset_targets']['MOOM157'][datetime(2022, 1, 31)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 2, 1)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 2, 2)] = 1
state['preset_targets']['MOOM157'][datetime(2022, 2, 3)] = 7
dy =  datetime(2022, 2, 13)
while dy < datetime(2022, 5, 12):
    state['preset_targets']['MOOM157'][dy] = 8
    dy = dy + timedelta(days=1)
state['preset_targets']['MOOM157'][datetime(2022, 5, 19)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 5, 20)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 5, 29)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 5, 30)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 6, 1)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 6, 2)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 6, 3)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 6, 4)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 6, 19)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 6, 20)] = 7
state['preset_targets']['MOOM157'][datetime(2022, 6, 29)] = 9
state['preset_targets']['MOOM157'][datetime(2022, 6, 30)] = 7
# CHECKED
state['preset_targets']['POND1'] = {}
state['preset_targets']['POND1'][datetime(2018, 6, 25)] = 5 # or 2
dy =  datetime(2018, 6, 27)
while dy < datetime(2018, 7, 4):
    state['preset_targets']['POND1'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['POND1'][datetime(2018, 7, 4)] = 7
dy =  datetime(2018, 7, 13)
while dy < datetime(2018, 9, 5):
    state['preset_targets']['POND1'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['POND1'][datetime(2018, 9, 5)] = 7
# CHECKED
state['preset_targets']['POND11'] = {}
state['preset_targets']['POND11'][datetime(2018, 12, 14)] = 8
state['preset_targets']['POND11'][datetime(2019, 2, 11)] = 8
state['preset_targets']['POND11'][datetime(2019, 2, 12)] = 8
state['preset_targets']['POND11'][datetime(2019, 2, 14)] = 8
state['preset_targets']['POND11'][datetime(2019, 3, 13)] = 8
state['preset_targets']['POND11'][datetime(2019, 3, 14)] = 8
state['preset_targets']['POND11'][datetime(2019, 3, 15)] = 8
state['preset_targets']['POND11'][datetime(2019, 3, 25)] = 8
state['preset_targets']['POND11'][datetime(2019, 3, 27)] = 8
state['preset_targets']['POND11'][datetime(2019, 3, 29)] = 8
state['preset_targets']['POND11'][datetime(2020, 2, 9)] = 3
state['preset_targets']['POND11'][datetime(2020, 2, 10)] = 8
# CHECKED
state['preset_targets']['POND14'] = {} # invalid zero THP
state['preset_targets']['POND14'][datetime(2018, 4, 4)] = 3
state['preset_targets']['POND14'][datetime(2018, 4, 5)] = 5
state['preset_targets']['POND14'][datetime(2018, 4, 6)] = 7
state['preset_targets']['POND14'][datetime(2018, 4, 7)] = 3
state['preset_targets']['POND14'][datetime(2018, 5, 1)] = 3
state['preset_targets']['POND14'][datetime(2018, 5, 3)] = 2
state['preset_targets']['POND14'][datetime(2018, 6, 26)] = 2
state['preset_targets']['POND14'][datetime(2018, 6, 27)] = 7
state['preset_targets']['POND14'][datetime(2019, 3, 25)] = 2
state['preset_targets']['POND14'][datetime(2019, 3, 26)] = 7
state['preset_targets']['POND14'][datetime(2019, 3, 8)] = 3 # also anomalous low voltage values below 10V - double check this date
state['preset_targets']['POND14'][datetime(2019, 9, 21)] = 3
state['preset_targets']['POND14'][datetime(2019, 9, 24)] = 2
state['preset_targets']['POND14'][datetime(2019, 9, 25)] = 1
state['preset_targets']['POND14'][datetime(2020, 2, 3)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 5)] = 5
state['preset_targets']['POND14'][datetime(2020, 2, 6)] = 5
state['preset_targets']['POND14'][datetime(2020, 2, 7)] = 5
state['preset_targets']['POND14'][datetime(2020, 2, 8)] = 7
state['preset_targets']['POND14'][datetime(2020, 2, 9)] = 0
state['preset_targets']['POND14'][datetime(2020, 2, 11)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 12)] = 0
state['preset_targets']['POND14'][datetime(2020, 2, 13)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 14)] = 0
state['preset_targets']['POND14'][datetime(2020, 2, 15)] = 0
state['preset_targets']['POND14'][datetime(2020, 2, 16)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 21)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 22)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 23)] = 3
state['preset_targets']['POND14'][datetime(2020, 2, 24)] = 0
state['preset_targets']['POND14'][datetime(2020, 2, 25)] = 0
state['preset_targets']['POND14'][datetime(2020, 2, 26)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 1)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 2)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 3)] = 2
state['preset_targets']['POND14'][datetime(2020, 3, 4)] = 2 # or 8
state['preset_targets']['POND14'][datetime(2020, 3, 5)] = 7
state['preset_targets']['POND14'][datetime(2020, 3, 6)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 8)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 10)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 11)] = 0
state['preset_targets']['POND14'][datetime(2020, 3, 12)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 13)] = 4
state['preset_targets']['POND14'][datetime(2020, 3, 14)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 15)] = 5
state['preset_targets']['POND14'][datetime(2020, 3, 16)] = 5
state['preset_targets']['POND14'][datetime(2020, 3, 17)] = 5
state['preset_targets']['POND14'][datetime(2020, 3, 18)] = 5
state['preset_targets']['POND14'][datetime(2020, 3, 19)] = 5
state['preset_targets']['POND14'][datetime(2020, 3, 20)] = 7
state['preset_targets']['POND14'][datetime(2020, 3, 21)] = 7
state['preset_targets']['POND14'][datetime(2020, 3, 23)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 24)] = 3
state['preset_targets']['POND14'][datetime(2020, 3, 25)] = 5
# CHECKED
state['preset_targets']['POND16'] = {}
state['preset_targets']['POND16'][datetime(2018, 12, 11)] = 8
state['preset_targets']['POND16'][datetime(2018, 12, 22)] = 3
state['preset_targets']['POND16'][datetime(2018, 12, 23)] = 5
state['preset_targets']['POND16'][datetime(2018, 12, 24)] = 5
state['preset_targets']['POND16'][datetime(2018, 12, 25)] = 7
state['preset_targets']['POND16'][datetime(2018, 12, 26)] = 7
state['preset_targets']['POND16'][datetime(2019, 1, 1)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 2)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 3)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 4)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 5)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 6)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 7)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 8)] = 0
state['preset_targets']['POND16'][datetime(2019, 1, 9)] = 8
state['preset_targets']['POND16'][datetime(2019, 1, 10)] = 4 
state['preset_targets']['POND16'][datetime(2019, 1, 11)] = 5 
state['preset_targets']['POND16'][datetime(2019, 1, 12)] = 4 
state['preset_targets']['POND16'][datetime(2019, 1, 13)] = 7 
state['preset_targets']['POND16'][datetime(2019, 1, 14)] = 7
# CHECKED
state['preset_targets']['PSYC3'] = {}
state['preset_targets']['PSYC3'][datetime(2021, 3, 17)] = 0
state['preset_targets']['PSYC3'][datetime(2021, 3, 18)] = 0
state['preset_targets']['PSYC3'][datetime(2021, 3, 19)] = 4
state['preset_targets']['PSYC3'][datetime(2021, 3, 20)] = 5
state['preset_targets']['PSYC3'][datetime(2021, 3, 21)] = 5
state['preset_targets']['PSYC3'][datetime(2021, 3, 22)] = 5
state['preset_targets']['PSYC3'][datetime(2021, 3, 23)] = 5 #NOTE this has been incorrectly filled
state['preset_targets']['PSYC3'][datetime(2021, 3, 24)] = 5
state['preset_targets']['PSYC3'][datetime(2021, 3, 25)] = 5
state['preset_targets']['PSYC3'][datetime(2021, 3, 26)] = 0
state['preset_targets']['PSYC3'][datetime(2021, 3, 27)] = 0
# CHECKED
state['preset_targets']['PSYC7'] = {}
state['preset_targets']['PSYC7'][datetime(2019, 1, 2)] = 3
state['preset_targets']['PSYC7'][datetime(2019, 1, 3)] = 4
state['preset_targets']['PSYC7'][datetime(2019, 1, 4)] = 4
state['preset_targets']['PSYC7'][datetime(2019, 1, 5)] = 4
state['preset_targets']['PSYC7'][datetime(2019, 1, 6)] = 5
state['preset_targets']['PSYC7'][datetime(2019, 1, 7)] = 5
# CHECKED
state['preset_targets']['TIRRA80'] = {}
dy =  datetime(2018, 1, 1)
while dy < datetime(2018, 2, 24):
    state['preset_targets']['TIRRA80'][dy] = 0 
    dy = dy + timedelta(days=1)
dy =  datetime(2018, 2, 24)
while dy < datetime(2018, 4, 23):
    state['preset_targets']['TIRRA80'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA80'][datetime(2018, 3, 14)] = 0
state['preset_targets']['TIRRA80'][datetime(2018, 3, 18)] = 0
state['preset_targets']['TIRRA80'][datetime(2018, 3, 21)] = 0
state['preset_targets']['TIRRA80'][datetime(2018, 3, 27)] = 0
dy =  datetime(2018, 4, 23)
while dy < datetime(2018, 5, 8):
    state['preset_targets']['TIRRA80'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA80'][datetime(2018, 5, 8)] = 7
dy =  datetime(2018, 5, 9)
while dy < datetime(2018, 8, 6):
    state['preset_targets']['TIRRA80'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA80'][datetime(2018, 6, 25)] = 9
state['preset_targets']['TIRRA80'][datetime(2018, 6, 26)] = 9
dy =  datetime(2018, 8, 6)
while dy < datetime(2018, 11, 7):
    state['preset_targets']['TIRRA80'][dy] = 3
    dy = dy + timedelta(days=1)
dy =  datetime(2018, 11, 7)
while dy < datetime(2018, 12, 19):
    state['preset_targets']['TIRRA80'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA80'][datetime(2018, 11, 7)] = 1
state['preset_targets']['TIRRA80'][datetime(2018, 11, 21)] = 8
state['preset_targets']['TIRRA80'][datetime(2018, 12, 11)] = 8
state['preset_targets']['TIRRA80'][datetime(2018, 12, 17)] = 8
state['preset_targets']['TIRRA80'][datetime(2018, 12, 19)] = 5
# CHECKED
state['preset_targets']['TIRRA88'] = {}
state['preset_targets']['TIRRA88'][datetime(2019, 2, 17)] = 0
state['preset_targets']['TIRRA88'][datetime(2019, 2, 18)] = 3
state['preset_targets']['TIRRA88'][datetime(2019, 2, 19)] = 4
state['preset_targets']['TIRRA88'][datetime(2019, 2, 20)] = 4
state['preset_targets']['TIRRA88'][datetime(2019, 2, 21)] = 5
state['preset_targets']['TIRRA88'][datetime(2019, 2, 22)] = 5
state['preset_targets']['TIRRA88'][datetime(2019, 2, 23)] = 5
state['preset_targets']['TIRRA88'][datetime(2019, 2, 24)] = 5
state['preset_targets']['TIRRA88'][datetime(2019, 2, 25)] = 5
state['preset_targets']['TIRRA88'][datetime(2019, 2, 26)] = 5
# CHECKED
state['preset_targets']['TIRRA91'] = {}
dy =  datetime(2022, 1, 1)
while dy < datetime(2022, 7, 1):
    state['preset_targets']['TIRRA91'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA91'][datetime(2022, 1, 17)] = 1
state['preset_targets']['TIRRA91'][datetime(2022, 1, 22)] = 4
state['preset_targets']['TIRRA91'][datetime(2022, 1, 23)] = 5
state['preset_targets']['TIRRA91'][datetime(2022, 1, 24)] = 5
state['preset_targets']['TIRRA91'][datetime(2022, 1, 25)] = 4
state['preset_targets']['TIRRA91'][datetime(2022, 1, 26)] = 4
state['preset_targets']['TIRRA91'][datetime(2022, 1, 27)] = 4
dy =  datetime(2022, 1, 28)
while dy < datetime(2022, 2, 24):
    state['preset_targets']['TIRRA91'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA91'][datetime(2022, 2, 24)] = 7
dy =  datetime(2022, 2, 25)
while dy < datetime(2022, 7, 1):
    state['preset_targets']['TIRRA91'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRA91'][datetime(2022, 5, 29)] = 1
state['preset_targets']['TIRRA91'][datetime(2022, 6, 17)] = 8
# CHECKED
state['preset_targets']['TIRRA93'] = {}
state['preset_targets']['TIRRA93'][datetime(2021, 2, 1)] = 3
state['preset_targets']['TIRRA93'][datetime(2021, 2, 2)] = 5
state['preset_targets']['TIRRA93'][datetime(2021, 2, 3)] = 3
state['preset_targets']['TIRRA93'][datetime(2021, 2, 4)] = 4
state['preset_targets']['TIRRA93'][datetime(2021, 2, 5)] = 5
state['preset_targets']['TIRRA93'][datetime(2021, 2, 6)] = 5
state['preset_targets']['TIRRA93'][datetime(2021, 2, 7)] = 4
state['preset_targets']['TIRRA93'][datetime(2021, 2, 8)] = 7
state['preset_targets']['TIRRA93'][datetime(2021, 2, 9)] = 0
state['preset_targets']['TIRRA93'][datetime(2021, 2, 10)] = 0
# CHECKED
state['preset_targets']['TIRRW1'] = {} # battery fault is completely dependent on load - whether well is flowing or not
dy =  datetime(2018, 5, 11)
while dy < datetime(2018, 6, 6):
    state['preset_targets']['TIRRW1'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['TIRRW1'][datetime(2018, 5, 15)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 5, 16)] = 5 # probably need THP and Flow to detect these properly
state['preset_targets']['TIRRW1'][datetime(2018, 5, 17)] = 0
state['preset_targets']['TIRRW1'][datetime(2018, 5, 18)] = 0
state['preset_targets']['TIRRW1'][datetime(2018, 5, 19)] = 0
state['preset_targets']['TIRRW1'][datetime(2018, 5, 21)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 5, 27)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 5, 28)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 5, 30)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 5, 31)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 6, 1)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 6, 2)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 6, 3)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 6, 4)] = 5
state['preset_targets']['TIRRW1'][datetime(2018, 6, 5)] = 5
# CHECKED
state['well_classes']['TOOL16'] = [1]
state['preset_targets']['TOOL16'] = {}
dy =  datetime(2022, 1, 1)
while dy < datetime(2022, 6, 27):
    state['preset_targets']['TOOL16'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['TOOL16'][datetime(2022, 1, 23)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 1, 24)] = 1
state['preset_targets']['TOOL16'][datetime(2022, 1, 25)] = 7
state['preset_targets']['TOOL16'][datetime(2022, 4, 24)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 4, 25)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 4, 26)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 4, 27)] = 7
state['preset_targets']['TOOL16'][datetime(2022, 5, 10)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 5, 11)] = 1
state['preset_targets']['TOOL16'][datetime(2022, 5, 12)] = 7
state['preset_targets']['TOOL16'][datetime(2022, 5, 19)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 5, 20)] = 7
state['preset_targets']['TOOL16'][datetime(2022, 5, 29)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 5, 30)] = 7
state['preset_targets']['TOOL16'][datetime(2022, 6, 2)] = 9
state['preset_targets']['TOOL16'][datetime(2022, 6, 3)] = 1
state['preset_targets']['TOOL16'][datetime(2022, 6, 4)] = 7
# CHECKED
state['preset_targets']['TOOL29'] = {}
dy =  datetime(2018, 4, 14)
while dy < datetime(2018, 6, 15):
    state['preset_targets']['TOOL29'][dy] = 3 
    dy = dy + timedelta(days=1)
state['preset_targets']['TOOL29'][datetime(2018, 4, 27)] = 7
state['preset_targets']['TOOL29'][datetime(2018, 4, 30)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 5, 6)] = 7
state['preset_targets']['TOOL29'][datetime(2018, 5, 11)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 5, 13)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 5, 15)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 5, 17)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 5, 18)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 5, 19)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 5, 20)] = 7
state['preset_targets']['TOOL29'][datetime(2018, 5, 22)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 5, 23)] = 7
state['preset_targets']['TOOL29'][datetime(2018, 5, 25)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 5, 26)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 5, 27)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 5, 28)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 5, 29)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 6, 1)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 6, 4)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 6, 5)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 6, 6)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 6, 7)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 6, 8)] = 1
state['preset_targets']['TOOL29'][datetime(2018, 6, 10)] = 7
state['preset_targets']['TOOL29'][datetime(2018, 6, 12)] = 4
state['preset_targets']['TOOL29'][datetime(2018, 6, 15)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 6, 16)] = 5
state['preset_targets']['TOOL29'][datetime(2018, 6, 17)] = 5
# CHECKED
state['preset_targets']['WELTN1'] = {}
state['preset_targets']['WELTN1'][datetime(2019, 4, 21)] = 1
state['preset_targets']['WELTN1'][datetime(2019, 4, 22)] = 1
state['preset_targets']['WELTN1'][datetime(2019, 4, 23)] = 2
# CHECKED
state['preset_targets']['WELTN4'] = {}
dy =  datetime(2020, 9, 1)
while dy < datetime(2020, 9, 12):
    state['preset_targets']['WELTN4'][dy] = 6
    dy = dy + timedelta(days=1)
# CHECKED
# NOTE WIPPOS1 is not flowing and has zero THP so very different data from normal wells and probably should not be included in training
state['preset_targets']['WIPPOS1'] = {}
state['preset_targets']['WIPPOS1'][datetime(2019, 4, 26)] = 3
state['preset_targets']['WIPPOS1'][datetime(2019, 4, 27)] = 3
state['preset_targets']['WIPPOS1'][datetime(2019, 4, 28)] = 3
dy =  datetime(2019, 4, 29)
while dy < datetime(2019, 8, 5):
    state['preset_targets']['WIPPOS1'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['WIPPOS1'][datetime(2019, 5, 11)] = 4
state['preset_targets']['WIPPOS1'][datetime(2019, 5, 12)] = 4
state['preset_targets']['WIPPOS1'][datetime(2019, 5, 13)] = 3
# CHECKED
state['preset_targets']['WKT3'] = {}
dy =  datetime(2016, 6, 29)
while dy < datetime(2017, 1, 9):
    state['preset_targets']['WKT3'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['WKT3'][datetime(2016, 7, 20)] = 9
state['preset_targets']['WKT3'][datetime(2016, 8, 9)] = 9
state['preset_targets']['WKT3'][datetime(2016, 8, 15)] = 8
state['preset_targets']['WKT3'][datetime(2016, 8, 22)] = 9
state['preset_targets']['WKT3'][datetime(2016, 8, 23)] = 9
state['preset_targets']['WKT3'][datetime(2016, 8, 29)] = 8
state['preset_targets']['WKT3'][datetime(2016, 9, 1)] = 9
state['preset_targets']['WKT3'][datetime(2016, 9, 2)] = 9 # check THP values
state['preset_targets']['WKT3'][datetime(2016, 9, 13)] = 9
state['preset_targets']['WKT3'][datetime(2016, 9, 14)] = 9
state['preset_targets']['WKT3'][datetime(2016, 9, 15)] = 9
state['preset_targets']['WKT3'][datetime(2016, 9, 17)] = 9
state['preset_targets']['WKT3'][datetime(2016, 9, 20)] = 9
state['preset_targets']['WKT3'][datetime(2016, 9, 24)] = 9
state['preset_targets']['WKT3'][datetime(2016, 10, 2)] = 9
state['preset_targets']['WKT3'][datetime(2016, 12, 1)] = 3
state['preset_targets']['WKT3'][datetime(2016, 12, 5)] = 3
state['preset_targets']['WKT3'][datetime(2016, 12, 7)] = 9
state['preset_targets']['WKT3'][datetime(2016, 12, 8)] = 9
state['preset_targets']['WKT3'][datetime(2016, 12, 15)] = 9
state['preset_targets']['WKT3'][datetime(2016, 12, 29)] = 3
state['preset_targets']['WKT3'][datetime(2017, 1, 2)] = 3
state['preset_targets']['WKT3'][datetime(2017, 1, 5)] = 3
dy =  datetime(2017, 1, 9)
while dy < datetime(2017, 1, 19):
    state['preset_targets']['WKT3'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['WKT3'][datetime(2017, 1, 13)] = 0
state['preset_targets']['WKT3'][datetime(2017, 1, 15)] = 0
dy =  datetime(2017, 1, 19)
while dy < datetime(2017, 1, 25):
    state['preset_targets']['WKT3'][dy] = 0
    dy = dy + timedelta(days=1)
dy =  datetime(2017, 1, 25)
while dy < datetime(2017, 1, 29):
    state['preset_targets']['WKT3'][dy] = 3
    dy = dy + timedelta(days=1)
dy =  datetime(2017, 1, 29)
while dy < datetime(2018, 7, 1):
    state['preset_targets']['WKT3'][dy] = 0
    dy = dy + timedelta(days=1)
state['preset_targets']['WKT3'][datetime(2017, 3, 8)] = 8
state['preset_targets']['WKT3'][datetime(2017, 3, 31)] = 3
state['preset_targets']['WKT3'][datetime(2017, 4, 18)] = 9
state['preset_targets']['WKT3'][datetime(2017, 4, 19)] = 9
state['preset_targets']['WKT3'][datetime(2017, 4, 22)] = 3
state['preset_targets']['WKT3'][datetime(2017, 4, 23)] = 3
state['preset_targets']['WKT3'][datetime(2017, 4, 24)] = 3
state['preset_targets']['WKT3'][datetime(2017, 5, 18)] = 9
state['preset_targets']['WKT3'][datetime(2017, 6, 16)] = 9
state['preset_targets']['WKT3'][datetime(2017, 9, 27)] = 8
state['preset_targets']['WKT3'][datetime(2017, 9, 28)] = 8
state['preset_targets']['WKT3'][datetime(2017, 9, 29)] = 9
state['preset_targets']['WKT3'][datetime(2017, 9, 30)] = 9
state['preset_targets']['WKT3'][datetime(2017, 10, 11)] = 9
state['preset_targets']['WKT3'][datetime(2017, 10, 31)] = 3
state['preset_targets']['WKT3'][datetime(2017, 11, 1)] = 3
state['preset_targets']['WKT3'][datetime(2018, 1, 24)] = 3
state['preset_targets']['WKT3'][datetime(2018, 2, 14)] = 9
state['preset_targets']['WKT3'][datetime(2018, 2, 20)] = 3
state['preset_targets']['WKT3'][datetime(2018, 3, 7)] = 9
state['preset_targets']['WKT3'][datetime(2018, 3, 8)] = 9
state['preset_targets']['WKT3'][datetime(2018, 3, 9)] = 9
state['preset_targets']['WKT3'][datetime(2018, 3, 21)] = 9
state['preset_targets']['WKT3'][datetime(2018, 4, 19)] = 8
state['preset_targets']['WKT3'][datetime(2018, 5, 3)] = 9
state['preset_targets']['WKT3'][datetime(2018, 6, 20)] = 9
state['preset_targets']['WKT3'][datetime(2018, 6, 26)] = 9
state['preset_targets']['WKT3'][datetime(2019, 10, 26)] = 0
state['preset_targets']['WKT3'][datetime(2019, 10, 27)] = 0
state['preset_targets']['WKT3'][datetime(2019, 10, 28)] = 0
state['preset_targets']['WKT3'][datetime(2019, 10, 29)] = 1
state['preset_targets']['WKT3'][datetime(2019, 10, 30)] = 2
state['preset_targets']['WKT3'][datetime(2019, 10, 31)] = 7
state['preset_targets']['WKT3'][datetime(2019, 11, 1)] = 0
state['preset_targets']['WKT3'][datetime(2019, 11, 2)] = 0
state['preset_targets']['WKT3'][datetime(2019, 11, 3)] = 2
state['preset_targets']['WKT3'][datetime(2019, 11, 4)] = 7
state['preset_targets']['WKT3'][datetime(2019, 11, 5)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 8)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 9)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 10)] = 5
state['preset_targets']['WKT3'][datetime(2019, 11, 11)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 12)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 13)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 14)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 15)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 16)] = 5
state['preset_targets']['WKT3'][datetime(2019, 11, 17)] = 5
state['preset_targets']['WKT3'][datetime(2019, 11, 18)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 19)] = 7
state['preset_targets']['WKT3'][datetime(2019, 11, 20)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 21)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 22)] = 0
state['preset_targets']['WKT3'][datetime(2019, 11, 23)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 24)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 25)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 26)] = 4
state['preset_targets']['WKT3'][datetime(2019, 11, 27)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 28)] = 3
state['preset_targets']['WKT3'][datetime(2019, 11, 29)] = 4
dy =  datetime(2019, 11, 30)
while dy < datetime(2019, 12, 9):
    state['preset_targets']['WKT3'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['WKT3'][datetime(2019, 12, 2)] = 4
state['preset_targets']['WKT3'][datetime(2019, 12, 7)] = 3
state['preset_targets']['WKT3'][datetime(2019, 12, 9)] = 3
state['preset_targets']['WKT3'][datetime(2019, 12, 10)] = 0
state['preset_targets']['WKT3'][datetime(2019, 12, 11)] = 3
state['preset_targets']['WKT3'][datetime(2019, 12, 12)] = 3
state['preset_targets']['WKT3'][datetime(2019, 12, 13)] = 3
dy =  datetime(2020, 3, 14)
while dy < datetime(2020, 4, 4):
    state['preset_targets']['WKT3'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['WKT3'][datetime(2020, 3, 20)] = 5
state['preset_targets']['WKT3'][datetime(2020, 3, 21)] = 7
state['preset_targets']['WKT3'][datetime(2020, 3, 27)] = 4
state['preset_targets']['WKT3'][datetime(2020, 3, 28)] = 5
state['preset_targets']['WKT3'][datetime(2020, 3, 29)] = 4
state['preset_targets']['WKT3'][datetime(2020, 3, 30)] = 4
state['preset_targets']['WKT3'][datetime(2020, 3, 31)] = 5
state['preset_targets']['WKT3'][datetime(2020, 4, 1)] = 7
state['preset_targets']['WKT3'][datetime(2020, 4, 2)] = 5
state['preset_targets']['WKT3'][datetime(2020, 4, 3)] = 4
state['preset_targets']['WKT3'][datetime(2020, 4, 4)] = 0
state['preset_targets']['WKT3'][datetime(2020, 4, 5)] = 0
state['preset_targets']['WKT3'][datetime(2020, 4, 6)] = 0
state['preset_targets']['WKT3'][datetime(2020, 4, 7)] = 0
dy =  datetime(2020, 4, 26)
while dy < datetime(2020, 8, 10):
    state['preset_targets']['WKT3'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['WKT3'][datetime(2020, 4, 27)] = 0
state['preset_targets']['WKT3'][datetime(2020, 4, 28)] = 0
state['preset_targets']['WKT3'][datetime(2020, 5, 18)] = 0
state['preset_targets']['WKT3'][datetime(2020, 5, 12)] = 1
state['preset_targets']['WKT3'][datetime(2020, 5, 19)] = 0
state['preset_targets']['WKT3'][datetime(2020, 5, 20)] = 1
state['preset_targets']['WKT3'][datetime(2020, 6, 13)] = 1
state['preset_targets']['WKT3'][datetime(2020, 6, 20)] = 1
state['preset_targets']['WKT3'][datetime(2020, 6, 21)] = 8
state['preset_targets']['WKT3'][datetime(2020, 7, 9)] = 1
state['preset_targets']['WKT3'][datetime(2020, 7, 10)] = 1
state['preset_targets']['WKT3'][datetime(2020, 7, 21)] = 1
state['preset_targets']['WKT3'][datetime(2020, 8, 6)] = 1
state['preset_targets']['WKT3'][datetime(2020, 8, 7)] = 1
state['preset_targets']['WKT3'][datetime(2020, 8, 10)] = 5
# CHECKED
state['preset_targets']['YAN1L'] = {}
state['preset_targets']['YAN1L'][datetime(2019, 5, 19)] = 8
state['preset_targets']['YAN1L'][datetime(2019, 6, 14)] = 5
state['preset_targets']['YAN1L'][datetime(2021, 4, 27)] = 3
state['preset_targets']['YAN2L'] = {}
state['preset_targets']['YAN2L'][datetime(2019, 2, 20)] = 4
state['preset_targets']['YAN2L'][datetime(2019, 2, 22)] = 4
state['preset_targets']['YAN2L'][datetime(2019, 6, 11)] = 5
state['preset_targets']['YAN2L'][datetime(2020, 10, 27)] = 6 # actual charging fault
state['preset_targets']['YAN2L'][datetime(2020, 10, 28)] = 6 # actual charging fault
state['preset_targets']['YAN2L'][datetime(2021, 5, 8)] = 8 # or 1
# CHECKED
state['preset_targets']['YAN2U'] = {}
dy =  datetime(2018, 1, 1)
while dy < datetime(2018, 1, 8):
    state['preset_targets']['YAN2U'][dy] = 3
    dy = dy + timedelta(days=1)
dy =  datetime(2018, 1, 8)
while dy < datetime(2018, 1, 30):
    state['preset_targets']['YAN2U'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN2U'][datetime(2018, 1, 10)] = 3
state['preset_targets']['YAN2U'][datetime(2018, 1, 11)] = 3
state['preset_targets']['YAN2U'][datetime(2018, 1, 12)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 1, 14)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 1, 15)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 1, 17)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 1, 18)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 1, 19)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 1, 29)] = 5
dy =  datetime(2018, 5, 21)
while dy < datetime(2018, 6, 16):
    state['preset_targets']['YAN2U'][dy] = 8
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN2U'][datetime(2018, 6, 15)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 6, 16)] = 3 
state['preset_targets']['YAN2U'][datetime(2018, 6, 17)] = 3 
dy =  datetime(2018, 11, 8)
while dy < datetime(2018, 12, 12):
    state['preset_targets']['YAN2U'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN2U'][datetime(2018, 11, 12)] = 0
state['preset_targets']['YAN2U'][datetime(2018, 11, 13)] = 1
state['preset_targets']['YAN2U'][datetime(2018, 11, 21)] = 1
dy =  datetime(2018, 12, 12)
while dy < datetime(2019, 1, 31):   
    state['preset_targets']['YAN2U'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN2U'][datetime(2019, 1, 31)] = 5
state['preset_targets']['YAN2U'][datetime(2019, 2, 1)] = 5
dy =  datetime(2019, 2, 2)
while dy < datetime(2019, 2, 24):   
    state['preset_targets']['YAN2U'][dy] = 7
    dy = dy + timedelta(days=1)
dy =  datetime(2019, 4, 13)
while dy < datetime(2019, 4, 22):   
    state['preset_targets']['YAN2U'][dy] = 5
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN2U'][datetime(2019, 4, 23)] = 7
dy =  datetime(2020, 11, 20)
while dy < datetime(2020, 12, 1):   
    state['preset_targets']['YAN2U'][dy] = 8
    dy = dy + timedelta(days=1)
dy =  datetime(2021, 1, 30)
while dy < datetime(2021, 3, 16):   
    state['preset_targets']['YAN2U'][dy] = 8
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN2U'][datetime(2021, 2, 12)] = 0
state['preset_targets']['YAN2U'][datetime(2021, 2, 23)] = 0
state['preset_targets']['YAN2U'][datetime(2021, 2, 27)] = 0 #got to here xxx
# CHECKED
state['preset_targets']['YAN4U'] = {}
state['preset_targets']['YAN4U'][datetime(2018, 3, 7)] = 1
state['preset_targets']['YAN4U'][datetime(2018, 3, 8)] = 2
state['preset_targets']['YAN4U'][datetime(2018, 3, 9)] = 1
state['preset_targets']['YAN4U'][datetime(2018, 3, 10)] = 7
dy =  datetime(2018, 5, 21)
while dy < datetime(2018, 6, 15):   
    state['preset_targets']['YAN4U'][dy] = 8
    dy = dy + timedelta(days=1)
dy =  datetime(2018, 10, 26)
while dy < datetime(2018, 10, 30):
    state['preset_targets']['YAN4U'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN4U'][datetime(2018, 10, 28)] = 1
state['preset_targets']['YAN4U'][datetime(2018, 10, 30)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 10, 31)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 11, 1)] = 3
dy =  datetime(2018, 11, 2)
while dy < datetime(2019, 1, 20):
    state['preset_targets']['YAN4U'][dy] = 4
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN4U'][datetime(2018, 11, 6)] = 1
state['preset_targets']['YAN4U'][datetime(2018, 11, 7)] = 2
state['preset_targets']['YAN4U'][datetime(2018, 11, 8)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 11, 13)] = 1
state['preset_targets']['YAN4U'][datetime(2018, 11, 17)] = 3
state['preset_targets']['YAN4U'][datetime(2018, 11, 21)] = 2
state['preset_targets']['YAN4U'][datetime(2018, 11, 22)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 11, 27)] = 5
state['preset_targets']['YAN4U'][datetime(2018, 11, 28)] = 5
state['preset_targets']['YAN4U'][datetime(2018, 11, 29)] = 5
state['preset_targets']['YAN4U'][datetime(2018, 11, 30)] = 5
state['preset_targets']['YAN4U'][datetime(2018, 12, 1)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 12, 8)] = 4 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 12, 9)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 12, 10)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 12, 11)] = 3 # or 7
state['preset_targets']['YAN4U'][datetime(2018, 12, 12)] = 3 # or 8
state['preset_targets']['YAN4U'][datetime(2019, 1, 6)] = 5
state['preset_targets']['YAN4U'][datetime(2019, 1, 7)] = 5
state['preset_targets']['YAN4U'][datetime(2019, 1, 9)] = 5
state['preset_targets']['YAN4U'][datetime(2019, 1, 12)] = 5
state['preset_targets']['YAN4U'][datetime(2019, 1, 15)] = 5
state['preset_targets']['YAN4U'][datetime(2019, 1, 20)] = 7
state['preset_targets']['YAN4U'][datetime(2019, 1, 21)] = 0
# CHECKED
state['preset_targets']['YAN8L'] = {}
state['preset_targets']['YAN8L'][datetime(2018, 2, 1)] = 3
state['preset_targets']['YAN8L'][datetime(2018, 2, 2)] = 5
state['preset_targets']['YAN8L'][datetime(2019, 8, 16)] = 4
state['preset_targets']['YAN8L'][datetime(2019, 8, 17)] = 3
state['preset_targets']['YAN8L'][datetime(2019, 8, 18)] = 4
state['preset_targets']['YAN8L'][datetime(2019, 8, 19)] = 5
state['preset_targets']['YAN8L'][datetime(2021, 7, 27)] = 3
state['preset_targets']['YAN8L'][datetime(2021, 7, 28)] = 3
state['preset_targets']['YAN8L'][datetime(2021, 7, 29)] = 4
state['preset_targets']['YAN8L'][datetime(2021, 7, 30)] = 8
dy =  datetime(2020, 12, 6)
while dy < datetime(2020, 12, 12):
    state['preset_targets']['YAN8L'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN8L'][datetime(2020, 12, 8)] = 0
state['preset_targets']['YAN8L'][datetime(2020, 12, 10)] = 0
dy =  datetime(2020, 12, 12)
while dy < datetime(2021, 3, 19):
    state['preset_targets']['YAN8L'][dy] = 0
    dy = dy + timedelta(days=1)
dy =  datetime(2021, 3, 19)
while dy < datetime(2021, 5, 11):
    state['preset_targets']['YAN8L'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN8L'][datetime(2021, 3, 23)] = 7
state['preset_targets']['YAN8L'][datetime(2021, 3, 24)] = 0
state['preset_targets']['YAN8L'][datetime(2021, 3, 25)] = 0
state['preset_targets']['YAN8L'][datetime(2021, 4, 16)] = 1
state['preset_targets']['YAN8L'][datetime(2021, 5, 5)] = 4
dy =  datetime(2021, 5, 11)
while dy < datetime(2021, 5, 19):
    state['preset_targets']['YAN8L'][dy] = 4
    dy = dy + timedelta(days=1)
dy =  datetime(2021, 5, 19)
while dy < datetime(2021, 6, 4):
    state['preset_targets']['YAN8L'][dy] = 3
    dy = dy + timedelta(days=1)
dy =  datetime(2021, 6, 4)
while dy < datetime(2021, 6, 22):
    state['preset_targets']['YAN8L'][dy] = 4
    dy = dy + timedelta(days=1)
dy =  datetime(2021, 6, 15)
while dy < datetime(2021, 7, 29):
    state['preset_targets']['YAN8L'][dy] = 3
    dy = dy + timedelta(days=1)
state['preset_targets']['YAN8L'][datetime(2021, 6, 23)] = 1
state['preset_targets']['YAN8L'][datetime(2021, 6, 27)] = 8
state['preset_targets']['YAN8L'][datetime(2021, 6, 28)] = 8
state['preset_targets']['YAN8L'][datetime(2021, 7, 29)] = 4
state['preset_targets']['YAN8L'][datetime(2021, 7, 30)] = 8
# CHECKED
###########################################################################

# 0=Normal
# 1=Battery Capacity
# 2=SIDLV - Battery Capacity
# 3=Battery Degrading
# 4=Battery Fault
# 5=SIDLV - Battery Fault
# 6=Charging Fault
# 7=Battery Recovering
# 8=Data Anomaly
# 9=Cloud Cover

state['procdata_con'] = connect_(config['procdata_info'])
classes = ['Normal', 'Battery Capacity', 'SIDLV - Battery Capacity', 
           'Battery Degrading', 'Battery Fault', 'SIDLV - Battery Fault', 
           'Charging Fault', 'Battery Recovering', 'Data Anomaly', 'Cloud Cover']

if __name__ == "__main__":
    with open(Path.config("well_labels.yaml"),'w') as file: 
        yaml.dump(state['preset_targets'], file)