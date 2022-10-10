import yaml 
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from typing import Sequence
from src.utils.PathManager import Paths as Path
from datetime import datetime, timedelta

def get_scaled_images(df: pd.DataFrame, 
                      scaler: StandardScaler, 
                      features: Sequence[str], 
                      num_days:int=7,
                      sequence_len:int=1440) -> np.ndarray:
    """Scale and transform features of an input dataframe using input scaler

    Each row in the dataframe corresponds to data collected for the full day, which can be a sequence of recordings 
    -i.e. df.ROC_VOLTAGE.loc[0] is a vector of size 1440, corresponding to ROC_VOLTAGE data collected minutely. 

    For each feature in the dataframe, the corresponding output'features are obtained by:
    1/ Normalising the feature using scaler.transform()
    2/ Concatenating data from the previous num_days.
    
    Args:
        df (pd.DataFrame): input dataframe with features as columns
        scaler (StandardScaler): scaler 
        features (Sequence[str]): columns/features to be transformed.
        num_days (int, optional): number of consecutive days to be include in the transformed sequence. Defaults to 7.
        sequence_len(int, optional): length of a one day sequence. One of 1440 (well_features) or 24 (weather). Defaults to 1440 for well features.
    Returns:
        np.ndarray: scaled and transformed image.
    """
    for index, feature in enumerate(features):
        image = np.vstack(df[feature].values) #Convert a 1D np.ndarray of list to 2D np.ndarray
        shape = image.shape
        if scaler is not None:
            if "Mask" not in feature:
                image = scaler[feature].transform(image.reshape(-1,1)).reshape(shape)
        image = image[:,-sequence_len*num_days:]
        if index == 0:
            all_image = image
        else:
            all_image = np.concatenate([all_image, image],axis=1)
    return all_image
    
def get_dataset_image_label(well_name: str, 
                            well_features: Sequence[str]=['ROC_VOLTAGE'], 
                            weather_features: Sequence[str] = None,
                            num_days:int=7,
                            file_post_fix:str="2016-01-01_2023-01-01_labelled", 
                            file_ext:str='pkl',
                            scaler:StandardScaler=None, 
                            label_mapping: dict=None, 
                            drop_labels: Sequence[int]=None, 
                            last_day:str|datetime=None,
                            label_col:str="labels") -> tuple[np.ndarray,np.ndarray, np.ndarray]:
    """Create an image label TS triplet with all preprocessing steps applied.

    Preprocessing steps include: 
    1/Reading in a predetermined dataset split 
    2/Normalise the features
    3/Concatenate features so each instance is a sequence of num_days days
    4/Relabel data based on label_mapping 
    5/Remove data with labels in drop_labels
    
    Args:
        well (str): well name
        well_features (Sequence[str], optional): well features. Defaults to ['ROC_VOLTAGE'].
        weather_features (Sequence[str], optional): weather features. Defaults to None.
        num_days (int, optional): number of consecutive days to be include in the transformed sequence. Defaults to 7.
        file_post_fix (str, optional): post-fix file name. Defaults to "2016-01-01_2023-01-01_labelled".
        file_ext (str, optional): labelled data file extension. Defaults to pkl
        scaler (StandardScaler, optional): data scaler. Defaults to None.
        label_mapping (dict, optional): dictionary specifying relabelling rules. Defaults to None - no relabelling.
        drop_labels (Sequence[int], optional): list of labels to be dropped from the dataset. Defaults to None.
        label_col (str, optional): name of label column 

    Returns:
        tuple[np.ndarray,np.ndarray, np.ndarray]: image_well, image_weather, label, timestamp tuple
    """
    #Load dataset
    if file_ext =="pkl":
        df = pd.read_pickle(Path.data(f"{well_name}_{file_post_fix}.{file_ext}"))
    elif file_ext == "csv":
        df = pd.read_csv(Path.data(f"{well_name}_{file_post_fix}.{file_ext}"), index_col="TS", parse_dates=["TS"])

    if last_day is not None:
        if isinstance(last_day,str):
            last_day = datetime.strptime(last_day,'%Y-%m-%d')
        df = df.loc[:last_day,:]

    #Map labels
    if drop_labels:
        df = df.drop(df[df.labels.isin(drop_labels)].index)
    if label_mapping:
        df = df.replace({"labels":label_mapping})
    all_well_image = get_scaled_images(df,scaler,well_features,num_days)
    all_weather_image = get_scaled_images(df,scaler,weather_features,num_days,24) \
                        if weather_features is not None else None
    label = df[label_col].values
    TS = df.index.to_numpy()
    # all_image = (all_well_image, all_weather_image) if all_weather_image is not None else all_well_image
    return all_well_image, all_weather_image, label, TS 


def get_random_split_from_image_label(image: np.ndarray, label: np.ndarray, TS: np.ndarray,
                                      train_ratio:float=0.8) -> tuple[np.ndarray]:
    """Split to training and validat sets from a triplet of image label TS

    Args:
        image (np.ndarray): image
        label (np.ndarray): label
        TS (np.ndarray): timestamp
        train_ratio (float, optional): fractional of original data to be used for training. Defaults to 0.8.

    Returns:
        tuple[np.ndarray]: tuple of two pairs of image/label/timestamp for training and validation sets
    """
    index = np.arange(len(image))
    np.random.shuffle(index)
    label=label.reshape(-1,1)
    train_index = index[:int(len(index)*train_ratio)]
    val_index = index[int(len(index)*train_ratio):]
    train_image = image[train_index,:]
    train_label = label[train_index,:]
    train_TS = TS[train_index]
    val_image = image[val_index,:]
    val_label = label[val_index,:]
    val_TS = TS[val_index]
    return train_image, train_label, train_TS, val_image, val_label, val_TS
    
def get_random_split_from_image_label(image_well: np.ndarray, 
                                      image_weather:np.ndarray,
                                      label: np.ndarray, TS: np.ndarray,
                                      train_ratio:float=0.8) -> tuple[np.ndarray]:
    """Split to training and validat sets from a tuple of image_well, image_weather, label TS

    Args:
        image_well (np.ndarray): image_well
        image_weather (np.ndarray): image_weather
        label (np.ndarray): label
        TS (np.ndarray): timestamp
        train_ratio (float, optional): fractional of original data to be used for training. Defaults to 0.8.

    Returns:
        tuple[np.ndarray]: tuple of two pairs of image_wells/image_weather/label/timestamp for training and validation sets
    """
    index = np.arange(len(image_well))
    np.random.shuffle(index)
    train_index = index[:int(len(index)*train_ratio)]
    val_index = index[int(len(index)*train_ratio):]
    train_image_well = image_well[train_index,:]
    train_image_weather = image_weather[train_index,:] if image_weather is not None else None 
    train_label = label[train_index,:]
    train_TS = TS[train_index]
    val_image_well = image_well[val_index,:]
    val_image_weather = image_weather[val_index,:] if image_weather is not None else None
    val_label = label[val_index,:]
    val_TS = TS[val_index]
    return train_image_well, train_image_weather, train_label, train_TS, val_image_well, val_image_weather, val_label, val_TS
    
def get_combined_data(well_name: str|Sequence[str], 
                      well_features: Sequence[str]=['ROC_VOLTAGE'],
                      weather_features: Sequence[str]=None, 
                      num_days:int=7, scaler:StandardScaler=None, 
                      label_mapping:dict=None, 
                      drop_labels:Sequence[int]=None,
                      last_day:str|datetime=None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an image_well, image_weather, label TS tuple with all preprocessing steps applied 
    from one well or from a group of wells

    Args:
        well_name (str | Sequence[str]): if str - name of one well, if list - name of a set of wells to be combined. 
        well_features (Sequence[str], optional): well features. Defaults to ['ROC_VOLTAGE'].
        weather_features (Sequence[str], optional): weather features. Defaults to None.
        num_days (int, optional): number of days to form data sequence. Defaults to 7.
        scaler (StandardScaler, optional): scaler. Defaults to None.
        label_mapping (dict, optional): dictionary specifying the relabelling rules. Defaults to None.
        drop_labels (Sequence[int], optional): list of labels to be dropped from dataset. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: image_well, image_weather, label, TS tuple of the dataset
    """
    #Dataset comprises of 1 well
    if isinstance(well_name,str):
        return get_dataset_image_label(well_name=well_name, 
                                       well_features=well_features,
                                       weather_features=weather_features, 
                                       num_days=num_days, scaler=scaler, 
                                       label_mapping=label_mapping, 
                                       drop_labels=drop_labels,
                                       last_day=last_day)
    #Dataset comprises of multiple wells
    if hasattr(well_name,'__iter__') and not isinstance(well_name,str):
        assert type(scaler)== type(well_name)
        assert len(scaler) == len(well_name)
        for index in range(len(well_name)):
            well_image, weather_image, well_label, well_TS = get_dataset_image_label(well_name=well_name[index], 
                                                                      well_features=well_features, 
                                                                      weather_features=weather_features,
                                                                      num_days=num_days, 
                                                                      scaler=scaler[index], 
                                                                      label_mapping=label_mapping, 
                                                                      drop_labels=drop_labels,
                                                                      last_day=last_day)
            if index == 0:
                image_well = well_image
                image_weather = weather_image if weather_features is not None else None 
                label = well_label
                TS = well_TS
            else:
                image_well = np.concatenate([image_well, well_image],axis=0)
                image_weather = np.concatenate([image_weather, weather_image],axis=0) if weather_features is not None else None 
                label = np.concatenate([label, well_label],axis=0)
                TS = np.concatenate([TS, well_TS])
        return image_well, image_weather, label, TS
    
def get_combined_regression_data(well_name: str|Sequence[str], 
                                 well_features: Sequence[str]=['ROC_VOLTAGE'],
                                 weather_features: Sequence[str]=None, 
                                 num_days:int=7, scaler:StandardScaler=None, 
                                 label_mapping:dict=None, 
                                 drop_labels:Sequence[int]=None,
                                 last_day:str|datetime=None,
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an image_well, image_weather, label TS tuple with all preprocessing steps applied 
    from one well or from a group of wells

    Args:
        well_name (str | Sequence[str]): if str - name of one well, if list - name of a set of wells to be combined. 
        well_features (Sequence[str], optional): well features. Defaults to ['ROC_VOLTAGE'].
        weather_features (Sequence[str], optional): weather features. Defaults to None.
        num_days (int, optional): number of days to form data sequence. Defaults to 7.
        scaler (StandardScaler, optional): scaler. Defaults to None.
        label_mapping (dict, optional): dictionary specifying the relabelling rules. Defaults to None.
        drop_labels (Sequence[int], optional): list of labels to be dropped from dataset. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: image_well, image_weather, label, TS tuple of the dataset
    """
    #Dataset comprises of 1 well
    if isinstance(well_name,str):
        return get_dataset_image_label(well_name=well_name, 
                                       well_features=well_features,
                                       weather_features=weather_features, 
                                       num_days=num_days,
                                       scaler=scaler, 
                                       file_post_fix="2016-01-01_2023-01-01_regression",
                                       label_col='days_to_failure',
                                       last_day=last_day)
    #Dataset comprises of multiple wells
    if hasattr(well_name,'__iter__') and not isinstance(well_name,str):
        assert type(scaler)== type(well_name)
        assert len(scaler) == len(well_name)
        for index in range(len(well_name)):
            well_image, weather_image, well_label, well_TS = get_dataset_image_label(well_name=well_name[index], 
                                                                      well_features=well_features, 
                                                                      weather_features=weather_features,
                                                                      num_days=num_days, 
                                                                      scaler=scaler[index], 
                                                                      file_post_fix="2016-01-01_2023-01-01_regression",
                                                                      label_col='days_to_failure',
                                                                      last_day=last_day)
            if index == 0:
                image_well = well_image
                image_weather = weather_image if weather_features is not None else None 
                label = well_label
                TS = well_TS
            else:
                image_well = np.concatenate([image_well, well_image],axis=0)
                image_weather = np.concatenate([image_weather, weather_image],axis=0) if weather_features is not None else None 
                label = np.concatenate([label, well_label],axis=0)
                TS = np.concatenate([TS, well_TS])
        return image_well, image_weather, label, TS

    
    