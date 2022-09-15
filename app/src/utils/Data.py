import yaml 
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from typing import Sequence
from src.utils.PathManager import Paths as Path
from datetime import timedelta

def get_scaler_from_config(mean: float|Sequence[float],var: float|Sequence[float],scale: float|Sequence[float]) -> StandardScaler:
    """Create a Standard Scaler object from known data statistics

    Args:
        mean (float | Sequence[float]): data mean
        var (float | Sequence[float]):  data variance
        scale (float | Sequence[float]):data scale

    Returns:
        StandardScaler: scaler object 
    """
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = var
    scaler.scale_= scale
    return scaler
    
def get_scaler(mode: str, well_type:str, config:dict, features:Sequence[str]) -> dict[StandardScaler]:
    """Get a dictionary of StandardScaler objects that correspond to statistics specified in config 

    Args:
        mode (str): either one or all. If one is selected, a scaler is specified for every individual well. If all is selected, a scaler is specified for each well group (12V or 24V)
        well_type (str): if one is selected, well_type is individual well's name. Otherwise, well_type specifies the well group 
        config (dict): config file with known data statistics
        features (Sequence[str]): which features are to be scaled. Example: ["ROC_VOLTAGE", "FLOW"]

    Returns:
        dict[StandardScaler]: dictionary of scaler for each feature of each well/well_group 
    """
    params = config[mode][well_type]
    scaler_dict = {feature: get_scaler_from_config(params[feature]['mean'], params[feature]['var'], params[feature]['scale']) for feature in features}
    return scaler_dict

def get_scaled_images(df: pd.DataFrame, scaler: StandardScaler, features: Sequence[str], num_days:int=7) -> np.ndarray:
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

    Returns:
        np.ndarray: scaled and transformed image.
    """
    for index, feature in enumerate(features):
        image = np.vstack(df[feature].values) #Convert a 1D np.ndarray of list to 2D np.ndarray
        shape = image.shape
        image = scaler[feature].transform(image.reshape(-1,1)).reshape(shape)
        image = image[:,-1440*num_days:]
        if index == 0:
            all_image = image
        else:
            all_image = np.concatenate([all_image, image],axis=1)
    return all_image
    
def get_dataset_from_image_label(image: np.ndarray, label: np.ndarray, batch_size:int = 32, reconstruction:bool=False) -> tf.data.Dataset:
    """Create a dataset wrapper from an (image, label) pair for model training/testing

    Args:
        image (np.ndarray): data 
        label (np.ndarray): label
        batch_size (int, optional): batch_size. Defaults to 32.
        reconstruction (bool, optional): whether to create a dataset from non-positive class for model pretraining. Defaults to False.

    Returns:
        tf.data.Dataset: dataset wrapper
    """
    if reconstruction:
        image = image[label==0]
        return tf.data.Dataset.from_tensor_slices((image,image)).batch(batch_size)
    return tf.data.Dataset.from_tensor_slices((image,label)).batch(batch_size)

def get_dataset(well:str, split_type:str='train', features:Sequence[str]=['ROC_VOLTAGE'], num_days:int= 7, 
                scaler: StandardScaler=None, label_mapping:dict=None, drop_labels:Sequence[int]=None, batch_size:int=32, reconstruction:bool=False) -> tf.data.Dataset:
    """Create a dataset with all preprocessing steps applied.

    Preprocessing steps include: 
    1/Reading in a predetermined dataset split 
    2/Normalise the features
    3/Concatenate features so each instance is a sequence of num_days days
    4/Relabel data based on label_mapping 
    5/Remove data with labels in drop_labels
    6/Create a dataset object with batch_size
    
    Args:
        well (str): well name
        split_type (str, optional): can be 'train', 'val', None. If 'train' or 'val' is provided, Defaults to 'train'.
        features (Sequence[str], optional): _description_. Defaults to ['ROC_VOLTAGE'].
        num_days (int, optional): number of consecutive days to be include in the transformed sequence. Defaults to 7.
        scaler (StandardScaler, optional): data scaler. Defaults to None.
        label_mapping (dict, optional): dictionary specifying relabelling rules. Defaults to None - no relabelling.
        drop_labels (Sequence[int], optional): list of labels to be dropped from the dataset. Defaults to None.
        batch_size (int, optional): batch_size. Defaults to 32.
        reconstruction (bool, optional): whether to create a dataset from non-positive class for model pretraining. Defaults to False.

    Returns:
        tf.data.Dataset: output dataset
    """
    image, label,_ = get_manual_split_data(well, features, num_days, split_type, scaler, label_mapping, drop_labels)
    return get_dataset_from_image_label(image, label, batch_size, reconstruction)

def get_manual_split_data(well_name: str, features: Sequence[str]=['ROC_VOLTAGE'], num_days:int=7,
                          split_type:str=None, scaler:StandardScaler=None, label_mapping: dict=None, drop_labels: Sequence[int]=None) -> tuple[np.ndarray,np.ndarray, np.ndarray]:
    """Create an image label pair with all preprocessing steps applied.

    Preprocessing steps include: 
    1/Reading in a predetermined dataset split 
    2/Normalise the features
    3/Concatenate features so each instance is a sequence of num_days days
    4/Relabel data based on label_mapping 
    5/Remove data with labels in drop_labels
    
    Args:
        well (str): well name
        split_type (str, optional): can be 'train', 'val', None. If 'train' or 'val' is provided, Defaults to 'train'.
        features (Sequence[str], optional): _description_. Defaults to ['ROC_VOLTAGE'].
        num_days (int, optional): number of consecutive days to be include in the transformed sequence. Defaults to 7.
        scaler (StandardScaler, optional): data scaler. Defaults to None.
        label_mapping (dict, optional): dictionary specifying relabelling rules. Defaults to None - no relabelling.
        drop_labels (Sequence[int], optional): list of labels to be dropped from the dataset. Defaults to None.

    Returns:
        tuple[np.ndarray,np.ndarray, np.ndarray]: output data, label, timestamp tuple
    """
    #Load dataset
    df_path = Path.data(f"{well_name}_labelled_{split_type}.pkl") if split_type is not None else Path.data(f"{well_name}_labelled.pkl") 
    df = pd.read_pickle(df_path)


    #Map labels
    if drop_labels:
        df = df.drop(df[df.labels.isin(drop_labels)].index)
    if label_mapping:
        df = df.replace({"labels":label_mapping})
    #Transform
    if scaler:
        all_image = get_scaled_images(df,scaler,features,num_days)
    else:
        for index, feature in enumerate(features):
            image = np.vstack(df[feature].values)
            image = image[:,-1440*num_days:]
            if index == 0:
                all_image = image
            else: 
                all_image = np.concatenate([all_image, image],axis=1)
    label = df.labels.values
    TS = df.index.to_numpy()
    return all_image, label, TS 

def get_random_split_from_image_label(image: np.ndarray, label: np.ndarray, TS: np.ndarray,
                                      train_ratio:float=0.8) -> tuple[np.ndarray]:
    """Split to training and validat sets from a pair of image label

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
    train_index = index[:int(len(index)*train_ratio)]
    val_index = index[int(len(index)*train_ratio):]
    train_image = image[train_index,:]
    train_label = label[train_index,:]
    train_TS = TS[train_index]
    val_image = image[val_index,:]
    val_label = label[val_index,:]
    val_TS = TS[val_index]
    return train_image, train_label, train_TS, val_image, val_label, val_TS
    
def get_combined_data(well_name: str|Sequence[str], features: Sequence[str]=['ROC_VOLTAGE'], 
                      num_days:int=7, scaler:StandardScaler=None, label_mapping:dict=None, drop_labels:Sequence[int]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an image label TS triplet with all preprocessing steps applied from one well or from a group of wells

    Args:
        well_name (str | Sequence[str]): if str - name of one well, if list - name of a set of wells to be combined. 
        features (Sequence[str], optional): data features. Defaults to ['ROC_VOLTAGE'].
        num_days (int, optional): number of days to form data sequence. Defaults to 7.
        scaler (StandardScaler, optional): scaler. Defaults to None.
        label_mapping (dict, optional): dictionary specifying the relabelling rules. Defaults to None.
        drop_labels (Sequence[int], optional): list of labels to be dropped from dataset. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: image, label, TS triplet of the dataset
    """
    #Dataset comprises of 1 well
    if isinstance(well_name,str):
        return get_manual_split_data(well_name, features, num_days, None, scaler, label_mapping, drop_labels)
    #Dataset comprises of multiple wells
    if hasattr(well_name,'__iter__') and not isinstance(well_name,str):
        assert type(scaler)== type(well_name)
        assert len(scaler) == len(well_name)
        for index in range(len(well_name)):
            well_image, well_label, well_TS = get_manual_split_data(well_name[index], features, num_days, None,scaler[index], label_mapping,drop_labels)
            if index == 0:
                image = well_image
                label = well_label
                TS = well_TS
            else:
                image = np.concatenate([image, well_image],axis=0)
                label = np.concatenate([label, well_label],axis=0)
                TS = np.concatenate([TS, well_TS])
        return image, label, TS
    
def create_label_dataframe(well_name:str, window_size:int = 6, save_pickle:bool=False):
    df = pd.read_csv(Path.data(f"{well_name}_raw.csv"), index_col='TS', parse_dates = ['TS'])
    daily_df = df.groupby(df.index.date).agg(list)
    label_df = pd.read_pickle(Path.data(f"{well_name}_labelled.pkl"))
    daily_df.index = pd.to_datetime(daily_df.index)
    
    def process(df, index, label, new_df, max_date = 4):
        days = pd.date_range(index-timedelta(days=max_date),index)
        try:
            window = df.loc[days,:]
        except Exception as e:
            return 
        new_df.loc[index,'labels']=label
        for col in df.columns:
            new_df.loc[index,col] = np.hstack(window[col].values).astype(object)
        
    new_df = pd.DataFrame(columns = label_df.columns)
    for index in label_df.index:
        label = label_df.loc[index,'labels']
        process(daily_df, index, label, new_df, window_size)
    new_df['series_length'] = new_df.ROC_VOLTAGE.apply(len)
    sub_df = new_df[new_df.series_length!=1440*(window_size+1)]
    if len(sub_df):
        print(f"Well: {well_name}, incomplete data: {sub_df.labels.value_counts()}")
        new_df.drop(index = sub_df.index,inplace=True)
    if save_pickle:
        new_df.to_pickle(Path.data(f"{well_name}_labelled.pkl"))
    return new_df
    
    