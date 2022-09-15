from src.utils import Path, get_dataset_from_image_label, get_combined_data, get_scaler, get_random_split_from_image_label
from typing import Sequence 
import yaml 
import tensorflow as tf 
import numpy as np
import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 
class ROC_Generator:
    """
    ROCGenerator class - provides dataset interface for model training and validation from stored csvs

    Method:
        self.setup: load data from csv and apply downstream transformations. Only call when neeeded as this might be computationally expensive.
    Attributes:
        self.dataset(tf.data.Dataset|tuple[tf.data.Dataset, tf.data.Dataset]): one dataset or a training/validation dataset pair.
    """
    with open(Path.config("transform_params.yaml"),'r') as file:
        config = yaml.safe_load(file)[0]
    
    def __init__(self, 
                 wells: str|Sequence[str], 
                 features: Sequence[str] = ["ROC_VOLTAGE"],
                 num_days:int=7,
                 normalise_mode:str='all',
                 label_mapping:dict={1:0,2:1,3:0,4:0,5:1},
                 drop_labels:Sequence[int]=[0,6,7,8,9],
                 split:bool=False,
                 split_ratio:float=0.8,
                 batch_size:int=32,
                 num_classes:int=2):
        """Model initialisation

        Args:
            wells (str | Sequence[str]): wells used to form dataset
            features (Sequence[str], optional): features to be included. Defaults to ["ROC_VOLTAGE"].
            num_days (int, optional): number of days to combine to form one data instance. Defaults to 7.
            normalise_mode (str, optional): one of ['all','one']. If 'one', normalise each well individually; if 'all', normalise wells by their Voltage Group. Defaults to 'all'.
            label_mapping (_type_, optional): labels remapping rules. Defaults to {1:0,2:1,3:0,4:0,5:1}.
            drop_labels (Sequence[int], optional): labels to drop. Defaults to [0,6,7,8,9].
            split (bool, optional): whether to split the dataset to training and validation. Defaults to False.
            split_ratio (float, optional): training fraction. Defaults to 0.8.
            batch_size (int, optional): batch_size. Defaults to 32.
            num_classes (int, optional): number of classes in the data. Defauls to 0.
        """
        self.wells = wells
        self.features = features 
        self.num_days = num_days 
        self.normalise_mode = normalise_mode 
        self.label_mapping = label_mapping 
        self.drop_labels = drop_labels 
        self.split = split 
        self.split_ratio = split_ratio
        self.num_classes = num_classes
        self.batch_size = batch_size 
        
    def _get_scaler(self):
        scaler = []
        for well in self.wells:
            scaler_well = well if self.normalise_mode == 'one' else self.config[well]
            scaler.append(get_scaler(self.normalise_mode, scaler_well, self.config, self.features))
        return scaler 
    
    def setup(self):
        self.scaler = self._get_scaler()
        image, label, TS = get_combined_data(self.wells, self.features, self.num_days, self.scaler, self.label_mapping, self.drop_labels)
        label = tf.keras.utils.to_categorical(label,self.num_classes)

        if self.split:
            train_image, train_label, train_TS, val_image, val_label, val_TS = get_random_split_from_image_label(image, label, TS, self.split_ratio)
            train_dataset = get_dataset_from_image_label(train_image, train_label,self.batch_size)
            val_dataset = get_dataset_from_image_label(val_image, val_label,self.batch_size)
            self.dataset = [train_dataset, val_dataset]
            self.TS = [train_TS, val_TS]
        else:
            self.dataset = get_dataset_from_image_label(image,label)
            self.TS = TS
        
        logger.debug(f"Prepared dataset for wells: {self.wells} with split: {self.split}")
