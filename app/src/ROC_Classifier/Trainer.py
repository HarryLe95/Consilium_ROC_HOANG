from src.utils import Path, get_classifier, IoU
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf 
from typing import Sequence
import logging 
import numpy as np 
import datetime

logger = logging.getLogger(__name__)

class ROC_Classifier:
    def __init__(self, 
                 num_classes: int = 2,
                 num_features: int = 1440,
                 optimiser:str='adam',
                 base_lr:float=1e-3,
                 early_stopping_patience:int=10,
                 reduce_lr_patience:int=5,
                 num_epochs:int=100,
                 metrics:str|Sequence[str]=['acc','iou'],
                 use_pretrain:bool=False,
                 pretrain_model:str=None,
                 save_model:bool=False,
                 save_name:str=None):
        self.num_classes = num_classes
        self.num_features = num_features 
        self.opt = optimiser
        self.base_lr = base_lr 
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.num_epochs = num_epochs 
        self.metrics_list = metrics 
        self.use_pretrain = use_pretrain 
        self.pretrain_model = pretrain_model 
        self.save_model = save_model 
        self.save_name = save_name 
    
    def get_opt(self):
        logger.debug(f"Using optimiser: {self.opt}")
        if self.opt == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.base_lr)
        elif self.opt == 'sgd':
            return tf.keras.optimizer.SGD(learning_rate=self.base_lr)
        else:
            logger.warning("Unsupported optimiser, reverting to adam.")
            return tf.keras.optimizers.Adam(learning_rate=self.base_lr)
    
    def get_model(self):
        if self.use_pretrain:
            logger.debug(f"Loading pretrained model at {self.pretrain_model}.")
            if self.custom_metrics is not None: 
                return tf.keras.models.load_model(Path.model('model', self.pretrain_model), custom_objects = self.custom_metrics)
            else:
                return tf.keras.model.load_model(Path.model(self.pretrain_model))
        logger.debug(f"Loading random init model.")
        return get_classifier(self.num_classes, self.num_features)
    
    def get_metrics(self):
        metrics = []
        custom_metrics = None
        for m in self.metrics_list:
            if "iou" in self.metrics_list:
                metrics.append(IoU(self.num_classes))
                custom_metrics = {"IoU":IoU}
            else:
                metrics.append(m)
        logger.debug(f"Using metrics for model training/testing: {self.metrics_list}")
        if custom_metrics is not None:
            logger.debug(f"Using custom metric: {list(custom_metrics.keys())}")
        return metrics, custom_metrics
            
    def get_callbacks(self):
        logger.debug(f"Using early stopping callback with patience: {self.early_stopping_patience}")
        logger.debug(f"Using reduce learning rate on plateau callback with patience: {self.reduce_lr_patience}")
        callbacks = [tf.keras.callbacks.EarlyStopping(    monitor='val_loss', mode = 'min', patience=self.early_stopping_patience),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode = 'min', patience=self.reduce_lr_patience)]
        return callbacks
        
    def get_loss(self):
        logger.debug("Using Crossentropy for model training")
        return tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    def setup(self):
        self.optimiser = self.get_opt()
        self.callbacks = self.get_callbacks()
        self.metrics, self.custom_metrics = self.get_metrics()
        self.loss = self.get_loss()
        self.model = self.get_model()
        self.model.compile(optimizer= self.optimiser,
                  loss=self.loss,
                  metrics=self.metrics)
        logger.debug("Compile model.")
        
    def fit(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
        logger.debug("Fit to training and validation dataset.")
        history = self.model.fit(train_dataset, validation_data= val_dataset, callbacks = self.callbacks, epochs = self.num_epochs)
        
        if self.save_model:
            self.model.save(Path.model("model",self.save_name))
            logger.debug(f"Save trained model to {self.save_name}")
        return history 
    
    def predict(self, test_dataset: tf.data.Dataset|np.ndarray): 
        logger.debug("Getting model's prediction on test dataset")
        predictions = self.model.predict(test_dataset)
        return predictions 
    
    def _predict(self, test_dataset: tf.data.Dataset|np.ndarray): 
        for i,(img,label) in enumerate(test_dataset):
            pred = self.model.predict(img)
            if i == 0:
                y_true = label
                y_pred = pred
            else:
                y_true = np.concatenate([y_true, label])
                y_pred = np.concatenate([y_pred, pred])
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        return y_true, y_pred
    
    def evaluate_binary(self, test_dataset: tf.data.Dataset|np.ndarray, TS:np.ndarray):
        y_true, y_pred = self._predict(test_dataset)
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        mask = y_true != y_pred 
        mismatch = TS[mask]
        logger.debug("Evaluating model's performance - confusion matrix:")
        logger.debug(f"\n{cm}")
        return cm, mismatch 
        