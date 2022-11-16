import tensorflow as tf
from tensorflow import keras 
import numpy as np 

class IoU(tf.keras.metrics.MeanIoU):
    """
    Get IOU metric from probability logits for training and testing evaluation. 
    
    This implementation improves over the built-in keras MeanIoU by thresholding the model's prediction (either via a hard threshold for binary or argmax for multiclass)
    before computing the confusion matrix. An option for having onehot multi-class Ground_Truth tensors as inputs is also provided.
    Similar to the builtin implementation, this metric is stateful, which can be used to compute 
    over batches in an epoch. Additionally provide an option to compute IoU over a specified list of target classes, which is available in 
    keras 2.8 but not 2.7.
    
    Example:
    >>> #Prepare inputs
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> num_classes = 5
    >>> num_samples = 10
    >>> y_true = np.random.randint(0,num_classes,(num_samples,))
    >>> y_pred_logits = np.random.random([num_samples,num_classes]) if num_classes > 2 else np.random.random(num_samples)
    >>> y_pred = tf.nn.softmax(y_pred_logits).numpy() if num_classes > 2 else y_pred_logits
    >>> y_pred_thres = np.argmax(y_pred,-1) if num_classes > 2 else y_pred>=0.5

    >>> #To compute the metric as an average of all classes
    >>> metrics = IoU(num_classes=num_classes) 
    >>> metrics.update_state(y_true,y_pred)
    >>> custom = metrics.result().numpy()

    >>> #To compare against the builtin mean iou: 
    >>> m = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    >>> m.update_state(y_true,y_pred_thres)
    >>> builtin = m.result().numpy()
    >>> custom==builtin
        True

    >>> #To reset state:
    >>> metrics.reset_state()

    >>> #To compute the metric over a specific class:
    >>> metrics = IoU(num_classes=num_classes, target_class_ids=[1.]) #IoU for label 1 specifically 
    >>> metrics.update_state(y_true,y_pred)
    >>> metrics.result().numpy()
    """
    def __init__(self, num_classes:int, threshold:float= 0.5, 
                 target_class_ids:list=None, name:str=None, dtype:str=None,
                 y_true_is_one_hot: bool=True, y_pred_is_probability: bool=True):
        """
        Initialising arguments
        Args:
            num_classes (int): if binary = 2, multiclass - >2
            threshold (int): only relevant for binary cases. Defaults to 0.5
            target_class_ids (list): list of classes to compute IoU over. If not provided, IoU will be computed over all classes. Defaults to None.
            y_true_is_one_hot (bool): if ingested gt labels are presented in one-hot format
            y_pred_is_probability (bool): if ingested prediction labels are presented in probability format.
        """
        super().__init__(num_classes=num_classes, name=name,dtype=dtype)
        self.threshold=threshold
        self.y_true_is_one_hot = y_true_is_one_hot
        self.y_pred_is_probability = y_pred_is_probability
        if target_class_ids:
            if max(target_class_ids) >= num_classes:
                raise ValueError(f'Target class id {max(target_class_ids)} is out of range, which is 'f'[{0}, {num_classes}).')
            self.target_class_ids = list(target_class_ids)
        else:
            self.target_class_ids = np.arange(num_classes)
    
    #Supplied methods for keras model loading 
    def get_config(self):
        return {
        'target_class_ids': self.target_class_ids,
        'threshold': self.threshold,
        'num_classes': self.num_classes,
        'name': self.name,
        'dtype': self._dtype,
        'y_true_is_one_hot': self.y_true_is_one_hot,
        'y_pred_is_probability': self.y_pred_is_probability
        }
    
    @classmethod
    def from_config(cls, config:dict):
        return cls(**config)
    
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray=None):
        """Accumulates the confusion matrix statistics.
        
        Before the confusion matrix is updated, the predicted values are thresholded
        to be:
          0 for values that are smaller than the `threshold`
          1 for values that are larger or equal to the `threshold`
          
        Args:
          y_true (np.ndarray): The ground truth values.
          y_pred (np.ndarray): The predicted values.
          sample_weight (np.ndarray): Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
            
        Returns:
          Update op.
        """
        y_pred = tf.cast(y_pred, self._dtype)
        if self.y_true_is_one_hot: 
            if self.num_classes!=2:
                y_true = tf.math.argmax(y_true,axis=-1)
        if self.y_pred_is_probability:
            if self.num_classes==2:
                y_pred = tf.cast(y_pred >= self.threshold, self._dtype)
            else:
                y_pred = tf.math.argmax(y_pred,axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)
    
    def result(self) -> float:
        """
        Compute the intersection-over-union via the confusion matrix.
        
        Return: 
            (float): IoU result
        """
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

        denominator = sum_over_row + sum_over_col - true_positives

        # Only keep the target classes
        true_positives = tf.gather(true_positives, self.target_class_ids)
        denominator = tf.gather(denominator, self.target_class_ids)

        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)


class Dice(IoU):
    """
    Implementation of the DICE coefficient. Derived from IoU implementation.
    """
    def __init__(self, num_classes:int, threshold:float= 0.5, 
                 target_class_ids:list=None, name:str=None, dtype:str=None,
                 y_true_is_one_hot: bool=True, y_pred_is_probability: bool=True):
        """
        Initialising arguments
        Args:
            num_classes:int - if binary = 2, multiclass - >2
            threshold:int - only relevant for binary cases - defaults to 0.5
            target_class_ids:list - list of classes to compute IoU over. If not provided, IoU will be computed over all classes
            y_true_is_one_hot: bool - if ingested gt labels are presented in one-hot format
            y_pred_is_probability: bool - if ingested prediction labels are presented in probability format.
        """
        super().__init__(num_classes=num_classes, threshold=threshold, target_class_ids=target_class_ids, name=name, dtype=dtype,
                        y_true_is_one_hot=y_true_is_one_hot, y_pred_is_probability=y_pred_is_probability)
    
    def result(self):
        """Compute the dice coefficient = 2tp/(2tp+fp+fn) via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col
        
        # Only keep the target classes
        true_positives = tf.gather(true_positives, self.target_class_ids)
        denominator = tf.gather(denominator, self.target_class_ids)

        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype))

        iou = tf.math.divide_no_nan(true_positives*2, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name='mean_dice'), num_valid_entries)
    
if __name__ == "__main__":
    num_classes = 5
    num_samples = 10
    
    #GT as a label vector and as a one-hot tensor
    y_true = np.random.randint(0,num_classes,(num_samples,))
    y_true_oh = keras.utils.to_categorical(y_true)
    
    #Prediction as a probability tensor and as a label vector
    y_pred_logits = np.random.random([num_samples,num_classes]) if num_classes > 2 else np.random.random(num_samples)
    y_pred = tf.nn.softmax(y_pred_logits).numpy() if num_classes > 2 else y_pred_logits
    y_pred_thres = np.argmax(y_pred,-1) if num_classes > 2 else y_pred>=0.5
    
    #Computing IOU
    #Built-in keras metric only deals with label vectors for prediction and ground-truths
    metric_builtin = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    metric_builtin.update_state(y_true,y_pred_thres)
    builtin_result = metric_builtin.result().numpy()
    
    #Custom IOU with ground-truths as onehot tensor and prediction as probability tensor 
    metric_IOU_oh_pr = IoU(num_classes=num_classes)
    metric_IOU_oh_pr.update_state(y_true_oh, y_pred)
    IOU_oh_pr_result = metric_IOU_oh_pr.result().numpy()
    
    #Custom IOU with ground-truths as label vector and predictions as probability tensor 
    metric_IOU_pr = IoU(num_classes=num_classes, y_true_is_one_hot=False)
    metric_IOU_pr.update_state(y_true, y_pred)
    IOU_pr_result = metric_IOU_pr.result().numpy()
    
    #Custom IOU with ground-truths and predictions as label vector
    metric_IOU = IoU(num_classes=num_classes, y_true_is_one_hot=False, y_pred_is_probability=False)
    metric_IOU.update_state(y_true, y_pred_thres)
    IOU_result = metric_IOU.result().numpy()
    
    #Custom IOU with ground-truths as one-hot tensor and predictions as label vector
    metric_IOU_oh = IoU(num_classes=num_classes, y_pred_is_probability=False)
    metric_IOU_oh.update_state(y_true_oh, y_pred_thres)
    IOU_oh_result=metric_IOU_oh.result().numpy()
    
    print(builtin_result == IOU_oh_pr_result == IOU_pr_result == IOU_result==IOU_oh_result)
    
    
    
    