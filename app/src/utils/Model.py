import tensorflow as tf

class AnomalyDetector(tf.keras.Model):
    def __init__(self, num_features: int):
        super(AnomalyDetector, self).__init__()
        self.num_features = num_features
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu", name='dense_1'),
            tf.keras.layers.Dense(512,  activation="relu", name='dense_2'),
            tf.keras.layers.Dense(256,  activation="relu", name='dense_3'),
            tf.keras.layers.Dense(128,  activation="relu", name='dense_4')])

        # self.decoder = tf.keras.Sequential([
        #     tf.keras.layers.Dense(128,  activation="relu", name='dense_5'),
        #     tf.keras.layers.Dense(516,  activation="relu", name='dense_6'),
        #     tf.keras.layers.Dense(1024, activation="relu", name='dense_7'),
        #     tf.keras.layers.Dense(1440, name = 'dense_8'),
        #     tf.keras.layers.Reshape((1440,))])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        return {"num_features": self.num_features}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
def get_classifier(num_classes: int, num_features: int) -> tf.keras.Model:
    """Get classification model

    Args:
        num_classes (int): number of output classes
        num_features (int): dimension of input features

    Returns:
        tf.keras.Model: classification model
    """
    model = AnomalyDetector(num_features)
    inputs = tf.keras.Input(shape=(1440*num_features))
    x = model.encoder(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    classification_model = tf.keras.Model(inputs, outputs)
    return classification_model

def get_regressor(num_features:int) -> tf.keras.Model: 
    """Generate regression model

    Args:
        num_features (int): dimension of input features 

    Returns:
        tf.keras.Model: regression model 
    """
    model = AnomalyDetector(num_features)
    inputs = tf.keras.Input(shape=(1440*num_features))
    x = model.encoder(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    regression_model = tf.keras.Model(inputs, outputs)
    return regression_model