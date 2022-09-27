import tensorflow as tf

class BaseClassifier(tf.keras.Model):
    def __init__(self, num_features: int):
        super(BaseClassifier, self).__init__()
        self.num_features = num_features
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu", name='dense_1'),
            tf.keras.layers.Dense(512,  activation="relu", name='dense_2'),
            tf.keras.layers.Dense(256,  activation="relu", name='dense_3'),
            tf.keras.layers.Dense(128,  activation="relu", name='dense_4')])
        
    def call(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def get_config(self):
        return {"num_features": self.num_features}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
def get_classifier(num_classes: int, num_well_features: int, num_weather_features: int=0) -> tf.keras.Model:
    """Get classification model

    Args:
        num_classes (int): number of output classes
        num_well_features (int): dimension of input well features
        num_weather_features (int): dimension of weather features. Defaults to 0

    Returns:
        tf.keras.Model: classification model
    """
    if num_weather_features != 0:
        well_model = BaseClassifier(num_well_features)
        weather_model = BaseClassifier(num_weather_features)
        well_inputs = tf.keras.Input(shape=(1440*num_well_features), name='well_features')
        weather_inputs = tf.keras.Input(shape=(24*num_weather_features), name='weather_features')
        well_features = well_model.encoder(well_inputs, training=False)
        weather_features = weather_model.encoder(weather_inputs, training=False)
        x = tf.keras.layers.concatenate([well_features, weather_features])
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        classification_model = tf.keras.Model([well_inputs, weather_inputs], outputs)
    else:
        well_model = BaseClassifier(num_well_features)
        well_inputs = tf.keras.Input(shape=(1440*num_well_features))
        well_features = well_model.encoder(well_inputs, training=False)
        x = tf.keras.layers.Dropout(0.2)(well_features)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        classification_model = tf.keras.Model(well_inputs, outputs)
    return classification_model
