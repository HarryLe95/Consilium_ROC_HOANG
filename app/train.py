from src.utils.Setup import logger_init, get_argparse
from src.ROC_Classifier.Trainer import ROC_Classifier
from src.ROC_Classifier.DataGenerator import ROC_Generator
from copy import deepcopy 
import numpy as np 
import tensorflow as tf

def main():
    args = get_argparse()
    args.train_wells = deepcopy(args.all_wells)
    for well in args.test_wells:
        args.train_wells.remove(well)
    args.num_classes = len(np.unique(np.array(list(args.label_mapping.values()))))
   
    #Init Logger and save folder
    logger_init(args)
    
    data_generator = ROC_Generator(wells=args.train_wells, features=args.features, num_days = args.num_days, normalise_mode=args.normalise_mode,
                                    label_mapping=args.label_mapping, drop_labels = args.drop_labels,
                                    split=True, split_ratio=0.8, num_classes=args.num_classes,
                                    last_day=args.last_day)
    
    data_generator.setup()

    test_generator = ROC_Generator(wells=args.test_wells, features=args.features, num_days = args.num_days, normalise_mode=args.normalise_mode,
                                    label_mapping=args.label_mapping, drop_labels = args.drop_labels,
                                    num_classes=args.num_classes, last_day = args.last_day)
    
    test_generator.setup()

    args.num_well_features = args.num_days*len(data_generator.well_features)
    args.num_weather_features = args.num_days*len(data_generator.weather_features) if data_generator.weather_features is not None else 0 

    trainer = ROC_Classifier(num_classes=args.num_classes, 
                             num_well_features=args.num_well_features, 
                             num_weather_features=args.num_weather_features,
                             optimiser=args.optimiser, base_lr=args.base_lr, early_stopping_patience=args.early_stopping_patience,
                             reduce_lr_patience=args.reduce_lr_patience, num_epochs = args.num_epochs, metrics = args.metrics, use_pretrain=args.use_pretrain,
                             pretrain_model=args.pretrain_model, save_model=args.save_model, save_name=args.save_name)
    trainer.setup()
    tf.keras.utils.plot_model(trainer.model,'Weather_Classification_Model.png')
    history = trainer.fit(x = data_generator.train_image, 
                          y = data_generator.train_label, 
                          validation_data=(data_generator.val_image, data_generator.val_label),
                          batch_size = args.batch_size)

    trainer.evaluate_binary(data_generator.val_image, data_generator.val_label, data_generator.TS[1])
    trainer.evaluate_binary(test_generator.image, test_generator.label, test_generator.TS)
    return trainer
    
if __name__ == "__main__":
    main()