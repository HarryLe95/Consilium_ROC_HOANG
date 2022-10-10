from logging.handlers import DatagramHandler
from src.utils.Setup import logger_init, get_argparse
from src.ROC_Classifier.Trainer import ROC_Classifier, ROC_Regressor
from src.ROC_Classifier.DataGenerator import ROC_Generator
from copy import deepcopy 
import numpy as np 
from src.utils.PathManager import Paths as Path

def main():
    args = get_argparse()
    args.train_wells = deepcopy(args.all_wells)
    for well in args.test_wells:
        args.train_wells.remove(well)
    args.num_features = args.num_days*len(args.features)
    
    #Init Logger and save folder
    logger_init(args)
    
    data_generator = ROC_Generator(args.train_wells, features=args.features, num_days = args.num_days, normalise_mode=args.normalise_mode,
                                    split=True, split_ratio=0.8)
    
    data_generator.setup_regression()
    args.num_well_features = args.num_days*len(data_generator.well_features)
    args.num_weather_features = args.num_days*len(data_generator.weather_features) if data_generator.weather_features is not None else 0 
    trainer = ROC_Regressor(num_well_features=args.num_well_features, 
                            num_weather_features=args.num_weather_features,
                            optimiser=args.optimiser, base_lr=args.base_lr, early_stopping_patience=args.early_stopping_patience,
                            reduce_lr_patience=args.reduce_lr_patience, num_epochs = args.num_epochs, metrics = args.metrics, use_pretrain=args.use_pretrain,
                            pretrain_model=args.pretrain_model, save_model=args.save_model, save_name=args.save_name)
    trainer.setup()
    history = trainer.fit(x = data_generator.train_image, 
                          y = data_generator.train_label, 
                          validation_data=(data_generator.val_image, data_generator.val_label),
                          batch_size = args.batch_size)
    
    data_generator = ROC_Generator(args.test_wells, features=args.features, num_days = args.num_days, normalise_mode=args.normalise_mode)
    
    data_generator.setup_regression()
    TS = data_generator.TS
    
    loss = trainer.evaluate(data_generator.image, data_generator.label)
    predictions = trainer.predict(data_generator.image,TS)
    predictions.to_pickle(Path.data(f"{args.test_wells[0]}_Regression_Output.pkl"))
    print("End")
    
    
if __name__ == "__main__":
    main()