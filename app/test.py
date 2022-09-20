from src.utils.Setup import logger_init, get_argparse
from src.ROC_Classifier.Trainer import ROC_Classifier 
from src.ROC_Classifier.DataGenerator import ROC_Generator
from copy import deepcopy 
import numpy as np 

def main():
    args = get_argparse()
    args.train_wells = deepcopy(args.all_wells)
    for well in args.test_wells:
        args.train_wells.remove(well)
    args.num_classes = len(np.unique(np.array(list(args.label_mapping.values()))))
    args.num_features = args.num_days*len(args.features)
    
    #Init Logger and save folder
    logger_init(args)
    
    data_generator = ROC_Generator(args.test_wells, features=args.features, num_days = args.num_days, normalise_mode=args.normalise_mode,
                                    label_mapping=args.label_mapping, drop_labels = args.drop_labels,
                                    batch_size=args.batch_size, num_classes=args.num_classes)
    
    data_generator.setup()
    test_dataset = data_generator.dataset
    TS = data_generator.TS
    trainer = ROC_Classifier(args.num_classes, num_features=args.num_features, optimiser=args.optimiser, base_lr=args.base_lr, early_stopping_patience=args.early_stopping_patience,
                             reduce_lr_patience=args.reduce_lr_patience, num_epochs = args.num_epochs, metrics = args.metrics, use_pretrain=args.use_pretrain,
                             pretrain_model=args.pretrain_model, save_model=args.save_model, save_name=args.save_name)
    trainer.setup()
    trainer.evaluate_binary(test_dataset, TS)
    
if __name__ == "__main__":
    main()