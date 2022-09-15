from app.src.utils.PathManager import Paths as Path 
from app.src.utils.Data import create_label_dataframe
import yaml 

if __name__ == "__main__":
    with open(Path.config("well_processing_config.yaml"),'r') as file:
        config = yaml.safe_load(file)[0]
    all_wells = config['all_wells']
    processed_wells = config['processed_wells']
    
    for well in all_wells:
        try:
            if well in processed_wells:
                continue
            print(f"Processing well: {well}")
            create_label_dataframe(well,save_pickle=True)
            processed_wells.append(well)
        except Exception as e:
            print(f"Error encountered when processing well {well}.")
            print(e)
            
    with open(Path.config("well_processing_config.yaml"),'w') as file:
        yaml.dump([config],file,default_flow_style=False)
    
    print("End")