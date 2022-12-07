from Dataset.DataManager import DataManager
from Dataset.Dataset import Dataset 
from utils.advancedanalytics_util import aauconnect_

class ROC:
    def __init__(self,
                 group_config:dict,
                 inference_config:dict, 
                 data_connection_config: dict,
                 roc_config:dict):
        
        self._validate_group_config(group_config)
        self.group_config = group_config 
        self._gp_info = self._get_group_info()
        self.inference_wells = [d['WELL_CD'] for d in self._gp_info]
        
        self._validate_inference_config(inference_config)
        self.inference_config = inference_config
        
        self._validate_data_connection_config(data_connection_config)
        self.data_connection_config = data_connection_config
        self._validate_roc_config(roc_config)
        self.roc_config = roc_config 
        
        self.dataset = Dataset(connection = aauconnect_(self.data_connection_config),**self.data_connection_config, **self.roc_config)
        self.data_manager = DataManager(wells= self.inference_wells,  dataset=self.dataset, **self.inference_config)
        
    def _validate_inference_config(self, config:dict)->None:
        assert("run_mode" in config), f"inference_info must contain keyword 'run_mode'"
        assert(config["run_mode"] in ["live", "backfill"]), f"inference_info's 'run_mode' must be one of ['live', 'backfill']"
        if config["run_mode"] == "back_fill":
            assert("backfill_start" in config), f"Backfill start date must be provided when running inference in backfill mode. inference_info dictionary must contain\
                'backfill_start' keyword."
            assert("backfill_date_format" in config), f"Keyword 'backfill_date_format' must be provided in inference_info dictionary"
        assert("inference_window" in config), f"inference_info must contain keyword 'inference_window'"
        assert("perform_model_training" in config), f"inference_info must contain keyword 'perform_model_training'"
        assert("perform_model_inference" in config), f"inference_info must contain keyword 'perform_model_inference'"
        assert("datetime_index_column" in config), f"inference_info must contain keyword 'datetime_index_column'"

    def _validate_general_connection_config(self, config:dict, parent_config:str)->None:
        assert("connection_type" in config), f"{parent_config} must contain keyword 'connection_type'"
        assert("path" in config), f"{parent_config}_info must contain keyword 'path'"
        assert("partition_mode" in config), f"{parent_config} must contain keyword 'partition_mode'"
    
    def _validate_data_connection_config(self, config:dict)->None:
        self._validate_general_connection_config(config, "data_connection_info")
        assert("file_prefix" in config), f"data_connection_info must contain keyword 'file_prefix'"
        assert("file_suffix" in config), f"data_connection_info must contain keyword 'file_suffix'"
             
    def _validate_roc_config(self,config:dict)->None:
        assert("features" in config), f"roc_info must contain keyword 'features'"
        assert("fill_method" in config), f"roc_info must contain keyword 'fill_method'"
        assert("datetime_index_column" in config), f"roc_info must contain keyword 'datetime_index_column'"

    def _validate_group_config(self,config:dict)->None:
        assert("group_connection_info" in config), f"group_info must contain keyword 'group_connection'"
        self._validate_general_connection_config(config["group_connection_info"],"group_info['group_connection_info']")
        assert("file" in config["group_connection_info"]), f"group_info['group_connection_info'] must contain keyword 'file'"
        assert("group_sql" in config), f"group_info must contain keyword 'group_sql'"
        assert("group_kwargs" in config), f"group_info must contain keyword 'group_kwargs'"
        assert("group_id" in config), f"group_info must contain keyword 'group_id'"
    
    def _get_group_info(self) -> dict:
        sql = self.group_config["group_sql"]
        args = {'GROUP_ID': self.group_config["group_id"]}
        kwargs = self.group_config["group_kwargs"]
        connection = aauconnect_(self.group_config["group_connection_info"])
        return connection.read(sql=sql, args=args, edit=[], orient='records', do_raise=True, **kwargs)

    def _store_event_data(self, df):
        well_cd = self._state['WELL_CD']
        sql = self._config['store_event_sql']
        args = df
        kwargs = self._config['store_event_kwargs']
        if 'file_prefix' in kwargs:
            kwargs['file_prefix'] = kwargs['file_prefix'].format(well_cd)
        return self._state['output_data_con'].write_many(sql=sql, args=args, edit=[], do_raise=True, **kwargs)

    def _store_time_data(self, df):
        well_cd = self._state['WELL_CD']
        sql = self._config['store_timedata_sql']
        args = df
        kwargs = self._config['store_timedata_kwargs']
        if 'file_prefix' in kwargs:
            kwargs['file_prefix'] = kwargs['file_prefix'].format(well_cd)
        return self._state['output_data_con'].write_many(sql=sql, args=args, edit=[], do_raise=True, **kwargs)

    def run_model_training(self):
        pass
    
    def _get_model_training_data(self):
        pass

    def get_nextrun(self, well_cd:str):
        pass

    def store_run_data(self, well_cd):
        next_run = self._state['processed_wells'][well_cd]['next_run']
        args = {}
        for key, value in next_run.items():
            args[key] = [value]
        sql = self._config['store_rundata_sql']
        kwargs = self._config['store_rundata_kwargs']
        if 'file' in kwargs:
            kwargs['file'] = kwargs['file'].format(well_cd)
        self._state['output_data_con'].write_many(sql=sql, args=args, edit=[], do_raise=True, **kwargs)
        self._state['processed_wells'][well_cd]['last_run'] = next_run
        self._state['processed_wells'][well_cd]['next_run'] = None

    def _write_log(self, record):
        sql = self._config['log_sql']
        kwargs = self._config['log_kwargs']
        self._state['log_con'].write(sql=sql, args=record, edit=[], do_raise=True, **kwargs)

    def get_flareline(self, well_cd):
        well_state = self._state['processed_wells'][well_cd]
        sql = self._config['flareline_sql']
        kwargs = self._config['flareline_kwargs']
        args = {'WELL_CD': well_cd, 'E': well_state['end']}
        df = self._state['meta_data_con'].read(sql=sql, args=args, edit=[], orient='df', do_raise=False, **kwargs)
        return df

    def get_tagdata(self, well_cd):
        well_state = self._state['processed_wells'][well_cd]
        sql = self._config['tag_data_sql']
        kwargs = self._config['tag_data_kwargs']
        if 'file_prefix' in kwargs:
            kwargs['file_prefix'] = kwargs['file_prefix'].format(well_cd)
            kwargs['start'] = well_state['start']
            kwargs['end'] = well_state['end']
        args = {'WELL_CD': well_cd, 'FLARELINE': well_state['flareline'], 'S': well_state['start'],
                'E': well_state['end']}
        df = self._state['input_data_con'].read(sql=sql, args=args, edit=[], orient='df', do_raise=False, **kwargs)
        return df

    def get_eventhistory(self, well_cd):
        well_state = self._state['processed_wells'][well_cd]
        sql = self._config['event_history_sql']
        kwargs = self._config['event_history_kwargs']
        if 'file_prefix' in kwargs:
            kwargs['file_prefix'] = kwargs['file_prefix'].format(well_cd)
            kwargs['start'] = well_state['start']
            kwargs['end'] = well_state['end']
        args = {'WELL_CD': well_cd, 'S': well_state['start'], 'E': well_state['end']}
        df = self._state['output_data_con'].read(sql=sql, args=args, edit=[], orient='df', do_raise=False, **kwargs)
        return df

    def run_model_inference(self, model=None, is_validation: bool = False) -> bool:
        logger.info("Running inference on BDTA model...")
        model_inference = ModelInference(self._state, self._config, model, is_validation)
        heuristic_inference = HeuristicInference(self._state, self._config, is_validation)
        for inference_well in self._gp_info:
            well_cd = inference_well['WELL_CD']
            self._state['WELL_CD'] = well_cd
            if well_cd in self._state['processed_wells']:
                data = self._state['processed_wells'][well_cd]['lastdata']
            else:
                self._state['processed_wells'][well_cd] = {}
                self._state['processed_wells'][well_cd]['lastdata'] = None
                self._state['processed_wells'][well_cd]['lastdataend'] = None
            next_run = self.get_nextrun(well_cd)
            proc_start = max(
                [t for t in [next_run['RUN_TSTART'], next_run['BDTA_LAST_RESULT_END'], next_run['BDTA_LAST_EVENT_END']]
                 if not pd.isnull(t)])
            proc_end = next_run['RUN_TEND']
            self._state['processed_wells'][well_cd]['start'] = proc_start
            self._state['processed_wells'][well_cd]['end'] = proc_end
            try:
                flareline_df = self.get_flareline(well_cd)
                if flareline_df is None or flareline_df.empty():
                    self._state['processed_wells'][well_cd]['flareline'] = ''
                else:
                    recs = flareline_df.to_dict(orient='records')
                    self._state['processed_wells'][well_cd]['flareline'] = recs[0]['FLARELINE']
                well_data_df = self.get_tagdata(well_cd)
                well_data_df.sort_values('TS', inplace=True)
                well_data_df.drop_duplicates(subset=['TS'], inplace=True)
                well_data_df.set_index('TS', inplace=True)
                well_data_df = well_data_df.asfreq('T')
                well_data_df[['WELL_CD']] = well_data_df[['WELL_CD']].fillna(value=well_cd)
                well_data_df[['FLARELINE']] = well_data_df[['FLARELINE']].fillna(
                    value=self._state['processed_wells'][well_cd]['flareline'])
                well_data_df.reset_index(inplace=True)
                event_hist_df = self.get_eventhistory(well_cd)
            except:
                continue
            model_inf_df = model_inference.run_model_inference(well_cd, well_data_df)
            heuristic_inf_ev_df, heuristic_inf_tm_df = heuristic_inference.run_inference(well_cd, well_data_df,
                                                                                         event_hist_df, model_inf_df)
            # ensure enough data is processed to overwrite the entire month files
            # include all events that overlap at the start or end of the month
            if not heuristic_inf_ev_df.empty:
                self._store_event_data(heuristic_inf_ev_df)
                next_run = self._state['processed_wells'][well_cd]['next_run']
                last_event_end = heuristic_inf_ev_df['EVENT_TS_MAXIMUM'].max()
                last_event_number = heuristic_inf_ev_df['BDTA_EVENT_NUMBER_COMBINED'].max()
                next_run['BDTA_LAST_EVENT_END'] = last_event_end
                next_run['BDTA_LAST_EVENT_NUMBER'] = last_event_number
                self._state['processed_wells'][well_cd]['next_run'] = next_run
            if not heuristic_inf_tm_df.empty:
                print('Timdata')
                print(heuristic_inf_tm_df)
                print(heuristic_inf_tm_df.columns)
                self._store_time_data(heuristic_inf_tm_df)
            self.store_run_data(well_cd)
        return True

if __name__ == "__main__":
    import config.__config__ as base_config
    import config.__state__ as base_state
    config = base_config.init()
    model = ROC(config['group_info'],config['inference_info'],config['data_connection_info'],config['roc_info'])
    data = model.data_manager.get_inference_dataset()
    print("END")