from typing import Union

import pandas as pd

from src.generators.base import DataParser
from src.generators.well_data import WellDataLoader
from src.operators.connection import ConnectionOperator
from utils.logging import get_logger

logger = get_logger(__name__)


class TrainingDataParser(DataParser):
    def __init__(self, connection_operator: ConnectionOperator, state, config):
        super().__init__(connection_operator, state, config)
        self._training_wells = tuple(self._config['training_default_vars']['training_wells'])

    def get_well_data_df(self) -> Union[pd.DataFrame, pd.Series]:
        # get label data from training wells and put into a single dataframe
        logger.info("Loading all label data for training wells")
        # label_data_file_list = self._get_label_file_list()
        train_label_df = self._create_train_label_df()

        # get tag data from training wells and put into a single dataframe
        logger.info("Loading all tag data for training wells")
        tag_data_file_list = self._get_tag_file_list(inference_type="training")
        well_data_loader = WellDataLoader(self._connection_operator, self._state, self._config)
        train_tag_df = pd.concat([well_data_loader.get_well_tag_data(well=training_well,
                                                                     tag_data_file_list=tag_data_file_list,
                                                                     tags=self._config['training_default_vars'][
                                                                         'feature_tags'],
                                                                     window_start_datetime=
                                                                     self._config['training_default_vars'][
                                                                         'train_data_start_datetime'],
                                                                     window_end_datetime=
                                                                     self._config['training_default_vars'][
                                                                         'train_data_end_datetime'])
                                  for training_well in self._config['training_default_vars']['training_wells']])

        # amend labels to tag data
        logger.info("Amending labels to train_tag_df")
        train_well_data_ls = []
        for training_well in self._config['training_default_vars']['training_wells']:
            tmp_train_well_data_df = train_tag_df[train_tag_df['WELL_CD'] == training_well]
            tmp_train_well_data_df.loc[:, 'TRAIN_LABEL'] = 0

            events_per_well = train_label_df[
                (train_label_df['WELL_CD'] == training_well) & (
                    train_label_df[self._config['training_default_vars']['train_label_tag']].str.contains('Accept'))][
                ['BDTA_START_TS', 'BDTA_END_TS']]

            for event_start, event_end in (zip(events_per_well['BDTA_START_TS'], events_per_well['BDTA_END_TS'])):
                tmp_train_well_data_df.loc[event_start:event_end, 'TRAIN_LABEL'] = 1
            train_well_data_ls.append(tmp_train_well_data_df)
        train_well_data_df = pd.concat(train_well_data_ls)

        logger.info(
            f"Created train_well_data_df: {train_tag_df.head()} and train_well_set: {set(self._config['training_default_vars']['training_wells'])}")
        return train_well_data_df

    def _create_train_label_df(self) -> pd.DataFrame:
        train_label_df = pd.DataFrame()
        for train_well in self._config['training_default_vars']['training_wells']:
            train_label_df = pd.concat([train_label_df, self._create_label_df(train_well)])
        return train_label_df
