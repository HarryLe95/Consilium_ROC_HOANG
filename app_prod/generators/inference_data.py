from typing import Tuple, Set
import pandas as pd
from src.generators.base import DataParser
from src.generators.well_data import WellDataLoader
from utils.logging import get_logger

logger = get_logger(__name__)

class InferenceDataParser(DataParser):
    def __init__(self, state: dict, config: dict, connection_operator, is_validation: bool = False):
        super().__init__(state, config, connection_operator)
        self._is_validation = is_validation
        self._well_data_loader = WellDataLoader(self._state, self._config, self._connection_operator)

    def get_well_data_df(self, state: dict, well: str) -> Tuple[pd.DataFrame, Set]:
        # get tag data from training wells and put into a single dataframe
        logger.info("Loading all tag data for inference wells")
        self._state = state
        inference_tag_df = self._well_data_loader.get_well_tag_data(well=well, tags=self._config['feature_tags'])
        if self._is_validation:
            # get label data from training wells and put into a single dataframe
            logger.info("Loading all label data for inference wells for comparison")
            inference_label_df = self._create_label_df(well)
            # amend labels to tag data
            logger.info("Amending labels to inference_tag_df")
            for comparison_label in self._config['inference_default_vars']['inference_comparison_labels']:
                inference_tag_df[comparison_label] = 0
                if inference_label_df is not None:
                    if comparison_label == 'REVIEW_STATUS':
                        events_per_well = \
                            inference_label_df[(inference_label_df[comparison_label].str.contains('Accept'))][
                                ['BDTA_START_TS', 'BDTA_END_TS']]
                    elif comparison_label == 'QC_REVIEW_STATUS':
                        events_per_well = inference_label_df[['BDTA_START_TS', 'BDTA_END_TS']]

                    for event_start, event_end in (
                            zip(events_per_well['BDTA_START_TS'], events_per_well['BDTA_END_TS'])):
                        inference_tag_df.loc[event_start:event_end, comparison_label] = 1
        return inference_tag_df
