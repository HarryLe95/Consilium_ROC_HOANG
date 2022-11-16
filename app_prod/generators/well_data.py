import pandas as pd
from utils.logging import get_logger

logger = get_logger(__name__)

class WellDataLoader:
    def __init__(self, state, config, connection_operator):
        self._connection_operator = connection_operator
        self._state = state
        self._config = config

    def get_well_tag_data(self, well: str, tags: list) -> pd.DataFrame:
        print ('get_well_tag_data', well)
        well_data_df = self._create_well_data_df(tags, well)
        # remove anomalous data points?
        if self._config['perform_err_correction']:
            logger.info("Performing anomalous data error correction")
            for tag in tags:
                well_data_df[tag][well_data_df[tag] < 0] = None
                well_data_df[tag][well_data_df[tag] > well_data_df[tag].median() * 10] = None
                if tag == 'PRES_TH':
                    well_data_df[tag][well_data_df[tag] == 0] = None
        # interpolate?
        if self._config['perform_interpolation']:
            well_data_df = well_data_df.interpolate(
                method=self._config['interpolation_method'],
                axis=0,
                limit=self._config['interpolation_forward_limit'])
        # normalise?
        if self._config['perform_normalisation']:
            logger.info("Normalising dataframe")
            for tag in tags:
                well_data_df[f"{tag}_NORM"] = well_data_df[tag] / well_data_df[tag].max()
        # fill nans?
        if self._config['perform_fillna']:
            well_data_df = well_data_df.fillna(self._config['fillna_value'])
        return well_data_df

    def _create_well_data_df(self, tags: list, well: str) -> pd.DataFrame:
        well_data_df = self._get_tag_dataframe(well)
        well_data_df.sort_values('TS', inplace=True)
        well_data_df.set_index('TS', inplace=True)
        well_data_df = well_data_df.asfreq('T')
        well_data_df = well_data_df[tags]
        well_data_df['WELL_CD'] = well
        return well_data_df

    def _get_tag_dataframe(self, well: str) -> pd.DataFrame:
        well_data_df = pd.DataFrame()
        sql = self._config['tag_data_sql']
        args = {}
        args['W'] = self._state['WELL_CD']
        args['S'] = self._state["start"]
        args['E'] = self._state["end"]
        kwargs = self._config['tag_data_kwargs']
        if 'W' in kwargs:
            kwargs['W'] = self._state['WELL_CD']
        if 'file_prefix' in kwargs:
            kwargs['file_prefix'] = kwargs['file_prefix'].format(self._state['WELL_CD'])
        kwargs['start'] = self._state["start"]
        kwargs['end'] = self._state["end"]
        try:
            well_data_df = self._state['tag_data_con'].read(sql=sql, args=args, edit=[], orient='df', do_raise=False, **kwargs)
        except Exception as e:
            logger.info(f"Error caught: {e}")
        return well_data_df
