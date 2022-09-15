from src.utils.PathManager import Paths as Path
import pandas as pd 
import yaml 
from geopy import distance 
import numpy as np 

def processWellLocation(file:str='well_location.csv'):
    well_loc_df = pd.read_csv(Path.data(file))
    well_loc_df['COORD'] = well_loc_df.apply(lambda x: (x.LATITUDE, x.LONGITUDE), axis=1)
    well_loc_df.to_csv(Path.data(file),index=False)
    well_loc_df.set_index("WELL_CD", inplace=True)
    well_loc_df = well_loc_df.loc[:,['COORD']]
    return well_loc_df
    
def processStationLocation(file:str='solar_location.csv'):
    station_loc_df = pd.read_csv(Path.data(file))
    if np.any(station_loc_df.Long.values < 0):
        rename_dict = {'Lat': "Long", "Long": "Lat"}
        station_loc_df.rename(columns=rename_dict, inplace=True)
    station_loc_df['COORD'] = station_loc_df.apply(lambda x: (x.Lat, x.Long), axis=1)
    station_loc_df.to_csv(Path.data(file),index=False)
    station_loc_df.set_index("Location", inplace=True)
    station_loc_df = station_loc_df.loc[:,["COORD"]]
    return station_loc_df
    
def getWellLocation(file:str="well_location.csv") -> pd.DataFrame :
    """Get well location dataframe

    Args:
        file (str, optional): well location file name. Defaults to "well_location.csv".

    Returns:
        pd.DataFrame: dataframe whose index is WELL_CD and whose column is COORD
    """
    try:
        df = pd.read_csv(Path.data(file), index_col = "WELL_CD", usecols =["WELL_CD", "COORD"])
        df['COORD'] = df.eval(df['COORD'])
        return df
    except Exception as e:
        print(f"Error encountered: {e}")
        return processWellLocation(file)

def getStationLocation(file:str='solar_location.csv') -> pd.DataFrame :
    """Get station location dataframe 

    Args:
        file (str, optional): station location file name. Defaults to 'solar_location.csv'.

    Returns:
        pd.DataFrame: dataframe whose index is Location and whose column is COORD
    """
    try:
        df = pd.read_csv(Path.data(file), index_col='Location', usecols=['Location', 'COORD'])
        df['COORD'] = df.eval(df['COORD'])
        return df 
    except Exception as e:
        print(f"Error encountered: {e}")
        return processStationLocation(file)
        
        
def getDistance(x_coord: tuple, y_coord: tuple):
    return distance.geodesic(x_coord, y_coord).kilometers

def getDistanceMatrix(first: pd.DataFrame, second: pd.DataFrame, first_coord_column:str="COORD", second_coord_column:str="COORD") -> pd.DataFrame: 
    """Generate df of the distance between every pair of the first and second group.
    
    The final_df is of size (MxN) where M and N are the numbers of row and column of the first and second dataframe respectively. 
    The index and columns of final_df are the indices of the first and second dataframes.

    Args:
        first (pd.DataFrame): first group df, should have a coordinate column containing (lat,long) tuple, and index containing the well's name
        second (pd.DataFrame): second group df, should havea coordinate column containing (lat,long) tuple and index containing the well's name
        first_coord_column (str, optional): name of coordinate column in the first df. Defaults to "COORD".
        second_coord_column (str, optional): name of the coordinate column in the second df. Defaults to "COORD".

    Returns:
        pd.DataFrame: final_df containing the pair-wise distance between first and second 
    """
    second_grid, first_grid = np.meshgrid(second[second_coord_column].values, first[first_coord_column].values)
    f = np.vectorize(getDistance)
    distance_grid = f(first_grid, second_grid)
    final_df = pd.DataFrame(data=distance_grid, index=first.index, columns=second.index)
    return final_df

def getNearestNeighbor(distance_df:pd.DataFrame) -> pd.DataFrame:
    """Get df containing the nearest neighbors whose distances are specified in distance_df
    
    Args:
        distance_df (pd.DataFrame): distance_df[row, col] contains the distance in KM between the object described in row index and the object described in 
        col index.
    
    Returns: 
        pd.DataFrame: final_df containing the name of the nearest col for each row.
    """ 
    final_df = distance_df.idxmin(axis='columns')

    with open(Path.config("nearest_station.yaml"),'w') as file:
        yaml.dump(final_df.to_dict(), file)

    return final_df

if __name__ == "__main__":
    well_loc = getWellLocation()
    station_loc = processStationLocation()
    
    with open(Path.config("meta_train_config.yaml"), 'r') as file:
        config = yaml.safe_load(file)[0]
        all_wells = config['all_wells']
    well_loc = well_loc.loc[all_wells,:]
    distance_matrix = getDistanceMatrix(well_loc, station_loc)
    getNearestNeighbor(distance_matrix)
    print("End")