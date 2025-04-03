import pandas as pd

def create_dim_building(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DIM_BUILDING table with a surrogate key from the merged dataset.
    """
    dim = df[['Building']].drop_duplicates().reset_index(drop=True)
    dim['buildingID'] = dim.index + 1
    dim.rename(columns={'Building': 'buildingName'}, inplace=True)
    return dim[['buildingID', 'buildingName']]


def create_dim_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DIM_TIME table by extracting unique timestamps and adding time breakdowns.
    """
    dim = df[['Timestamp']].drop_duplicates().reset_index(drop=True)
    dim['timeID'] = dim.index + 1
    dim['Year'] = dim['Timestamp'].dt.year
    dim['Month'] = dim['Timestamp'].dt.month
    dim['Day'] = dim['Timestamp'].dt.day
    dim['Hour'] = dim['Timestamp'].dt.hour
    dim['Minute'] = dim['Timestamp'].dt.minute
    dim['Second'] = dim['Timestamp'].dt.second
    return dim[['timeID', 'Timestamp', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]


def create_fact_measurements(df: pd.DataFrame, dim_building: pd.DataFrame, dim_time: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a FACT_MEASUREMENTS table by mapping each fact row to the corresponding
    building and time IDs from the dimension tables.
    """
    fact = df.merge(dim_building, left_on='Building', right_on='buildingName', how='left')
    fact = fact.merge(dim_time[['timeID', 'Timestamp']], on='Timestamp', how='left')
    fact['measurementID'] = fact.index + 1
    return fact[['measurementID', 'buildingID', 'timeID', 'SupplyTemp', 'ReturnTemp', 'SupplyTempReturnTempDerivation', 'OutsideTemp', 'SetbackActive']]