import pandas as pd


def load_measurement(file_path: str, new_value_col: str) -> pd.DataFrame:
    """
    Loads a CSV file with columns '_time' and '_value', renames them,
    converts _time to datetime, and returns a DataFrame with Timestamp and new_value_col.
    """
    df = pd.read_csv(
        file_path,
        parse_dates=["_time"],
        date_format="ISO8601"
    )
    df.rename(columns={"_value": new_value_col, "_time": "Timestamp"}, inplace=True)
    return df[["Timestamp", new_value_col]]


def load_building_data(building: str, input_dir_name: str, file_prefix: str) -> pd.DataFrame:
    """
    Loads and merges the supply, return, outside, and ground truth CSVs for one building.
    file_prefix should be the common part of the filename (e.g., "Building 2").
    """
    supply = load_measurement(f"{input_dir_name}{file_prefix} supply temperature.csv", "SupplyTemp")
    ret = load_measurement(f"{input_dir_name}{file_prefix} return temperature.csv", "ReturnTemp")
    outside = load_measurement(f"{input_dir_name}{file_prefix} outside temperature.csv", "OutsideTemp")
    ground_truth = load_measurement(f"{input_dir_name}{file_prefix} ground truth.csv", "SetbackActive")

    # Merge on Timestamp using outer joins to preserve all records
    merged = supply.merge(ret, on="Timestamp", how="outer") \
        .merge(outside, on="Timestamp", how="outer") \
        .merge(ground_truth, on="Timestamp", how="outer")
    merged["Building"] = building
    return merged