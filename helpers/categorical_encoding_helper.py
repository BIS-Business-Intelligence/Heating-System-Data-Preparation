import numpy as np


def add_categorical_encoding_features(df, timestamp_col="Timestamp"):
    """
    Adds categorical encoding columns for month, hour, and minute from a timestamp column.
    Columns added: Month, Hour, Minute
    """
    ts = df[timestamp_col]
    df["Month"] = ts.dt.month.astype("category")
    df["Hour"] = ts.dt.hour.astype("category")
    df["Minute"] = ts.dt.minute.astype("category")
    df["DayOfWeek"] = ts.dt.dayofweek.astype("category")
    df["IsWeekend"] = (ts.dt.dayofweek >= 5).astype("category")  # 5=Saturday, 6=Sunday
    return df

